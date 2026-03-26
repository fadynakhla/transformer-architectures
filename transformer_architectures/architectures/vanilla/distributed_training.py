from typing import Any, Callable, ContextManager, Optional
import functools
import math

import mlflow
import pydantic
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from nltk.translate import (  # pyright: ignore[reportMissingTypeStubs]
    bleu_score,
    gleu_score,
)

from transformer_architectures import config
from transformer_architectures.architectures import vanilla
from transformer_architectures.architectures.vanilla import data, datamodule
from transformer_architectures.training import base_train_config, distributed

IGNORE_ID = -100


class TrainingConfig(base_train_config.BaseTrainConfig):
    learning_rate: float
    warmup_steps: int
    label_smoothing: float
    # num_samples: int
    # Token-budget bucketing. When set, batch_size is ignored for the train dataloader
    # and batches are sized to contain ~token_budget real tokens instead.
    token_budget: Optional[int] = None
    # Window size for windowed sort (see TokenBudgetBatchSampler). None = global sort.
    sort_window: Optional[int] = None

    def batch_config(self) -> dict[str, Any]:
        if self.token_budget is not None:
            return {
                "batch_sampling": "token_budget",
                "per_device_token_budget": self.token_budget,
                "token_budget": self.token_budget * self.grad_accumulation_steps,
                "sort_window": self.sort_window,
            }
        return {
            "batch_sampling": "fixed",
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "batch_size": self.per_device_train_batch_size
            * self.grad_accumulation_steps,
        }

    @pydantic.field_validator("eval_metric")
    @classmethod
    def _check_eval_metric(cls, v: str) -> str:
        allowed = {"bleu", "gleu", "avg_loss"}
        if v not in allowed:
            raise ValueError(f"eval_metric must be one of {allowed}, got '{v}'")
        return v


class ModelConfig(pydantic.BaseModel):
    encoding: str = "r50k_base"
    model_max_len: int = 512
    num_stacks: int = 6
    embed_dim: int = 1024
    num_heads: int = 16
    ff_dim: int = 4096
    dropout: float = 0.3


class TrainableTransformer(distributed.TrainableArchitecture[TrainingConfig]):
    architecture_name: str = "vanilla_transformer"

    def __init__(
        self,
        train_config: TrainingConfig,
        model_config: ModelConfig,
        dataset_config: data.DatasetConfig,
        mlflow_config: base_train_config.MLFlowConfig,
    ) -> None:
        self.train_config = train_config
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.mlflow_config = mlflow_config

        self.max_bleu = 0.0
        self.max_gleu = 0.0

    def build_model(self) -> vanilla.Transformer:
        return vanilla.Transformer(
            vocab_size=self.tokenizer.vocab_size,
            num_stacks=self.model_config.num_stacks,
            embed_dim=self.model_config.embed_dim,
            num_heads=self.model_config.num_heads,
            ff_dim=self.model_config.ff_dim,
            dropout=self.model_config.dropout,
        )

    def build_datamodule(self) -> datamodule.TransformerDataModule:
        return datamodule.TransformerDataModule(
            dataset_config=self.dataset_config,
            tokenizer=self.tokenizer,
            per_device_train_batch_size=self.train_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.train_config.per_device_eval_batch_size,
        )

    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(
            model.parameters(),
            lr=self.train_config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

    def build_scheduler(
        self, optimizer: optim.Optimizer, steps_per_epoch: int
    ) -> LRScheduler:
        total_steps = steps_per_epoch * self.train_config.epochs
        return optim.lr_scheduler.LambdaLR(
            optimizer, make_cosine_schedule(self.train_config.warmup_steps, total_steps)
        )

    def build_criterion(self) -> nn.CrossEntropyLoss:
        return nn.CrossEntropyLoss(
            ignore_index=IGNORE_ID, label_smoothing=self.train_config.label_smoothing
        )

    def make_run_params(self) -> dict[str, Any]:
        return {
            "grad_accumulation_steps": self.train_config.grad_accumulation_steps,
            "learning_rate": self.train_config.learning_rate,
            "label_smoothing": self.train_config.label_smoothing,
            "model_config": self.model_config.model_dump(),
            "batch_config": self.train_config.batch_config(),
        }

    def train_step(
        self,
        model: nn.Module,
        batch: data.LabeledBatch,
        criterion: nn.Module,
        autocast_ctx: ContextManager,
    ) -> torch.Tensor:
        with autocast_ctx:
            predictions = model(
                encoder_input=batch.input_ids,
                encoder_attention_mask=batch.attention_mask,
                decoder_input=batch.decoder_input_ids,
                decoder_attention_mask=batch.decoder_attention_mask,
            )
            loss: torch.Tensor = criterion(
                predictions.view(-1, predictions.size(-1)),
                batch.target.view(-1),
            )
        return loss

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        data_module: distributed.DataModule,
        criterion: nn.Module,
        autocast_ctx: ContextManager,
        stage: str,
        epoch: int,
        global_step: int,
        distributed_ctx: distributed.DistributedContext
    ) -> dict[str, float]:
        total_loss = 0.0

        hypotheses: list[list[str]] = []
        references: list[list[list[str]]] = []

        dataloader = data_module.dataloader(stage)
        for i, batch in enumerate(dataloader):
            batch.to(distributed_ctx.device)
            with autocast_ctx:
                predictions = model(
                    encoder_input=batch.input_ids,
                    encoder_attention_mask=batch.attention_mask,
                    decoder_input=batch.decoder_input_ids,
                    decoder_attention_mask=batch.decoder_attention_mask,
                )
                loss: torch.Tensor = criterion(
                    predictions.view(-1, predictions.size(-1)),
                    batch.target.view(-1),
                )
            total_loss += loss.item()
            predicted_sequences = torch.argmax(predictions, dim=-1)
            predicted_sequences = torch.masked_fill(
                predicted_sequences, batch.target == IGNORE_ID, 0
            )
            hypotheses.extend(
                [p.split() for p in data_module.tokenizer.batch_decode(predicted_sequences)]
            )
            targets = torch.masked_fill(batch.target, batch.target == IGNORE_ID, 0)
            references.extend(
                [[t.split()] for t in data_module.tokenizer.batch_decode(targets)]
            )
            if distributed_ctx.is_head and i == 0 and (stage == "test" or epoch % 5 == 0):
                for j, (h, r) in enumerate(zip(hypotheses, references)):
                    mlflow.log_text(
                        text=f"{h}",
                        artifact_file=f"{stage}/epoch_{epoch}/sample{j}/hypothesis.txt",
                    )
                    mlflow.log_text(
                        text=f"{r}",
                        artifact_file=f"{stage}/epoch_{epoch}/sample{j}/references.txt",
                    )

        avg_loss = total_loss / len(dataloader)
        gleu = float(gleu_score.corpus_gleu(references, hypotheses))
        bleu = float(bleu_score.corpus_bleu(references, hypotheses))  # type: ignore
        if distributed_ctx.is_head:
            mlflow.log_metrics(
                {f"{stage}_loss": avg_loss, f"{stage}_gleu": gleu, f"{stage}_bleu": bleu},
                step=global_step,
            )
        return {"avg_loss": avg_loss, "gleu": gleu, "bleu": bleu}



    @functools.cached_property
    def tokenizer(self) -> vanilla.Tokenizer:
        return make_tokenizer(self.model_config)

    @classmethod
    def from_yaml_config(cls, path: str) -> "TrainableTransformer":
        model_config = config.load_config(
            path, section="Model", model_class=ModelConfig
        )
        train_config = config.load_config(
            path, section="Training", model_class=TrainingConfig
        )
        dataset_config = config.load_config(
            path, section="Dataset", model_class=data.DatasetConfig
        )
        mlflow_config = config.load_config(
            path, section="MLFlow", model_class=base_train_config.MLFlowConfig
        )
        return cls(train_config, model_config, dataset_config, mlflow_config)


def make_tokenizer(config: ModelConfig) -> vanilla.Tokenizer:
    return vanilla.Tokenizer(
        config.encoding,
        config.model_max_len,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def make_schedule(model_size: int, warmup_steps: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        if step == 0:
            step = 1
        return model_size**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)

    return schedule


def make_cosine_schedule(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (
            1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))
        )

    return schedule
