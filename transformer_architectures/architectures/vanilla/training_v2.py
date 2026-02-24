from typing import Any, Callable, Literal, ContextManager, Optional
import math
from contextlib import nullcontext

import loguru
import mlflow
import pydantic
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from nltk.translate import (  # pyright: ignore[reportMissingTypeStubs]
    bleu_score,
    gleu_score,
)

from transformer_architectures import config
from transformer_architectures.architectures import vanilla
from transformer_architectures.datasets import wmt_en_fr
from transformer_architectures.training import checkpointing, grad_logging

IGNORE_ID = -100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME = "vanilla_transformer_large"


logger = loguru.logger


mlflow.set_tracking_uri("http://10.9.9.249:5000")
mlflow.set_experiment("Vanilla Transformer Large - ENFR Test1")
mlflow.config.enable_system_metrics_logging()  # pyright: ignore[reportPrivateImportUsage]
mlflow.config.set_system_metrics_sampling_interval( # pyright: ignore[reportPrivateImportUsage]
    30
)


CONFIG_PATH = "configs/vanilla_large.yaml"


class TrainingConfig(pydantic.BaseModel):
    batch_size: int
    grad_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    epochs: int
    label_smoothing: float
    num_samples: int
    log_interval: int = 25
    precision: Literal["fp32", "bf16"] = "bf16"
    # Token-budget bucketing. When set, batch_size is ignored for the train dataloader
    # and batches are sized to contain ~token_budget real tokens instead.
    token_budget: Optional[int] = None
    # Window size for windowed sort (see TokenBudgetBatchSampler). None = global sort.
    sort_window: Optional[int] = None


class ModelConfig(pydantic.BaseModel):
    encoding: str = "r50k_base"
    model_max_len: int = 512
    num_stacks: int = 6
    embed_dim: int = 1024
    num_heads: int = 16
    ff_dim: int = 4096
    dropout: float = 0.3


model_config = config.load_config(CONFIG_PATH, section="Model", model_class=ModelConfig)
train_config = config.load_config(
    CONFIG_PATH, section="Training", model_class=TrainingConfig
)

params: dict[str, int | float | dict[str, Any]] = {
    "batch_size": train_config.batch_size * train_config.grad_accumulation_steps,
    "grad_accumulation_steps": train_config.grad_accumulation_steps,
    "learning_rate": train_config.learning_rate,
    "label_smoothing": train_config.label_smoothing,
    "model_config": model_config.model_dump(),
}


def train() -> None:
    tokenizer, model = make_tokenizer_and_model(model_config)
    data_module = vanilla.TransformerDataModule(
        data=load_data(train_config.num_samples),
        tokenizer=tokenizer,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        test_split=0.05,
        val_split=0.05,
        token_budget=train_config.token_budget,
        sort_window=train_config.sort_window,
    )
    data_module.setup()
    with mlflow.start_run():
        mlflow.log_params(params=params)
        mlflow.log_text(
            text=f"{data_module.train_dataset[0]}", artifact_file="sample_batch.txt"
        )
        model = model.to(DEVICE)
        autocast_ctx = make_autocast_ctx(train_config.precision)
        criterion = nn.CrossEntropyLoss(
            ignore_index=IGNORE_ID, label_smoothing=train_config.label_smoothing
        )
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        epoch_steps = math.ceil(
            len(data_module.train_dataloader()) / train_config.grad_accumulation_steps
        )
        total_steps = train_config.epochs * epoch_steps
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, make_cosine_schedule(train_config.warmup_steps, total_steps)
        )
        global_step = 0
        max_bleu = 0.0
        for epoch in range(train_config.epochs):
            global_step = train_epoch(
                model,
                data_module,
                criterion,
                optimizer,
                scheduler,
                train_config.grad_accumulation_steps,
                epoch,
                global_step,
                train_config.log_interval,
                autocast_ctx,
            )
            bleu, _ = evaluate(model, data_module, criterion, autocast_ctx, stage="val", epoch=epoch, global_step=global_step)
            if bleu > max_bleu:
                logger.info(f"New best BLEU score: {bleu}. Saving checkpoint.")
                max_bleu = bleu
                checkpointing.save_checkpoint(
                    model, optimizer, scheduler, data_module.generator, NAME, epoch, global_step
                )
        evaluate(model, data_module, criterion, autocast_ctx, stage="test", epoch=train_config.epochs, global_step=global_step)


def train_epoch(
    model: vanilla.Transformer,
    data_module: vanilla.TransformerDataModule,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    gradient_accumulation_steps: int,
    epoch: int,
    global_step: int,
    log_interval: int,
    autocast_ctx: ContextManager,
) -> int:
    model.train()

    dataloader = data_module.train_dataloader()
    accumulated_loss = 0.0
    accumulated_batches = 0

    optimizer_steps = math.ceil(len(dataloader) / gradient_accumulation_steps)
    progress_bar = tqdm.tqdm(total=optimizer_steps, desc=f"Epoch {epoch}")

    def _step() -> None:
        nonlocal accumulated_loss, accumulated_batches, global_step
        optimizer.step()
        scheduler.step()
        if global_step % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            log_train_metrics(model, accumulated_loss, lr, epoch, global_step)
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": accumulated_loss})
        accumulated_loss = 0.0
        accumulated_batches = 0
        global_step += 1
        optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        if i == 0:
            log_batch(batch, global_step, epoch)
        batch.to(DEVICE)
        with autocast_ctx:
            predictions = model(
                encoder_input=batch.input_ids,
                encoder_attention_mask=batch.attention_mask,
                decoder_input=batch.decoder_input_ids,
                decoder_attention_mask=batch.decoder_attention_mask,
            )
            loss: torch.Tensor = criterion(
                predictions.view(-1, model.vocab_size),
                batch.target.view(-1),
            )
        loss /= gradient_accumulation_steps
        loss.backward()
        accumulated_loss += loss.detach().item()
        accumulated_batches += 1

        if accumulated_batches == gradient_accumulation_steps:
            _step()

    if accumulated_batches > 0:
        _step()

    progress_bar.close()
    return global_step


def log_train_metrics(
    model: nn.Module, loss: float, lr: float, epoch: int, step: int
) -> None:
    mlflow.log_metrics(
        {"train_loss": loss, "learning_rate": lr, "epoch": epoch}, step=step
    )
    grad_logging.async_log(model, step)


@torch.no_grad()
def evaluate(
    model: vanilla.Transformer,
    data_module: vanilla.TransformerDataModule,
    criterion: nn.CrossEntropyLoss,
    autocast_ctx: ContextManager,
    stage: Literal["val", "test"],
    epoch: int,
    global_step: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0

    hypotheses: list[list[str]] = []
    references: list[list[list[str]]] = []

    dataloader = data_module.val_dataloader() if stage == "val" else data_module.test_dataloader()
    for i, batch in enumerate(dataloader):
        batch.to(DEVICE)
        with autocast_ctx:
            predictions = model(
                encoder_input=batch.input_ids,
                encoder_attention_mask=batch.attention_mask,
                decoder_input=batch.decoder_input_ids,
                decoder_attention_mask=batch.decoder_attention_mask,
            )
            loss: torch.Tensor = criterion(
                predictions.view(-1, model.vocab_size),
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
        if i == 0 and (stage == "test" or epoch % 5 == 0):
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
    bleu = float(bleu_score.corpus_bleu(references, hypotheses))
    mlflow.log_metrics({f"{stage}_loss": avg_loss, f"{stage}_gleu": gleu, f"{stage}_bleu": bleu}, step=global_step)
    return bleu, gleu


def make_tokenizer_and_model(
    config: ModelConfig,
) -> tuple[vanilla.Tokenizer, vanilla.Transformer]:
    tokenizer = vanilla.Tokenizer(
        config.encoding,
        config.model_max_len,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    model = vanilla.Transformer(
        vocab_size=tokenizer.vocab_size,
        num_stacks=config.num_stacks,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
    )
    return tokenizer, model


def load_data(num_samples: int) -> list[vanilla.SourceTarget]:
    return [
        vanilla.SourceTarget(source=en, target=fr)
        for en, fr in wmt_en_fr.load_parallel_sentences(
            "/data/datasets/wmt/en-fr", num_samples=num_samples
        )
    ]


def make_autocast_ctx(precision: Literal["fp32", "bf16"]) -> ContextManager:
    use_cuda = (DEVICE.type == "cuda")

    match precision:
        case "bf16":
            if not use_cuda:
                logger.warning("bf16 requested but running on CPU; using fp32.")
                autocast_ctx = nullcontext()
            else:
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("bf16 requested but not supported on this GPU.")
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        case "fp32":
            autocast_ctx = nullcontext()
        case _:
            raise ValueError(f"precision type: {precision} is not supported.")

    return autocast_ctx


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


def log_batch(batch: vanilla.LabeledBatch, step: int, epoch: int) -> None:
    mlflow.log_dict(
        {
            "input_ids": batch.input_ids.tolist(),
            "decoder_input_ids": batch.decoder_input_ids.tolist(),
            "target": batch.target.tolist(),
        },
        f"batches/epoch_{epoch}_step_{step}.json",
    )


if __name__ == "__main__":
    train()
