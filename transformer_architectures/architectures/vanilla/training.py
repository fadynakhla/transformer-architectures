from typing import Callable, Optional
import math

import aim
import loguru
import pydantic
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from nltk.translate import bleu_score, gleu_score

from transformer_architectures import config
from transformer_architectures.architectures import vanilla
from transformer_architectures.dataloading import wmt_en_fr
from transformer_architectures.training import checkpointing, grad_logging

IGNORE_ID = -100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME = "vanilla_transformer_large"


logger = loguru.logger

run = aim.Run(
    repo="aim://0.0.0.0:53800",
    experiment="Vanilla Transformer Large - ENFR",
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
    log_interval: int = 50


class ModelConfig(pydantic.BaseModel):
    encoding: str = "r50k_base"
    model_max_len: int = 512
    num_stacks: int = 6
    embed_dim: int = 1024
    num_heads: int = 16
    ff_dim: int = 4096
    dropout: float = 0.3


model_config = config.load_config(CONFIG_PATH, section="Model", model_class=ModelConfig)
train_config = config.load_config(CONFIG_PATH, section="Training", model_class=TrainingConfig)

run["hparams"] = {
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
        val_split=0,
    )
    data_module.setup()
    aim_text = aim.Text(text=f"{data_module.train_dataset[0]}")
    run.track(aim_text, name="example", context={"subset": "train"})
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_ID, label_smoothing=train_config.label_smoothing
    )
    optimizer = optim.Adam(
        model.parameters(), lr=train_config.learning_rate, betas=(0.9, 0.98), eps=1e-9
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
        )
        bleu, _ = eval_epoch(model, data_module, criterion, epoch)
        if bleu > max_bleu:
            logger.info(f"New best BLEU score: {bleu}. Saving checkpoint.")
            max_bleu = bleu
            checkpointing.save_checkpoint(
                model, optimizer, scheduler, data_module.generator, NAME, run, epoch
            )


def train_epoch(
    model: vanilla.Transformer,
    data_module: vanilla.TransformerDataModule,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    gradient_accumulation_steps: int,
    epoch: int,
    global_step: int,
    log_interval: int
) -> int:
    model.train()

    dataloader = data_module.train_dataloader()
    total_batches = len(dataloader)
    total_groups = math.ceil(total_batches / gradient_accumulation_steps)
    final_group_size = (
        total_batches % gradient_accumulation_steps or gradient_accumulation_steps
    )
    accumulated_loss = 0.0

    progress_bar = tqdm.tqdm(total=total_groups, desc=f"Epoch {epoch}")
    for i, batch in enumerate(dataloader):
        if i == 0:
            log_batch(batch, global_step, epoch)
        batch.to(DEVICE)
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
        current_group = i // gradient_accumulation_steps + 1
        accumulation_steps = (
            final_group_size
            if current_group == total_groups
            else gradient_accumulation_steps
        )
        loss /= accumulation_steps
        loss.backward()
        accumulated_loss += loss.detach().item()

        if (i + 1) % gradient_accumulation_steps == 0 or i == total_batches - 1:
            optimizer.step()
            scheduler.step()
            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                log_train_metrics(model, accumulated_loss, lr, epoch, global_step)

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": accumulated_loss})
            accumulated_loss = 0.0
            global_step += 1
            optimizer.zero_grad()

    progress_bar.close()
    return global_step

def log_train_metrics(
    model: nn.Module, loss: float, lr: float, epoch: int, step: int
) -> None:
    run.track(
        loss, name="train_loss", step=step, epoch=epoch, context={"subset": "train"}
    )
    run.track(
        lr, name="learning_rate", step=step, epoch=epoch, context={"subset": "train"}
    )
    grad_logging.async_log(run, model, step, epoch)


@torch.no_grad()
def eval_epoch(
    model: vanilla.Transformer,
    data_module: vanilla.TransformerDataModule,
    criterion: nn.CrossEntropyLoss,
    epoch: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0

    hypotheses: list[list[str]] = []
    references: list[list[list[str]]] = []

    dataloader = data_module.test_dataloader()
    for i, batch in enumerate(dataloader):
        batch.to(DEVICE)
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
            [
                p.split()
                for p in data_module.tokenizer.batch_decode(predicted_sequences)
            ]
        )
        targets = torch.masked_fill(batch.target, batch.target == IGNORE_ID, 0)
        references.extend(
            [[t.split()] for t in data_module.tokenizer.batch_decode(targets)]
        )
        if i == 0 and epoch % 5 == 0:
            for j, (h, r) in enumerate(zip(hypotheses, references)):
                hyp_text = aim.Text(text=f"{h}")
                ref_text = aim.Text(text=f"{r}")
                run.track(
                    hyp_text,
                    name=f"eval_sample_{j}",
                    epoch=epoch,
                    context={"subset": "hypothesis"},
                )
                run.track(
                    ref_text,
                    name=f"eval_sample_{j}",
                    epoch=epoch,
                    context={"subset": "references"},
                )

    avg_loss = total_loss / len(dataloader)
    run.track(avg_loss, name="eval_loss", epoch=epoch, context={"subset": "eval"})
    gleu = gleu_score.corpus_gleu(references, hypotheses)
    run.track(gleu, name="gleu", epoch=epoch, context={"subset": "eval"})
    bleu = bleu_score.corpus_bleu(references, hypotheses)
    run.track(bleu, name="bleu", epoch=epoch, context={"subset": "eval"})
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


def log_batch(batch: vanilla.LabeledBatch, step: int, epoch) -> None:
    input_text = aim.Text(text=f"{batch.input_ids}")
    decoder_text = aim.Text(text=f"{batch.decoder_input_ids}")
    target_text = aim.Text(text=f"{batch.target}")
    run.track(
        input_text,
        name="sample_batch",
        step=step,
        epoch=epoch,
        context={"subset": "input_ids"},
    )
    run.track(
        decoder_text,
        name="sample_batch",
        step=step,
        epoch=epoch,
        context={"subset": "decoder_input_ids"},
    )
    run.track(
        target_text,
        name="sample_batch",
        step=step,
        epoch=epoch,
        context={"subset": "target"},
    )


if __name__ == "__main__":
    train()
