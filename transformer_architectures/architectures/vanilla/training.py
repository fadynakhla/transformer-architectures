from typing import Callable, Optional
import math

import aim
import pydantic
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from nltk.translate import bleu_score, gleu_score

from transformer_architectures.architectures import vanilla
from transformer_architectures.dataloading import wmt_en_fr

IGNORE_ID = -100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


run = aim.Run(
    repo="aim://0.0.0.0:53800",
    experiment="Vanilla Transformer - ENFR",
)


class TrainingConfig(pydantic.BaseModel):
    batch_size: int
    grad_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    epochs: int
    label_smoothing: float


class ModelConfig(pydantic.BaseModel):
    encoding: str = "r50k_base"
    model_max_len: int = 512
    num_stacks: int = 6
    embed_dim: int = 512
    num_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1


model_config = ModelConfig()
train_config = TrainingConfig(
    batch_size=8,
    grad_accumulation_steps=16,
    learning_rate=1,
    warmup_steps=4000,
    epochs=50,
    label_smoothing=0.1,
)
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
        data=load_data(),
        tokenizer=tokenizer,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        val_split=0,
    )
    data_module.setup()
    aim_text = aim.Text(text=f"{data_module.train_dataset[0]}")
    run.track(aim_text, name="example", context={"subset": "train"})
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_ID, label_smoothing=train_config.label_smoothing
    )
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, make_schedule(model_config.embed_dim, train_config.warmup_steps)
    )
    global_step = 0
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
        )
        eval_epoch(model, data_module, criterion, epoch)


def train_epoch(
    model: vanilla.Transformer,
    data_module: vanilla.TransformerDataModule,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    gradient_accumulation_steps: int,
    epoch: int,
    global_step: int,
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
        accumulated_loss += loss.item()

        if (i + 1) % gradient_accumulation_steps == 0 or i == total_batches - 1:
            optimizer.step()
            scheduler.step()
            run.track(
                accumulated_loss,
                name="train_loss",
                step=global_step,
                epoch=epoch,
                context={"subset": "train"},
            )
            run.track(
                scheduler.get_last_lr()[0],
                name="learning_rate",
                step=global_step,
                context={"subset": "train"},
            )

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": accumulated_loss})
            accumulated_loss = 0.0
            global_step += 1
            optimizer.zero_grad()

    progress_bar.close()
    return global_step


def eval_epoch(
    model: vanilla.Transformer,
    data_module: vanilla.TransformerDataModule,
    criterion: nn.CrossEntropyLoss,
    epoch: int,
) -> None:
    model.eval()
    total_loss = 0.0

    hypotheses: list[list[str]] = []
    references: list[list[list[str]]] = []

    dataloader = data_module.test_dataloader()
    for batch in dataloader:
        batch.to(DEVICE)
        with torch.no_grad():
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

    avg_loss = total_loss / len(dataloader)
    run.track(avg_loss, name="eval_loss", epoch=epoch, context={"subset": "eval"})
    gleu = gleu_score.corpus_gleu(references, hypotheses)
    run.track(gleu, name="gleu", epoch=epoch, context={"subset": "eval"})
    bleu = bleu_score.corpus_bleu(references, hypotheses)
    run.track(bleu, name="bleu", epoch=epoch, context={"subset": "eval"})


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


def load_data() -> list[vanilla.SourceTarget]:
    return [
        vanilla.SourceTarget(source=en, target=fr)
        for en, fr in wmt_en_fr.load_parallel_sentences("/data/datasets/wmt/en-fr")
    ]


def make_schedule(model_size: int, warmump_steps: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        if step == 0:
            step = 1
        return model_size**-0.5 * min(step**-0.5, step * warmump_steps**-1.5)

    return schedule


if __name__ == "__main__":
    train()
