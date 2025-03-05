from typing import Callable, Optional

import aim

import tqdm
import pydantic
import torch
import torch.nn as nn
import torch.optim as optim

from transformer_architectures.architectures import vanilla


IGNORE_ID = -100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


run = aim.Run(
    repo="aim://0.0.0.0:53800",
    experiment="Vanilla Transformer - Overfitting",
)


class TrainingConfig(pydantic.BaseModel):
    batch_size: int
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
    batch_size=2, learning_rate=0.001, warmup_steps=2, epochs=10000, label_smoothing=0
)
run["hparams"] = {
    "batch_size": train_config.batch_size,
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
        test_split=0,
        val_split=0,
    )
    data_module.setup()
    aim_text = aim.Text(text=f"{data_module.train_dataset[0]}")
    run.track(aim_text, name="example", context={"subset": "train"})
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_ID, label_smoothing=train_config.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, make_schedule(model_config.embed_dim, train_config.warmup_steps)
    )
    for epoch in range(train_config.epochs):
        train_epoch(model, data_module, criterion, optimizer, scheduler, epoch)



def train_epoch(
    model: vanilla.Transformer,
    data_module: vanilla.TransformerDataModule,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
) -> None:
    model.train()
    progress_bar = tqdm.tqdm(data_module.train_dataloader(), desc=f"Epoch {epoch}")
    for batch in progress_bar:
        optimizer.zero_grad()
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
        loss.backward()

        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({"loss": loss.item()})
        run.track(loss.item(), name="train_loss", epoch=epoch, context={"subset": "train"})
    run.track(epoch, name="epoch", context={"subset": "train"})


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
        vanilla.SourceTarget(source="hello", target="world"),
        vanilla.SourceTarget(source="foo", target="bar"),
    ]

def make_schedule(model_size: int, warmump_steps: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        if step == 0:
            step = 1
        return model_size ** -0.5 * min(step ** -0.5, step * warmump_steps ** -1.5)
    return schedule


if __name__ == "__main__":
    train()
