from typing import Callable
import math

import aim
import loguru
import pydantic
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from transformer_architectures import config
from transformer_architectures.architectures import vit
# from transformer_architectures.datasets import wmt_en_fr
from transformer_architectures.training import checkpointing, grad_logging

IGNORE_ID = -100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME = "vision_transformer_large"


logger = loguru.logger

run = aim.Run(
    repo="aim://0.0.0.0:53800",
    experiment="Vision Transformer Large - OpenImages",
)

CONFIG_PATH = "configs/vision_large.yaml"


class TrainingConfig(pydantic.BaseModel):
    data_path: str
    batch_size: int = 8
    grad_accumulation_steps: int = 1024
    learning_rate: float
    warmup_steps: int
    epochs: int
    num_samples: int
    log_interval: int = 2


class ModelConfig(pydantic.BaseModel):
    patch_size: int = 16
    image_size: int = 512
    num_stacks: int = 24
    embed_dim: int = 1024
    num_heads: int = 16
    ff_dim: int = 4096
    dropout: float = 0.3
    num_classes: int | None = None


model_config = config.load_config(CONFIG_PATH, section="Model", model_class=ModelConfig)
train_config = config.load_config(
    CONFIG_PATH, section="Training", model_class=TrainingConfig
)

run["hparams"] = {
    "batch_size": train_config.batch_size * train_config.grad_accumulation_steps,
    "grad_accumulation_steps": train_config.grad_accumulation_steps,
    "learning_rate": train_config.learning_rate,
    "model_config": model_config.model_dump(),
}


def train() -> None:
    logger.info("Loading Data")
    data_module = vit.TransformerDataModule(
        data_dir=train_config.data_path,
        data_samples=train_config.num_samples,
        image_size=(model_config.image_size, model_config.image_size),
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        test_split=0.05,
        val_split=0,
    )
    data_module.setup()
    logger.info("Succesfully loaded {} images", len(data_module.train_dataset.dataset))
    model_config.num_classes = len(data_module.mid_to_index)
    logger.info("Loading Model")
    model = make_model(model_config)
    aim_text = aim.Text(text=f"{data_module.train_dataset[0]}")
    run.track(aim_text, name="example", context={"subset": "train"})
    model = model.to(DEVICE)
    logger.info("Model loaded to device: {}", DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.1
    )
    epoch_steps = math.ceil(
        len(data_module.train_dataloader()) / train_config.grad_accumulation_steps
    )
    total_steps = train_config.epochs * epoch_steps
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, make_cosine_schedule(train_config.warmup_steps, total_steps)
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
            train_config.log_interval,
        )
        # bleu, _ = eval_epoch(model, data_module, criterion, epoch)
        # if bleu > max_bleu:
        #     logger.info(f"New best BLEU score: {bleu}. Saving checkpoint.")
        #     max_bleu = bleu
        #     checkpointing.save_checkpoint(
        #         model, optimizer, scheduler, data_module.generator, NAME, run, epoch
        #     )


def train_epoch(
    model: vit.VisionTransformer,
    data_module: vit.TransformerDataModule,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    gradient_accumulation_steps: int,
    epoch: int,
    global_step: int,
    log_interval: int,
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
        predictions = model(images=batch.images)
        loss: torch.Tensor = criterion(
            predictions,
            batch.labels,
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


# @torch.no_grad()
# def eval_epoch(
#     model: vit.VisionTransformer,
#     data_module: vit.TransformerDataModule,
#     criterion: nn.CrossEntropyLoss,
#     epoch: int,
# ) -> tuple[float, float]:
#     model.eval()
#     total_loss = 0.0

#     hypotheses: list[list[str]] = []
#     references: list[list[list[str]]] = []

#     dataloader = data_module.test_dataloader()
#     for i, batch in enumerate(dataloader):
#         batch.to(DEVICE)
#         predictions = model(
#             encoder_input=batch.input_ids,
#             encoder_attention_mask=batch.attention_mask,
#             decoder_input=batch.decoder_input_ids,
#             decoder_attention_mask=batch.decoder_attention_mask,
#         )
#         loss: torch.Tensor = criterion(
#             predictions.view(-1, model.vocab_size),
#             batch.target.view(-1),
#         )
#         total_loss += loss.item()
#         predicted_sequences = torch.argmax(predictions, dim=-1)
#         predicted_sequences = torch.masked_fill(
#             predicted_sequences, batch.target == IGNORE_ID, 0
#         )
#         hypotheses.extend(
#             [p.split() for p in data_module.tokenizer.batch_decode(predicted_sequences)]
#         )
#         targets = torch.masked_fill(batch.target, batch.target == IGNORE_ID, 0)
#         references.extend(
#             [[t.split()] for t in data_module.tokenizer.batch_decode(targets)]
#         )
#         if i == 0 and epoch % 5 == 0:
#             for j, (h, r) in enumerate(zip(hypotheses, references)):
#                 hyp_text = aim.Text(text=f"{h}")
#                 ref_text = aim.Text(text=f"{r}")
#                 run.track(
#                     hyp_text,
#                     name=f"eval_sample_{j}",
#                     epoch=epoch,
#                     context={"subset": "hypothesis"},
#                 )
#                 run.track(
#                     ref_text,
#                     name=f"eval_sample_{j}",
#                     epoch=epoch,
#                     context={"subset": "references"},
#                 )

#     avg_loss = total_loss / len(dataloader)
#     run.track(avg_loss, name="eval_loss", epoch=epoch, context={"subset": "eval"})
#     gleu = gleu_score.corpus_gleu(references, hypotheses)
#     run.track(gleu, name="gleu", epoch=epoch, context={"subset": "eval"})
#     bleu = bleu_score.corpus_bleu(references, hypotheses)
#     run.track(bleu, name="bleu", epoch=epoch, context={"subset": "eval"})
#     return bleu, gleu


def make_model(
    config: ModelConfig,
) -> vit.VisionTransformer:

    model_class = vit.VisionTransformer
    if config.num_classes:
        model_class = vit.VisionTransformerForImageClassification

    return model_class(**config.model_dump())



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


def log_batch(batch: vit.LabeledBatch, step: int, epoch: int) -> None:
    image = aim.Image(image=batch.images[0], caption=f"{batch.labels[0]}")
    run.track(
        image,
        name="sample_input",
        step=step,
        epoch=epoch,
        context={"subset": "image_and_label"},
    )


if __name__ == "__main__":
    train()
