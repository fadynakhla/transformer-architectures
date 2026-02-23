from typing import Any, Callable
import functools
import math

import loguru
import mlflow
import pydantic
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.ops as ops
import tqdm

from transformer_architectures import config
from transformer_architectures.architectures import vit

# from transformer_architectures.datasets import wmt_en_fr
from transformer_architectures.training import checkpointing, grad_logging

IGNORE_ID = -100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME = "vision_transformer_large"


logger = loguru.logger


mlflow.set_tracking_uri("http://10.9.9.249:5000")
mlflow.set_experiment("Vision Transformer Large - OpenImages")
mlflow.config.enable_system_metrics_logging()  # pyright: ignore[reportPrivateImportUsage]
mlflow.config.set_system_metrics_sampling_interval(
    30
)  # pyright: ignore[reportPrivateImportUsage]


CONFIG_PATH = "configs/vision_large.yaml"


class TrainingConfig(pydantic.BaseModel):
    data_path: str
    annotations_path: str
    batch_size: int
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
    dropout: float = 0.1
    num_classes: int | None = None


model_config = config.load_config(CONFIG_PATH, section="Model", model_class=ModelConfig)
train_config = config.load_config(
    CONFIG_PATH, section="Training", model_class=TrainingConfig
)


def train() -> None:
    logger.info("Creating DataModule")
    data_module = vit.TransformerDataModule(
        data_dir=train_config.data_path,
        annotations_dir=train_config.annotations_path,
        data_samples=train_config.num_samples,
        image_size=model_config.image_size,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        test_split=0.05,
        val_split=0,
    )
    data_module.setup()
    logger.info(
        "Succesfully created DataModule with {} train images",
        len(data_module.train_dataset),
    )
    model_config.num_classes = len(data_module.mid_to_index)
    logger.info("Loading Model")
    model = make_model(model_config)
    model = model.to(DEVICE)
    logger.info("Model loaded to device: {}", DEVICE)

    params: dict[str, int | float | dict[str, Any]] = {
        "batch_size": train_config.batch_size * train_config.grad_accumulation_steps,
        "grad_accumulation_steps": train_config.grad_accumulation_steps,
        "learning_rate": train_config.learning_rate,
        "model_config": model_config.model_dump(),
    }

    # logger.info("Calculating pos_weight to correct for class imbalance.")
    # pos_weight = compute_pos_weight(data_module).to(DEVICE)
    # criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    criterion = functools.partial(
        ops.sigmoid_focal_loss, alpha=0.9, gamma=2, reduction="mean"
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-9,
        weight_decay=0.1,
    )
    epoch_steps = math.ceil(
        len(data_module.train_dataloader()) / train_config.grad_accumulation_steps
    )
    total_steps = train_config.epochs * epoch_steps
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, make_cosine_schedule(train_config.warmup_steps, total_steps)
    )
    global_step = 0
    min_eval_loss = 1e9
    with mlflow.start_run():
        mlflow.log_params(params=params)
        mlflow.log_text(
            text=f"{data_module.train_dataset[0]}", artifact_file="sample_batch.txt"
        )
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
            metrics = eval_epoch(model, data_module, criterion, global_step)
            eval_loss = metrics["eval_loss"]
            if eval_loss < min_eval_loss:
                logger.info(f"New best eval loss: {eval_loss}. Saving checkpoint.")
                min_eval_loss = eval_loss
                checkpointing.save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    data_module.generator,
                    NAME,
                    epoch,
                    global_step,
                )


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
            log_batch(batch, global_step)
        batch.to(DEVICE)
        predictions = model(images=batch.images)
        loss: torch.Tensor = criterion(
            predictions,
            batch.labels,
        )
        # loss = (per_logit_loss * batch.masks).sum() / batch.masks.sum().clamp_min(1)
        # loss = per_logit_loss.mean()
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
    mlflow.log_metrics(
        {"train_loss": loss, "learning_rate": lr, "epoch": epoch}, step=step
    )
    grad_logging.async_log(model, step)


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
def eval_epoch(
    model: vit.VisionTransformer,
    data_module: vit.TransformerDataModule,
    criterion: nn.BCEWithLogitsLoss,
    global_step: int,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0

    cm = torch.zeros((data_module.num_classes, 2, 2), dtype=torch.long, device=DEVICE)
    dataloader = data_module.test_dataloader()

    for _, batch in enumerate(dataloader):
        batch.to(DEVICE)
        logits = model(images=batch.images)
        loss: torch.Tensor = criterion(
            logits,
            batch.labels,
        )
        # loss = (per_logit_loss * batch.masks).sum() / batch.masks.sum().clamp_min(1)
        total_loss += loss.item()

        preds = torch.sigmoid(logits) >= threshold
        targets = batch.labels >= 0.5

        tp = (preds & targets).sum(dim=0)
        fp = (preds & ~targets).sum(dim=0)
        fn = (~preds & targets).sum(dim=0)
        tn = (~preds & ~targets).sum(dim=0)

        cm[:, 0, 0] += tn
        cm[:, 0, 1] += fp
        cm[:, 1, 0] += fn
        cm[:, 1, 1] += tp

    avg_loss = total_loss / len(dataloader)

    metrics = {"eval_loss": avg_loss} | calculate_metrics_from_confusion(cm)

    mlflow.log_metrics(metrics, step=global_step)
    return metrics


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
def compute_pos_weight(data_module: vit.TransformerDataModule) -> torch.Tensor:
    pos = torch.zeros(data_module.num_classes, dtype=torch.float64)
    # known = torch.zeros(data_module.num_classes, dtype=torch.float64)
    dataloader = data_module.train_dataloader()
    for batch in dataloader:
        pos += (batch.labels * batch.masks).sum(0)
        # known += batch.masks.sum(0)
    neg = torch.ones(data_module.num_classes, dtype=torch.float64) - pos
    w = (neg / pos.clamp_min(1)).to(torch.float32)
    return w.clamp_(max=50)


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


def log_batch(batch: vit.LabeledBatch, step: int) -> None:
    image = batch.images[0].permute(1, 2, 0).cpu().numpy()
    mlflow.log_image(image, artifact_file=f"sample_inputs/step_{step}.png")


def calculate_metrics_from_confusion(cm: torch.Tensor) -> dict[str, float]:
    # ---- derive metrics from confusion ----
    # per-class counts
    tn_c = cm[:, 0, 0].to(torch.float64)
    fp_c = cm[:, 0, 1].to(torch.float64)
    fn_c = cm[:, 1, 0].to(torch.float64)
    tp_c = cm[:, 1, 1].to(torch.float64)

    eps = 1e-12

    # per-class metrics
    prec_c = tp_c / (tp_c + fp_c + eps)
    rec_c = tp_c / (tp_c + fn_c + eps)
    # acc_c = (tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c)
    f1_c = 2.0 * prec_c * rec_c / (prec_c + rec_c + eps)
    fpr_c = fp_c / (fp_c + tn_c + eps)
    fnr_c = fn_c / (fn_c + tp_c + eps)

    # macro (mean over classes)
    precision_macro = float(torch.nanmean(prec_c).item())
    recall_macro = float(torch.nanmean(rec_c).item())
    # accuracy_macro = float(torch.nanmean(acc_c).item())
    f1_macro = float(torch.nanmean(f1_c).item())
    fpr_macro = float(torch.nanmean(fpr_c).item())
    fnr_macro = float(torch.nanmean(fnr_c).item())

    # micro (sum counts over classes then compute)
    tp = tp_c.sum()
    fp = fp_c.sum()
    fn = fn_c.sum()
    tn = tn_c.sum()
    precision_micro = float((tp / (tp + fp + eps)).item())
    recall_micro = float((tp / (tp + fn + eps)).item())
    accuracy_micro = float(((tp + tn) / (tp + tn + fp + fn)).item())
    f1_micro = (
        2.0 * precision_micro * recall_micro / (precision_micro + recall_micro + eps)
    )
    fpr_micro = float((fp / (fp + tn + eps)).item())
    fnr_micro = float((fn / (fn + tp + eps)).item())

    metrics = {
        "accuracy": accuracy_micro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "fpr_micro": fpr_micro,
        "fnr_micro": fnr_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        # "accuracy_macro": accuracy_macro,
        "f1_macro": f1_macro,
        "fpr_macro": fpr_macro,
        "fnr_macro": fnr_macro,
    }
    return metrics


if __name__ == "__main__":
    train()
