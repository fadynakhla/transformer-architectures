from typing import Any, Optional
import os

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

MODEL_DIR = "/data/trained/"


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    model_name: str,
    epoch: int,
    global_step: int,
    overwrite: bool = False,
) -> None:
    run_name = mlflow.active_run().info.run_name  # type: ignore
    run_id = mlflow.active_run().info.run_id  # type: ignore
    save_dir = os.path.join(MODEL_DIR, model_name, f"{run_name}_{run_id[:6]}")
    if epoch == 0:
        make_dir(save_dir, overwrite)
    save_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "generator_state": generator.get_state(),
        "run_id": run_id,
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    checkpoint_path: str,
) -> tuple[int, int, str]:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    generator.set_state(checkpoint["generator_state"])
    return checkpoint["epoch"], checkpoint["global_step"], checkpoint["run_id"]


def resume_mlflow_run(
    run_id: Optional[str] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
) -> mlflow.ActiveRun:
    if run_id is None:
        return mlflow.start_run()

    if epoch is None or global_step is None:
        raise ValueError(
            f"Run ID provided but either epoch or global_step is null. Provide all three or none of them."
        )

    run = mlflow.start_run()
    mlflow.set_tag("resumed_from", run_id)
    mlflow.set_tag("resumed_epoch", epoch)
    mlflow.set_tag("resumed_global_step", global_step)
    return run


def make_dir(save_dir: str, overwrite: bool) -> None:
    if not overwrite and os.path.exists(save_dir):
        raise FileExistsError(f"Directory {save_dir} already exists.")
    os.makedirs(save_dir, exist_ok=overwrite)
