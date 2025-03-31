import os

import aim
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
    run: aim.Run,
    epoch: int,
    overwrite: bool = False,
) -> None:
    save_dir = os.path.join(MODEL_DIR, model_name, f"run_{run.hash[:6]}")
    if epoch == 0:
        make_dir(save_dir, overwrite)
    save_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "generator_state": generator.get_state(),
        "run_hash": run.hash,
        "epoch": epoch,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    run: aim.Run,
) -> int:
    dir = f"/data/trained/vanilla_transformer/run_{run.hash()[:6]}"
    epoch = max(
        [
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(dir)
            if f.startswith("epoch")
        ]
    )
    checkpoint = torch.load(
        f"/data/trained/vanilla_transformer/run_{run.hash()[:6]}/epoch_{epoch}.pt"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    generator.set_state(checkpoint["generator_state"])
    return checkpoint["epoch"]


def make_dir(save_dir: str, overwrite: bool) -> None:
    if not overwrite and os.path.exists(save_dir):
        raise FileExistsError(f"Directory {save_dir} already exists.")
    os.makedirs(save_dir, exist_ok=overwrite)
