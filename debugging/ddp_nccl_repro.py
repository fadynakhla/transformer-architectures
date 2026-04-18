"""Minimal NCCL hang reproducer. Launch with torchrun on each Spark."""
import contextlib
import faulthandler
import logging
import os
import signal
import socket
import sys
import time

import pydantic
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from transformer_architectures import config


class ReproConfig(pydantic.BaseModel):
    hidden_dim: int = 1024
    num_blocks: int = 16
    batch_size: int = 8
    seq_len: int = 4096

    steps: int = 10_000_000
    grad_accum_steps: int = 4
    lr: float = 1e-4
    dtype: str = "bf16"
    bucket_cap_mb: int = 25

    log_every: int = 1000
    tick_every_s: float = 30.0
    verbose_step_trace: bool = False

    inject_rank: int = -1
    inject_step: int = -1
    inject_sleep_ms: int = 0


DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class StackOfBlocks(nn.Module):
    def __init__(self, hidden_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_blocks)
        )
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = x + b(x)
        return self.head(x)


def setup_train_logger(rank: int) -> logging.Logger:
    """Return a file-only logger for the hang watcher.

    Writes to /tmp/train_rank{rank}.log. Only the main training loop should
    use it — if it stops writing, the watcher fires. File-only keeps the tqdm
    bar on stderr uncluttered.
    """
    logger = logging.getLogger(f"rank{rank}.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(f"/tmp/train_rank{rank}.log", mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(fh)
    return logger


def install_diagnostic_signals(rank: int) -> None:
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

    def die(signum, frame):
        tqdm.write(f"rank={rank} received signal={signum}", file=sys.stderr)
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, die)
    signal.signal(signal.SIGINT, die)


# def maybe_inject_sleep(cfg: ReproConfig, rank: int, step: int) -> None:
#     if (
#         cfg.inject_rank == rank
#         and cfg.inject_step == step
#         and cfg.inject_sleep_ms > 0
#     ):
#         tqdm.write(
#             f"rank={rank} injecting sleep step={step} ms={cfg.inject_sleep_ms}"
#         )
#         time.sleep(cfg.inject_sleep_ms / 1000.0)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/repro_hang.yaml"
    cfg = config.load_config(config_path, section="Repro", model_class=ReproConfig)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    train_logger = setup_train_logger(rank)
    install_diagnostic_signals(rank)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    print(
        f"starting host={socket.gethostname()} rank={rank} "
        f"world_size={world_size} device={device} pid={os.getpid()}",
        flush=True,
    )

    dist.init_process_group(backend="nccl", init_method="env://")

    dtype = DTYPE_MAP[cfg.dtype]
    use_autocast = dtype in (torch.bfloat16, torch.float16)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if use_autocast
        else contextlib.nullcontext()
    )

    base_model = StackOfBlocks(cfg.hidden_dim, cfg.num_blocks).to(device)
    ddp_model = DDP(
        base_model,
        device_ids=[device.index],
        output_device=device.index,
        bucket_cap_mb=cfg.bucket_cap_mb,
    )
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    x = torch.empty(cfg.batch_size, cfg.seq_len, cfg.hidden_dim, device=device)
    y = torch.empty(cfg.batch_size, cfg.seq_len, cfg.hidden_dim, device=device)

    dist.barrier()
    train_logger.info(
        "ddp ready blocks=%d hidden_dim=%d bucket_cap_mb=%d dtype=%s",
        cfg.num_blocks, cfg.hidden_dim, cfg.bucket_cap_mb, cfg.dtype,
    )

    accum_loss = torch.zeros((), device=device)
    last_tick = time.monotonic()
    global_step = 0
    loss_val = float("nan")

    pbar = tqdm(
        range(cfg.steps),
        desc=f"rank{rank}",
        dynamic_ncols=True,
        mininterval=0.5,
        smoothing=0.1,
    )
    for step in pbar:
        # maybe_inject_sleep(cfg, rank, step)

        x.normal_()
        y.normal_()

        is_sync_step = ((step + 1) % cfg.grad_accum_steps == 0)
        sync_ctx = contextlib.nullcontext() if is_sync_step else ddp_model.no_sync()

        if cfg.verbose_step_trace:
            train_logger.info(
                "rank=%d step=%d sync=%s before_forward", rank, step, is_sync_step
            )

        with sync_ctx:
            with autocast_ctx:
                out = ddp_model(x)
                loss = criterion(out, y) / cfg.grad_accum_steps
            accum_loss += loss.detach()
            loss.backward()

        if cfg.verbose_step_trace:
            train_logger.info(
                "rank=%d step=%d sync=%s after_backward", rank, step, is_sync_step
            )

        if is_sync_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_val = float(accum_loss.item())
            accum_loss.zero_()
            global_step += 1
            pbar.set_postfix(gs=global_step, loss=f"{loss_val:.4f}")

            now = time.monotonic()
            if cfg.tick_every_s > 0 and now - last_tick >= cfg.tick_every_s:
                train_logger.info(
                    "rank=%d tick global_step=%d step=%d loss=%f",
                    rank, global_step, step, loss_val,
                )
                last_tick = now

            if cfg.log_every > 0 and global_step % cfg.log_every == 0:
                train_logger.info(
                    "rank=%d global_step=%d loss=%f", rank, global_step, loss_val,
                )

    train_logger.info("rank=%d finished cleanly", rank)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
