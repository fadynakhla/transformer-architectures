import socket
from typing import Any, ContextManager, Generic, Literal, Protocol, TypeVar
import abc
import contextlib
import math

import loguru
import mlflow
import ray.train.torch
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from transformer_architectures.training import (
    base_train_config,
    checkpointing,
    data_utils,
    grad_logging,
)
from transformer_architectures.training.distributed import context, datamodule

logger = loguru.logger


_TC = TypeVar("_TC", bound=base_train_config.BaseTrainConfig)


class TrainableArchitecture(Protocol, Generic[_TC]):
    architecture_name: str
    train_config: _TC
    mlflow_config: base_train_config.MLFlowConfig
    mlflow_run_id: str | None

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        ...

    @abc.abstractmethod
    def build_datamodule(self) -> datamodule.DataModule:
        ...

    @abc.abstractmethod
    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        ...

    @abc.abstractmethod
    def build_scheduler(
        self, optimizer: optim.Optimizer, steps_per_epoch: int
    ) -> optim.lr_scheduler.LRScheduler:
        ...

    @abc.abstractmethod
    def build_criterion(self) -> nn.Module:
        ...

    @abc.abstractmethod
    def make_run_params(self) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        criterion: nn.Module,
        autocast_ctx: ContextManager,
    ) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        data_module: datamodule.DataModule,
        criterion: nn.Module,
        autocast_ctx: ContextManager,
        stage: str,
        epoch: int,
        global_step: int,
        distributed_ctx: context.DistributedContext,
    ) -> dict[str, float]:
        ...

    @abc.abstractmethod
    def log_batch(self, batch: Any, step: int, epoch: int) -> None: ...

    def train_epoch(
        self,
        model: nn.Module,
        data_module: datamodule.DataModule,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        grad_accumulation_steps: int,
        epoch: int,
        global_step: int,
        log_interval: int,
        log_grad_distributions: bool,
        autocast_ctx: ContextManager,
        distributed_ctx: context.DistributedContext,
    ) -> int:
        model.train()

        dataloader = data_module.train_dataloader()
        total_batches = len(dataloader)
        total_batches = synchronize_int_min(total_batches, distributed_ctx)

        total_groups, final_acc_steps = divmod(total_batches, grad_accumulation_steps)
        if final_acc_steps:
            total_groups += 1
        else:
            final_acc_steps = grad_accumulation_steps

        accumulated_loss = 0.0

        progress_bar: tqdm.tqdm | None = None
        if distributed_ctx.is_head:
            progress_bar = tqdm.tqdm(total=total_groups, desc=f"Epoch {epoch}")
        for i, batch in enumerate(dataloader):
            if i == 0 and distributed_ctx.is_head:
                self.log_batch(batch, global_step, epoch)
            if i >= total_batches:
                break
            batch.to(distributed_ctx.device)

            is_grad_acc_step = (i + 1) % grad_accumulation_steps == 0
            is_final_step = i == total_batches - 1
            is_update_step = is_grad_acc_step or is_final_step

            sync_context = (
                contextlib.nullcontext()
                if is_update_step
                or not isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model.no_sync()
            )
            with sync_context:
                loss = self.train_step(model, batch, criterion, autocast_ctx)
                acc_norm = final_acc_steps if is_final_step else grad_accumulation_steps
                loss /= acc_norm
                loss.backward()
                accumulated_loss += loss.detach().item()

            if is_update_step:
                optimizer.step()
                scheduler.step()
                if progress_bar is not None:
                    progress_bar.set_postfix({"loss": accumulated_loss}, refresh=False)
                    progress_bar.update(1)

                is_log_step = global_step % log_interval == 0
                if distributed_ctx.is_head and (is_log_step or is_final_step):
                    lr = float(scheduler.get_last_lr()[0])
                    log_train_metrics(
                        model=model,
                        loss=accumulated_loss,
                        lr=lr,
                        epoch=epoch,
                        epoch_frac=epoch + (i / total_batches),
                        step=global_step,
                        log_distributions=log_grad_distributions,
                    )

                accumulated_loss = 0.0
                global_step += 1
                optimizer.zero_grad()
        if progress_bar is not None:
            progress_bar.close()
        return global_step

    def distributed_train_loop(self) -> None:
        distributed_ctx = context.DistributedContext.from_ray_context()
        self.mlflow_setup(distributed_ctx)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.run_training(distributed_ctx)

    def run_training(self, distributed_ctx: context.DistributedContext):
        model = self.build_model()
        model = ray.train.torch.prepare_model(model)

        data_module = self.build_datamodule()
        data_module.setup(distributed_ctx)

        if distributed_ctx.is_head:
            mlflow.log_params(params=self.make_run_params())
            mlflow.log_text(
                text=f"{data_module.train_dataset[0]}",
                artifact_file="sample_batch.txt",
            )
        autocast_ctx = make_autocast_ctx(
            self.train_config.precision, distributed_ctx.device
        )
        criterion = self.build_criterion()
        optimizer = self.build_optimizer(model)
        epoch_steps = math.ceil(
            len(data_module.train_dataloader())
            / self.train_config.grad_accumulation_steps
        )
        epoch_steps = synchronize_int_min(epoch_steps, distributed_ctx)
        scheduler = self.build_scheduler(optimizer, epoch_steps)
        global_step = 0
        best_eval = 0.0 if self.train_config.comparator == ">" else 1e9
        for epoch in range(self.train_config.epochs):
            global_step = self.train_epoch(
                model,  # type: ignore
                data_module,
                criterion,
                optimizer,
                scheduler,
                self.train_config.grad_accumulation_steps,
                epoch,
                global_step,
                self.train_config.log_interval,
                self.train_config.log_grad_distributions,
                autocast_ctx,
                distributed_ctx,
            )
            if distributed_ctx.is_head:
                eval_results = self.evaluate(
                    unwrap_model(model),
                    data_module,
                    criterion,
                    autocast_ctx,
                    stage="val",
                    epoch=epoch,
                    global_step=global_step - 1,
                    distributed_ctx=distributed_ctx,
                )
                eval_score = eval_results[self.train_config.eval_metric]
                if eval(f"{eval_score} {self.train_config.comparator} {best_eval}"):
                    logger.info(
                        f"New best {self.train_config.eval_metric} result: {eval_score}. Saving checkpoint."
                    )
                    best_eval = eval_score
                    checkpointing.save_checkpoint(
                        unwrap_model(model),
                        optimizer,
                        scheduler,
                        data_module.generator,
                        self.architecture_name,
                        epoch,
                        global_step,
                    )
            torch.distributed.barrier()
        if distributed_ctx.is_head:
            eval_results = self.evaluate(
                unwrap_model(model),
                data_module,
                criterion,
                autocast_ctx,
                stage="test",
                epoch=self.train_config.epochs,
                global_step=global_step - 1,
                distributed_ctx=distributed_ctx,
            )
        torch.distributed.barrier()

    def mlflow_setup(self, distributed_ctx: context.DistributedContext):
        if self.mlflow_run_id is None and not distributed_ctx.is_head:
            return
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        mlflow.set_experiment(self.mlflow_config.experiment_name)
        if self.mlflow_config.enable_system_metrics:
            node_id = f"{socket.gethostname()}-rank{distributed_ctx.world_rank}"
            mlflow.set_system_metrics_node_id(node_id)
            mlflow.config.enable_system_metrics_logging()  # pyright: ignore[reportPrivateImportUsage]
            mlflow.config.set_system_metrics_sampling_interval(  # pyright: ignore[reportPrivateImportUsage]
                self.mlflow_config.system_metrics_interval
            )


def make_autocast_ctx(
    precision: Literal["fp32", "bf16"], device: torch.device
) -> ContextManager:
    use_cuda = device.type == "cuda"
    autocast_ctx: ContextManager
    match precision:
        case "bf16":
            if not use_cuda:
                logger.warning("bf16 requested but running on CPU; using fp32.")
                autocast_ctx = contextlib.nullcontext()
            else:
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("bf16 requested but not supported on this GPU.")
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        case "fp32":
            autocast_ctx = contextlib.nullcontext()
        case _:
            raise ValueError(f"precision type: {precision} is not supported.")
    return autocast_ctx


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def log_train_metrics(
    model: nn.Module,
    loss: float,
    lr: float,
    epoch: int,
    epoch_frac: float,
    step: int,
    log_distributions: bool,
) -> None:
    mlflow.log_metrics(
        {"train_loss": loss, "learning_rate": lr, "epoch": epoch, "epoch_frac": epoch_frac}, step=step
    )

    grad_logging.log_grads(unwrap_model(model), step, log_distributions)


def synchronize_int_min(val_to_sync: int, distributed_ctx: context.DistributedContext) -> int:
    # Synchronize to the shortest rank when using variable-length samplers
    if distributed_ctx.world_size > 1:
        t = torch.tensor([val_to_sync], device=distributed_ctx.device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
        val_to_sync = int(t.item())
    return val_to_sync
