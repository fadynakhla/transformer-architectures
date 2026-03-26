from typing import Any, Literal, Protocol, TypeVar
import abc

import torch
from torch.utils import data as torchd

from transformer_architectures.training.distributed import context


_T = TypeVar("_T")


class DataModule(Protocol):

    generator: torch.Generator

    _train_dataset: torchd.Dataset | None = None
    _val_dataset: torchd.Dataset | None = None
    _test_dataset: torchd.Dataset | None = None

    @abc.abstractmethod
    def setup(self, ctx: context.DistributedContext) -> None:
        ...

    @abc.abstractmethod
    def train_dataloader(self) -> torchd.DataLoader[Any]:
        ...

    @abc.abstractmethod
    def val_dataloader(self) -> torchd.DataLoader[Any]:
        ...

    @abc.abstractmethod
    def test_dataloader(self) -> torchd.DataLoader[Any]:
        ...

    def dataloader(
        self, stage: Literal["train", "val", "test"]
    ) -> torchd.DataLoader[Any]:
        match stage:
            case "train":
                return self.train_dataloader()
            case "val":
                return self.val_dataloader()
            case "test":
                return self.test_dataloader()

    @property
    def train_dataset(self) -> torchd.Dataset:
        if self._train_dataset is None:
            raise ValueError(
                f"train dataset is None! Did you run setup and ensure that it sets _train_dataset?"
            )
        return self._train_dataset

    @property
    def val_dataset(self) -> torchd.Dataset:
        if self._val_dataset is None:
            raise ValueError(
                f"val dataset is None! Did you run setup and ensure that it sets _val_dataset?"
            )
        return self._val_dataset

    @property
    def test_dataset(self) -> torchd.Dataset:
        if self._test_dataset is None:
            raise ValueError(
                f"test dataset is None! Did you run setup and ensure that it sets _test_dataset?"
            )
        return self._test_dataset


def scatter_objects(
    chunks: list[list[_T]] | None, ctx: context.DistributedContext, src: int = 0
) -> list[_T]:
    recv: list[_T | None] = [None]
    torch.distributed.scatter_object_list(recv, chunks, src=src)
    return recv[0]  # type: ignore


def broadcast_objects(
    data: list[_T] | None, ctx: context.DistributedContext, src: int = 0
) -> list[_T]:
    if ctx.world_rank == src:
        assert data is not None, "data must not be None on src worker"
        length = torch.tensor([len(data)], device=ctx.device)
    else:
        length = torch.tensor([0], device=ctx.device)
    torch.distributed.broadcast(length, src=src)
    if ctx.world_rank != src:
        data = [None] * int(length.item())  # type: ignore
    torch.distributed.broadcast_object_list(data, src=src)  # type: ignore
    return data  # type: ignore
