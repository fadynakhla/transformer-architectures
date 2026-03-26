from typing import Any, Literal, Protocol, Sequence, TypeVar, runtime_checkable
import abc
import math
import random

import torch
from torch.utils import data as torchd

from transformer_architectures.training import distributed

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


class DataModule(Protocol):

    generator: torch.Generator

    _train_dataset: torchd.Dataset | None = None
    _val_dataset: torchd.Dataset | None = None
    _test_dataset: torchd.Dataset | None = None

    @abc.abstractmethod
    def setup(self, ctx: distributed.DistributedContext) -> None:
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


@runtime_checkable
class HasLen(Protocol):
    def __len__(self) -> int:
        ...


DataSplit = tuple[torchd.Subset[_T_co], torchd.Subset[_T_co], torchd.Subset[_T_co]]


def train_val_test_split_torchd(
    dataset: torchd.Dataset[_T_co],
    val_split: float,
    test_split: float,
    generator: torch.Generator,
) -> DataSplit[_T_co]:
    if not isinstance(dataset, HasLen):
        raise ValueError("Dataset must implement __len__")
    total_size = len(dataset)
    val_size = math.floor(total_size * val_split)
    test_size = math.floor(total_size * test_split)
    train_size = total_size - val_size - test_size
    subs = torchd.random_split(dataset, [train_size, val_size, test_size], generator)
    return subs[0], subs[1], subs[2]


def train_val_test_split(
    data: Sequence[_T],
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> tuple[list[_T], list[_T], list[_T]]:
    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)
    val_size = math.floor(len(data) * val_split)
    test_size = math.floor(len(data) * test_split)
    test = [data[i] for i in indices[:test_size]]
    val = [data[i] for i in indices[test_size : test_size + val_size]]
    train = [data[i] for i in indices[test_size + val_size :]]
    return train, val, test


def split_into_chunks(data: Sequence[_T], world_size: int) -> list[list[_T]]:
    chunk_len, remainder = divmod(len(data), world_size)
    data = list(data[: len(data) - remainder])
    chunks: list[list[_T]] = []
    for i in range(world_size):
        start = i * chunk_len
        end = (i + 1) * chunk_len
        chunks.append(data[start:end])
    return chunks


def scatter_objects(
    chunks: list[list[_T]] | None, ctx: distributed.DistributedContext, src: int = 0
) -> list[_T]:
    recv: list[_T | None] = [None]
    torch.distributed.scatter_object_list(recv, chunks, src=src)
    return recv[0]  # type: ignore


def broadcast_objects(
    data: list[_T] | None, ctx: distributed.DistributedContext, src: int = 0
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
