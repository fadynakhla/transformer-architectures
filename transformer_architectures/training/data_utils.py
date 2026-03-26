from typing import Any, Literal, Protocol, Sequence, TypeVar, runtime_checkable
import abc
import math
import random

import torch
from torch.utils import data as torchd

from transformer_architectures.training.distributed.context import DistributedContext

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)



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
