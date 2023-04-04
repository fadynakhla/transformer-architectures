import abc
from typing import Callable, Literal, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFunction(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, dropout: Optional[float] = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else None

    @abc.abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


class ScaledDotProductAttention(AttentionFunction):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        if self.dropout:
            attention = self.dropout(attention)
        return torch.matmul(attention, value), attention


class AdditiveAttention(AttentionFunction):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("AdditiveAttention not implemented yet.") # TODO: Implement this.
