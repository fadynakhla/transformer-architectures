from typing import Optional
import torch
import torch.nn as nn

from transformers.util_layers import layernorm


class ResidualConnection(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, dropout: Optional[float] = 0.1) -> None:
        super().__init__()
        self.layer_norm = layernorm.LayerNorm(hidden_size, eps)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.dropout:
            residual = self.dropout(residual)
        return self.layer_norm(input + residual)
