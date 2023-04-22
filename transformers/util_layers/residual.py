from typing import Optional
import torch
import torch.nn as nn

from transformers.util_layers import layernorm


class ResidualConnection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        normalize_outputs: bool,
        eps: float = 1e-5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layer_norm = (
            layernorm.LayerNorm(hidden_size, eps) if normalize_outputs else None
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        residual = self.dropout(residual)
        output = input + residual
        if self.layer_norm:
            output = self.layer_norm(output)
        return output
