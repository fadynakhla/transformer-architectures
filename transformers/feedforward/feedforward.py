from typing import Optional
import torch
import torch.nn as nn

from transformers.util_layers import layernorm, residual


class FeedForwardSubLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ff_size: int,
        dropout: float = 0.1,
        normalize_inputs: bool = False,
        normalize_residual: Optional[bool] = None,
    ):
        super().__init__()
        layers = [
            nn.Linear(hidden_size, ff_size),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
        ]
        self.layers = nn.ModuleList(layers)
        self.layer_norm = layernorm.LayerNorm(hidden_size) if normalize_inputs else None
        normalize_residual = (
            normalize_residual
            if normalize_residual is not None
            else not normalize_inputs
        )
        self.residual_connection = residual.ResidualConnection(
            hidden_size=hidden_size,
            normalize_outputs=normalize_residual,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states
        if self.layer_norm:
            output = self.layer_norm(output)
        for layer in self.layers:
            output = layer(output)
        output = self.residual_connection(hidden_states, output)
        return output
