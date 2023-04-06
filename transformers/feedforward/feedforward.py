from typing import Optional
import torch
import torch.nn as nn

from transformers.util_layers import layernorm


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1, normalize_inputs: bool = False):
        super().__init__()
        layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        ]
        self.layers = nn.ModuleList(layers)
        self.layer_norm = layernorm.LayerNorm(hidden_size) if normalize_inputs else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states
        if self.layer_norm:
            output = self.layer_norm(output)
        for layer in self.layers:
            output = layer(output)
        return output
