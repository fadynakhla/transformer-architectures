from typing import Optional
import torch
import torch.nn as nn

from transformers.util_layers import layernorm


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layer_norm = layernorm.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)
