import pydantic
import torch
from torch import nn

from transformer_architectures.util_layers import layernorm, residual


class FeedForwardLayerConfig(pydantic.BaseModel):
    hidden_size: int
    ff_size: int
    pre_layernorm: bool = False
    dropout: float = 0.1


class FeedForwardSubLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ff_size: int,
        pre_layernorm: bool,
        dropout: float,
    ):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
        )
        self.pre_layernorm = pre_layernorm
        self.layer_norm = layernorm.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states
        if self.pre_layernorm:
            output = self.layer_norm(output)
        output = self.ffn(output)
        output = hidden_states + self.dropout(output)
        if not self.pre_layernorm:
            output = self.layer_norm(output)
        return output
