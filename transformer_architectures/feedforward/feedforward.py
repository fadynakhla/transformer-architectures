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
        layers = [
            nn.Linear(hidden_size, ff_size),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
        ]
        self.layers = nn.ModuleList(layers)
        self.layer_norm = layernorm.LayerNorm(hidden_size) if pre_layernorm else None
        post_layernorm = not pre_layernorm
        self.residual_connection = residual.ResidualConnection(
            hidden_size=hidden_size,
            normalize_outputs=post_layernorm,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states
        if self.layer_norm:
            output = self.layer_norm(output)
        for layer in self.layers:
            output = layer(output)
        output = self.residual_connection(hidden_states, output)
        return output
