from typing import Optional
import torch
import torch.nn as nn

from transformers import transformer_blocks
from transformers.util_layers import layernorm


class Encoder(nn.Module):
    def __init__(
        self,
        num_stacks: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        pre_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_stack = nn.ModuleList(
            [
                transformer_blocks.TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    is_decoder=False,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_stacks)
            ]
        )
        self.layer_norm = layernorm.LayerNorm(embed_dim) if pre_layernorm else None

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass of the encoder

        Args:
            x (torch.Tensor): _description_
            attention_mask (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        for encoder_block in self.encoder_stack:
            hidden_states, _ = encoder_block(hidden_states, attention_mask)
        if self.layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        return hidden_states
