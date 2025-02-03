from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformer_architectures.attention import multihead_attention
from transformer_architectures.feedforward import feedforward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
        pre_layernorm: bool = False,
    ) -> None:
        super().__init__()
        attention_config = multihead_attention.AttentionLayerConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_class="scaled_dot_product",
            attention_dropout_prob=dropout,
            pre_layernorm=pre_layernorm,  # Will be configurable in the future current setting is for og transformer
        )
        self.self_attention = multihead_attention.SelfAttentionSubLayer.from_config(
            attention_config
        )
        self.ff = feedforward.FeedForwardSubLayer(
            hidden_size=embed_dim,
            ff_size=ff_dim,
            pre_layernorm=pre_layernorm,
            dropout=dropout,
        )

        self.cross_attention = (
            (multihead_attention.CrossAttentionSubLayer.from_config(attention_config))
            if is_decoder
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        x, self_attention_tensor = self.self_attention(hidden_states, attention_mask)
        if encoder_hidden_states is not None:
            assert self.cross_attention, "Got encoder embeddings"
            x, cross_attention_tensor = self.cross_attention(
                x, encoder_hidden_states, encoder_attention_mask
            )
        x = self.ff(x)
        outputs = (x, self_attention_tensor)
        if encoder_hidden_states is not None:
            return outputs + (cross_attention_tensor,)
        return outputs
