from typing import Optional
import torch
import torch.nn as nn
from transformers.attention import multihead_attention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
    ) -> None:
        super().__init__()
        attention_config = multihead_attention.MultiHeadAttentionConfig(
            num_heads=num_heads,
            hidden_size=embed_dim,
            attention_class="scaled_dot_product",
            attention_dropout_prob=dropout,
        )
        self.self_attention = multihead_attention.MultiHeadAttention.from_config(
            attention_config
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        if is_decoder:
            self.cross_attention = multihead_attention.MultiHeadAttention.from_config(
                attention_config
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        attention_mask = (
            attention_mask.unsqueeze(1).unsqueeze(2) if attention_mask else None
        )
        x = x + self.dropout(
            self.attention(
                self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attention_mask
            )[0]
        )
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class TransformerBlock2(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
    ) -> None:
        super().__init__()
        attention_config = multihead_attention.MultiHeadAttentionConfig(
            num_heads=num_heads,
            hidden_size=embed_dim,
            attention_class="scaled_dot_product",
            attention_dropout_prob=dropout,
        )
        self.self_attention = multihead_attention.MultiHeadAttention.from_config(
            attention_config
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        if is_decoder:
            self.cross_attention = multihead_attention.MultiHeadAttention.from_config(
                attention_config
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        attention_mask = (
            attention_mask.unsqueeze(1).unsqueeze(2) if attention_mask else None
        )
        x = x + self.dropout(
            self.attention(
                self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attention_mask
            )[0]
        )
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x
