from typing import Literal, Optional, Tuple
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: Optional[float] = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        if self.dropout:
            attention = self.dropout(attention)
        return torch.matmul(attention, value), attention


class AdditiveAttention(nn.Module):
    def __init__(self, dropout: Optional[float] = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("AdditiveAttention not implemented yet.")


class MultiHeadAttentionConfig(pydantic.BaseModel):
    num_heads: int
    hidden_size: int
    attention_class: Literal["scaled_dot_product", "additive"]
    attention_dropout_prob: float


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, attention: nn.Module) -> None:
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(4)]
        )
        self.num_heads = num_heads
        self.attention = attention

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x)
            .view(nbatches, -1, self.num_heads, int(x.size(-1) // self.num_heads))
            .transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, attention = self.attention(query, key, value, attn_mask=attn_mask)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.num_heads * x.size(-1))
        )
        return self.linears[-1](x), attention

    @classmethod
    def from_config(cls, config: MultiHeadAttentionConfig):
        if config.attention_class == "scaled_dot_product":
            attention = ScaledDotProductAttention(config.attention_dropout_prob)
        elif config.attention_class == "additive":
            attention = AdditiveAttention(config.attention_dropout_prob)
        else:
            raise ValueError(f"Unknown attention class {config.attention_class}")
        return cls(
            config.num_heads,
            config.hidden_size,
            attention=attention,
        )


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attention: nn.Module) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, embed_dim, attention)
        self.layer_norm = nn.LayerNorm(embed_dim)
