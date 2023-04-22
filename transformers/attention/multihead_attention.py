import abc
from typing import Callable, Literal, Optional, Tuple

import torch
import torch.nn as nn
import pydantic

from transformers.attention import attention_functions as attn_fns
from transformers.util_layers import layernorm, residual


class MultiHeadAttentionConfig(pydantic.BaseModel):
    num_heads: int
    hidden_size: int
    attention_class: Literal["scaled_dot_product", "additive"]
    attention_dropout_prob: float


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads: int, embed_dim: int, attention: attn_fns.AttentionFunction
    ) -> None:
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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x)
            .view(nbatches, -1, self.num_heads, int(x.size(-1) // self.num_heads))
            .transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, attention = self.attention(query, key, value, attn_mask=attention_mask)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.num_heads * x.size(-1))
        )
        return self.linears[-1](x), attention

    @classmethod
    def from_config(cls, config: MultiHeadAttentionConfig):
        if config.attention_class == "scaled_dot_product":
            attention = attn_fns.ScaledDotProductAttention(
                config.attention_dropout_prob
            )
        elif config.attention_class == "additive":
            attention = attn_fns.AdditiveAttention(config.attention_dropout_prob)
        else:
            raise ValueError(f"Unknown attention class {config.attention_class}")
        return cls(
            config.num_heads,
            config.hidden_size,
            attention=attention,
        )


class AttentionLayerConfig(pydantic.BaseModel):
    embed_dim: int
    num_heads: int
    attention_class: Literal["scaled_dot_product", "additive"]


class SelfAttentionSubLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention: attn_fns.AttentionFunction,
        normalize_inputs: bool = False,
        normalize_residual: Optional[bool] = None,
    ) -> None:
        """Generic self-attention sublayer.

        Used in both encoder and decoder stacks. Allows for different
        normalization schemes e.g. the original transformer paper uses
        layer normalization on the output of the residual connection
        while T5 uses layer normalization that bypasses the residual
        connection.

        Args:
            embed_dim (int): Number of dimensions in the input.
            num_heads (int): Numer of attention heads.
            attention (attn_fns.AttentionFunction): The attention
              function to use.
            normalize_inputs (bool, optional): Whether to normalize the
              inputs before passing them to multihead attention.
              Defaults to False.
            normalize_residual (Optional[bool], optional): Whether to
              normalize the output of the residual connection. Typically
              determined by normalize_inputs. Defaults to None.
        """
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, embed_dim, attention)
        self.layer_norm = layernorm.LayerNorm(embed_dim) if normalize_inputs else None
        normalize_residual = (
            normalize_residual
            if normalize_residual is not None
            else not normalize_inputs
        )
        self.residual_connection = residual.ResidualConnection(
            hidden_size=embed_dim,
            normalize_outputs=normalize_residual,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed_hidden_states = hidden_states
        if self.layer_norm:
            normed_hidden_states = self.layer_norm(normed_hidden_states)
        attention_output, attention = self.attention(
            normed_hidden_states,
            normed_hidden_states,
            normed_hidden_states,
            attn_mask=attention_mask,
        )
        outputs = self.residual_connection(hidden_states, attention_output)
        return outputs, attention


class CrossAttentionSubLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention: attn_fns.AttentionFunction,
        normalize_inputs: bool = False,
        normalize_residual: Optional[bool] = None,
    ) -> None:
        """Generic cross-attention sublayer.

        Typically used only in decoder stack when using an
        encoder-decoder architecture. Allows for different normalization
        schemes e.g. the original transformer paper uses layer
        normalization on the output of the residual connection while T5
        uses layer normalization that bypasses the residual connection.

        Args:
            embed_dim (int): Number of dimensions in the input.
            num_heads (int): Numer of attention heads.
            attention (attn_fns.AttentionFunction): The attention
              function to use.
            normalize_inputs (bool, optional): Whether to normalize the
              inputs before passing them to multihead attention.
              Defaults to False.
            normalize_residual (Optional[bool], optional): Whether to
              normalize the output of the residual connection. Typically
              determined by normalize_inputs. Defaults to None.
        """
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, embed_dim, attention)
        self.layer_norm = layernorm.LayerNorm(embed_dim) if normalize_inputs else None
        normalize_residual = (
            normalize_residual
            if normalize_residual is not None
            else not normalize_inputs
        )
        self.residual_connection = residual.ResidualConnection(
            hidden_size=embed_dim,
            normalize_outputs=normalize_residual,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed_hidden_states = hidden_states
        if self.layer_norm:
            normed_hidden_states = self.layer_norm(normed_hidden_states)
        attention_output, attention = self.attention(
            normed_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            attn_mask=encoder_attention_mask,
        )
        outputs = self.residual_connection(hidden_states, attention_output)
        return outputs, attention
