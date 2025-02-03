from typing import List, Literal, Optional, Tuple, TypeVar
import abc

import pydantic
import torch
import torch.nn as nn

from transformer_architectures.attention import attention_functions as attn_fns
from transformer_architectures.util_layers import layernorm, residual


class MultiHeadAttentionConfig(pydantic.BaseModel):
    num_heads: int
    hidden_size: int
    attention_class: Literal["scaled_dot_product", "additive"]
    attention_dropout_prob: float


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads: int, embed_dim: int, attention: attn_fns.Attention
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
            attention_mask = self.broadcast_mask(attention_mask)
        nbatches = query.size(0)
        query, key, value = self.project(query, key, value, nbatches)
        x, attention = self.attention(query, key, value, mask=attention_mask)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.num_heads * x.size(-1))
        )
        return self.linears[-1](x), attention

    def project(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, nbatches: int
    ) -> List[torch.Tensor]:
        """Linearly project the queries, keys, and values and split the
        result into multiple heads.

        The initial shape is (batch_size, seq_len, embed_dim) and the
        resulting shape is (batch_size, num_heads, seq_len, embed_dim/num_heads).

        Args:
            query (torch.Tensor): Queries
            key (torch.Tensor): Keys
            value (torch.Tensor): Values
            nbatches (int): Number of batches

        Returns:
            List[torch.Tensor]: The projected queries, keys, and values
        """
        res = []
        for l, x in zip(self.linears, (query, key, value)):
            proj = (
                l(x)
                .view(nbatches, -1, self.num_heads, int(x.size(-1) // self.num_heads))
                .transpose(1, 2)
            )
            res.append(proj)
        return res

    def broadcast_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Make the attention mask broadcastable to the shape of the
        projected query, key, and value.

        Args:
            mask (torch.Tensor): Attention mask of shape (batch_size, seq_len)
              or (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: The broadcastable attention mask.
        """
        mask = mask.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(-2)
        return mask


class AttentionLayerConfig(pydantic.BaseModel):
    embed_dim: int
    num_heads: int
    attention_class: Literal["scaled_dot_product", "additive"]
    attention_dropout_prob: float
    pre_layernorm: bool = False


T = TypeVar("T", bound="AttentionLayerFromConfigMixin")


class AttentionLayerFromConfigMixin(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention: attn_fns.Attention,
        pre_layernorm: bool,
    ) -> None:
        """Needs to be implemented by subclasses."""

    @classmethod
    def from_config(cls: type[T], config: AttentionLayerConfig) -> T:
        """Instantiates an AttentionLayer from a configuration.

        Args:
            config (AttentionLayerConfig): Config for attention sub
              layers.

        Raises:
            ValueError: Raises value error if the attention function is
              not recognized.

        Returns:
            _type_: _description_
        """
        attention: attn_fns.Attention
        if config.attention_class == "scaled_dot_product":
            attention = attn_fns.ScaledDotProductAttention(
                config.attention_dropout_prob
            )
        elif config.attention_class == "additive":
            attention = attn_fns.AdditiveAttention(config.attention_dropout_prob)
        else:
            raise ValueError(f"Unknown attention class {config.attention_class}")
        return cls(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            attention=attention,
            pre_layernorm=config.pre_layernorm,
        )


class SelfAttentionSubLayer(nn.Module, AttentionLayerFromConfigMixin):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention: attn_fns.Attention,
        pre_layernorm: bool,
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
            attention (attn_fns.Attention): The attention
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
        self.layer_norm = layernorm.LayerNorm(embed_dim) if pre_layernorm else None
        post_layernorm = not pre_layernorm
        self.residual_connection = residual.ResidualConnection(
            hidden_size=embed_dim,
            normalize_outputs=post_layernorm,
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
            attention_mask=attention_mask,
        )
        outputs = self.residual_connection(hidden_states, attention_output)
        return outputs, attention


class CrossAttentionSubLayer(nn.Module, AttentionLayerFromConfigMixin):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention: attn_fns.Attention,
        pre_layernorm: bool,
    ) -> None:
        """Generic cross-attention sublayer.

        Typically used only in decoder stack when using an
        encoder-decoder architecture. Allows for different normalization
        schemes. e.g. the original transformer paper uses layer
        normalization on the output of the residual connection while T5
        uses layer normalization that bypasses the residual connection.

        Args:
            embed_dim (int): Number of dimensions in the input.
            num_heads (int): Numer of attention heads.
            attention (attn_fns.Attention): The attention
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
        self.layer_norm = layernorm.LayerNorm(embed_dim) if pre_layernorm else None
        post_layernorm = not pre_layernorm
        self.residual_connection = residual.ResidualConnection(
            hidden_size=embed_dim,
            normalize_outputs=post_layernorm,
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
            attention_mask=encoder_attention_mask,
        )
        outputs = self.residual_connection(hidden_states, attention_output)
        return outputs, attention
