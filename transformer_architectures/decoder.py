from typing import Optional
import torch
import torch.nn as nn

from transformer_architectures import transformer_blocks
from transformer_architectures.util_layers import layernorm


class Decoder(nn.Module):
    def __init__(
        self,
        num_stacks: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        pre_layernorm: bool = False,
        is_encoder_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.decoder_stack = nn.ModuleList(
            [
                transformer_blocks.TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    is_decoder=is_encoder_decoder,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_stacks)
            ]
        )
        self.layer_norm = layernorm.LayerNorm(embed_dim) if pre_layernorm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        encoder_output: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the decoder.

        Args:
            hidden_states (torch.Tensor): input to the decoder
            attention_mask (torch.Tensor): attention mask for the
              decoder input hidden_states
            encoder_output (torch.Tensor): output of the encoder
            encoder_attention_mask (torch.Tensor): attention mask for
              the encoder output

        Returns:
            torch.Tensor: decoder output
        """
        for decoder_block in self.decoder_stack:
            hidden_states, _ = decoder_block(
                hidden_states, attention_mask, encoder_output, encoder_attention_mask
            )
        if self.layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class LMHead(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.lm_head(hidden_states), dim=-1)


class DecoderForGeneration(Decoder):
    def __init__(
        self,
        vocab_size: int,
        num_stacks: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        pre_layernorm: bool = False,
        is_encoder_decoder: bool = False,
    ) -> None:
        super().__init__(
            num_stacks,
            embed_dim,
            num_heads,
            ff_dim,
            dropout,
            pre_layernorm,
            is_encoder_decoder,
        )
        self.lm_head = LMHead(vocab_size, embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        encoder_output: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = super().forward(
            hidden_states, attention_mask, encoder_output, encoder_attention_mask
        )
        return self.lm_head(hidden_states)
