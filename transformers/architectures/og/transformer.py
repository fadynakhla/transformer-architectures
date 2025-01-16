import torch
import torch.nn as nn

from transformers import encoder, decoder
from transformers.positional_encoding import positional_encoding


class Transformer(nn.Module):
    """Implementation of the original Trasnformer"""

    def __init__(
        self,
        vocab_size: int,
        num_stacks: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pre_layernorm = False
        self.is_encoder_decoder = True
        self.learned_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.positional_encoding = positional_encoding.SinusoidalPositionalEncoding(
            embed_dim
        )
        self.encoder = encoder.Encoder(
            num_stacks=num_stacks,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            pre_layernorm=self.pre_layernorm,
        )
        self.decoder = decoder.Decoder(
            num_stacks=num_stacks,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            pre_layernorm=self.pre_layernorm,
            is_encoder_decoder=self.is_encoder_decoder,
        )
        self.lm_head = decoder.LMHead(vocab_size=vocab_size, embed_dim=embed_dim)

    def forward(
        self,
        encoder_input: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_embeddings = self.positional_encoding(
            self.learned_embeddings(encoder_input)
        )
        decoder_embeddings = self.positional_encoding(
            self.learned_embeddings(decoder_input)
        )

        encoder_output = self.encoder(input_embeddings, encoder_attention_mask)
        decoder_output = self.decoder(
            decoder_embeddings,
            decoder_attention_mask,
            encoder_output,
            encoder_attention_mask,
        )

        return self.lm_head(decoder_output)

if __name__=="__main__":
    model = Transformer(vocab_size=100, num_stacks=6, embed_dim=512, num_heads=8, ff_dim=2048, dropout=0.1)
    encoder_input = torch.rand(2, 10).long()
    encoder_attention_mask = torch.ones(2, 10)
    decoder_input = torch.rand(2, 3)
    decoder_attention_mask = torch.ones(2, 3)
