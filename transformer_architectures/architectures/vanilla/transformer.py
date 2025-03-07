import torch
from torch import nn

from transformer_architectures import decoder, encoder, masking
from transformer_architectures.positional_encoding import positional_encoding


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
        self.vocab_size = vocab_size
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
        decoder_attention_mask = self.suplement_causal_mask(decoder_attention_mask)
        decoder_output = self.decoder(
            decoder_embeddings,
            decoder_attention_mask,
            encoder_output,
            encoder_attention_mask,
        )

        return self.lm_head(decoder_output)

    def suplement_causal_mask(
        self, decoder_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask_dim = decoder_attention_mask.dim()
        causal_mask = masking.causal_mask(decoder_attention_mask)
        match mask_dim:
            # single inputs should still be "batched"
            # case 1:
            #     return decoder_attention_mask & causal_mask.squeeze(0)
            case 2:
                return decoder_attention_mask.unsqueeze(-2) & causal_mask
            case 3:
                return decoder_attention_mask & causal_mask
            case _:
                raise ValueError(f"Invalid dimension of input mask: {mask_dim}")


if __name__ == "__main__":
    device = torch.device("cuda")
    model = Transformer(
        vocab_size=50000,
        num_stacks=6,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
    ).to(device=device)
    for _ in range(1000):
        encoder_input = torch.rand(4, 1000).long().to(device=device)
        encoder_attention_mask = torch.ones(4, 1000).long().to(device=device)
        decoder_input = torch.rand(4, 587).long().to(device=device)
        decoder_attention_mask = torch.ones(4, 587).long().to(device=device)
        output = model(
            encoder_input, encoder_attention_mask, decoder_input, decoder_attention_mask
        )
    print(f"Model output shape: {output.shape}")
