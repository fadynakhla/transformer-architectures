from typing import Optional
import torch
from torch import Tensor, nn

from transformer_architectures import decoder, embedding, encoder, masking
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
        self.pre_layernorm = True
        self.is_encoder_decoder = True
        self.vocab_size = vocab_size
        self.encoder_embeddings = embedding.Embedding(
            vocab_size=vocab_size, embed_dim=embed_dim
        )
        self.encoder_pe = positional_encoding.SinusoidalPositionalEncoding(
            embed_dim, dropout=dropout
        )
        self.decoder_embeddings = embedding.Embedding(
            vocab_size=vocab_size, embed_dim=embed_dim
        )
        self.decoder_pe = positional_encoding.SinusoidalPositionalEncoding(
            embed_dim, dropout=dropout
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
        # self.tie_weights()

    def forward(
        self,
        encoder_input: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_out = self.encode(encoder_input, encoder_attention_mask)
        return self.decode(
            decoder_input, decoder_attention_mask, encoder_out, encoder_attention_mask
        )

    def encode(
        self, encoder_input: torch.Tensor, encoder_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode the input sequence."""
        input_embeddings = self.encoder_embeddings(encoder_input)
        input_embeddings = self.encoder_pe(input_embeddings)
        return self.encoder(input_embeddings, encoder_attention_mask)

    def decode(
        self,
        decoder_input: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode the input sequence."""
        decoder_embeddings = self.decoder_embeddings(decoder_input)
        decoder_embeddings = self.decoder_pe(decoder_embeddings)
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

    def tie_weights(self) -> None:
        """Tie weights of lm head and embeddings"""
        self.encoder_embeddings.embeddings.weight = (
            self.decoder_embeddings.embeddings.weight
        )
        self.lm_head.lm_head.weight = self.decoder_embeddings.embeddings.weight


if __name__ == "__main__":
    import time

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Transformer(
        vocab_size=50000,
        num_stacks=6,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
    ).to(device=device)
    iterations = 1000
    start = time.monotonic()
    for _ in range(iterations):
        encoder_input = torch.rand(8, 1000).long().to(device=device)
        encoder_attention_mask = torch.ones(8, 1000).long().to(device=device)
        decoder_input = torch.rand(8, 687).long().to(device=device)
        decoder_attention_mask = torch.ones(8, 687).long().to(device=device)
        output = model(
            encoder_input, encoder_attention_mask, decoder_input, decoder_attention_mask
        )
    total = time.monotonic() - start
    print("Run Statistics:")
    print(f"Total time (s): {total:.2f}")
    print(f"Iterations: {iterations}")
    speed = iterations / total if iterations >= total else total / iterations
    units = "iter/sec" if iterations >= total else "sec/iter"
    print(f"Speed ({units}): {speed:.2f}")
    print(f"Device: {device}")
    print(f"Model output shape: {output.shape}")
