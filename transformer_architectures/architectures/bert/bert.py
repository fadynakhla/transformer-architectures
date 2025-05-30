import torch
from torch import nn

from transformer_architectures import embedding, encoder
from transformer_architectures.positional_encoding import positional_encoding


class Bert(nn.Module):
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
        self.vocab_size = vocab_size
        self.embeddings = nn.Sequential(
            embedding.Embedding(vocab_size=vocab_size, embed_dim=embed_dim),
            positional_encoding.SinusoidalPositionalEncoding(
                embed_dim, dropout=dropout
            ),
        )
        self.encoder = encoder.Encoder(
            num_stacks=num_stacks,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            pre_layernorm=self.pre_layernorm,
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        encodings = self.encode(input_ids, attention_mask)
        return encodings

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode the input sequence."""
        input_embeddings = self.embeddings(input_ids)
        return self.encoder(input_embeddings, attention_mask)


class BertForSeqClassification(Bert):
    def __init__(
        self,
        vocab_size: int,
        num_stacks: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(vocab_size, num_stacks, embed_dim, num_heads, ff_dim, dropout)
        # self.prediction_head: nn.Module = nn.Linear(embed_dim, 1)
