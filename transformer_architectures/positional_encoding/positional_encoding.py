import abc
import torch
import torch.nn as nn


POS_ENCODING_PERIOD = 10000


class PositionalEncoding(abc.ABC, nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the positional encoding.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: sum of input and positional encoding
        """
        return x + self.positional_encoding(x)

    @abc.abstractmethod
    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Abstract method for generating the positional encoding.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: positional encoding
        """
        pass


class SinusoidalPositionalEncoding(PositionalEncoding):
    """Generates the positional encoding for the transformer following
    the method described in the paper "Attention Is All You Need".
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__(embed_dim)
        self.set_encoding_period()

    def set_encoding_period(self, period: int = POS_ENCODING_PERIOD) -> None:
        self.period = period

    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Generates the positional encoding according to the sinusoidal
        method described in "Attension is All You Need".

        Args:
            x (torch.Tensor): input tensor (typically shape (batch_size, seq_len, embed_dim))

        Returns:
            torch.Tensor: positional encoding
        """
        _, seq_len, _ = x.shape
        pos = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
        # We could write this naively with the following:
        # div_term = 1 / (POS_ENCODING_PERIOD ** (torch.arange(0, embed_dim, 2) / embed_dim))
        #
        # or we can use chained log and exp instead since log(1/(a^b)) = -b*log(a)
        div_term = torch.exp(
            -(
                (torch.arange(0, self.embed_dim, 2) / self.embed_dim)
                * torch.log(self.period)
            )
        )

        encodings = torch.zeros((seq_len, self.embed_dim))
        encodings[:, 0::2] = torch.sin(pos * div_term)
        encodings[:, 1::2] = torch.cos(pos * div_term)
        return encodings.unsqueeze(0)
