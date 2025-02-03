import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        """Initialize LayerNorm module.

        Args:
            hidden_size (int): Size of input tensor.
            eps (float, optional): Add to std denominator to prevent div
              by zero. Defaults to 1e-5.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of LayerNorm.

        Normalize input tensor with its mean and standard deviation
        scaled by learned weight and bias.

        Args:
            input (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: normed and scaled tensor.
        """
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)
        return self.weight * (input - mean) / (std + self.eps) + self.bias
