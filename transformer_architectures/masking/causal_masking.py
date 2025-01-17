import torch
import torch.nn as nn


def causal_mask(input: torch.Tensor, diagonal: int) -> torch.Tensor:
    sequence_length = input.size(-2)
    ones = torch.ones((1, sequence_length, sequence_length)).to(input.device)
    mask = torch.tril(ones, diagonal=diagonal)  # upper triangular part has ones
    return mask
