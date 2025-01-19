import torch


def causal_mask(input: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    sequence_length = input.size(-2)
    shape = (1, sequence_length, sequence_length)
    ones = torch.ones((1, sequence_length, sequence_length), dtype=torch.uint8, device=input.device)
    mask = torch.tril(ones, diagonal=diagonal)  # lower triangular part has ones
    return mask
