import torch


def causal_mask(tensor: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    shape = (tensor.size(-1), tensor.size(-1))
    ones = torch.ones(shape, dtype=torch.uint8, device=tensor.device)
    mask = torch.tril(ones, diagonal=diagonal).unsqueeze(0)
    return mask
