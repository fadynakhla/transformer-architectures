from typing import Optional

import torch


class GenerationMixin:
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_input_ids: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
        max_length: int = 20,
    ) -> torch.Tensor:
        return torch.randn(1, 1, 1)
