from typing import Dict, List, Optional

import torch
import tiktoken


class Tokenizer:

    def __init__(self, special_tokens: Dict[str, int]) -> None:
        self.encoding = tiktoken.Encoding(name="transformer", )

    def __call__(
        self,
        encoder_inputs: List[str],
        decoder_inputs: List[str],
        padding: Optional[str] = None,
        truncation: bool = False
    ) -> Dict[str, torch.Tensor]:
        encoder_input_ids = tiktoken.Encoding
