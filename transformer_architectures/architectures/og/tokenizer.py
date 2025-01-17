from typing import Dict, List, Optional

import torch

from transformer_architectures import tokenization


class Tokenizer(tokenization.BaseTokenizer):

    def __call__(
        self,
        encoder_inputs: List[str],
        decoder_inputs: List[str],
        padding: Optional[str] = None,
        truncation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.batch_encode(encoder_inputs, decoder_inputs, padding, truncation)

    def batch_encode(
        self,
        encoder_inputs: List[str],
        decoder_inputs: List[str],
        padding: Optional[str] = None,
        truncation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encoder_input_ids_list = self.encoding.encode_batch(encoder_inputs)
        decoder_input_ids_list = self.encoding.encode_batch(decoder_inputs)
        if padding:
            self._pad_and_truncate(encoder_input_ids_list, padding, truncation)
            self._pad_and_truncate(decoder_input_ids_list, padding, truncation)

    def _pad_and_truncate(input_ids: list[list[int]], padding: Optional[str], truncation: bool) -> tuple[torch.Tensor, torch.Tensor]:
        ...
