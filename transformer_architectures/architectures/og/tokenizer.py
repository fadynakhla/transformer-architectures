from typing import Dict, List, Literal, Optional

import torch

from transformer_architectures import tokenization
from transformer_architectures import masking


PaddingOptions = Literal["longest", "max"]


class Tokenizer(tokenization.BaseTokenizer):
    def __init__(
        self,
        base_encoding_name: str,
        model_max_len: int,
        pad_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        additional_special_tokens: Optional[set[str]] = None,
    ) -> None:
        pad_token = pad_token or "<pad>"
        eos_token = eos_token or "<eos>"
        super().__init__(base_encoding_name, model_max_len, pad_token, eos_token, additional_special_tokens)

    def __call__(
        self,
        encoder_inputs: List[str],
        decoder_inputs: List[str],
        padding: Optional[PaddingOptions] = None,
        truncation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.batch_encode(encoder_inputs, decoder_inputs, padding, truncation)

    def batch_encode(
        self,
        encoder_inputs: List[str],
        decoder_inputs: List[str],
        padding: Optional[PaddingOptions] = None,
        truncation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encoder_input_ids_list = self.encoding.encode_batch(encoder_inputs)
        decoder_input_ids_list = self.encoding.encode_batch(decoder_inputs)
        if padding:
            self._pad_and_truncate(encoder_input_ids_list, padding, truncation)
            self._pad_and_truncate(decoder_input_ids_list, padding, truncation)

    def _pad_and_truncate(
        self, input_ids_list: list[list[int]], padding: PaddingOptions, truncation: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad_length = self._determine_pad_length(input_ids_list, padding, truncation)
        padded_inputs = [
            (seq + [self.pad_token_id] * max(pad_length - len(seq), 0))[:pad_length]
            for seq in input_ids_list
        ]
        input_ids = torch.LongTensor(padded_inputs)
        attention_mask = (input_ids != self.pad_token_id).type(torch.uint8)
        return input_ids, attention_mask

    # def _pad_and_truncate_decode(
    #     self, input_ids_list: list[list[int]], padding: PaddingOptions, truncation: bool
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     input_ids, pad_attention_mask = self._pad_and_truncate(
    #         input_ids_list, padding, truncation
    #     )
    #     return input_ids, pad_attention_mask &


    def _determine_pad_length(
        self, input_ids: list[list[int]], padding: PaddingOptions, truncation: bool
    ) -> int:
        match padding:
            case "max":
                pad_len = self.model_max_len
            case "longest":
                pad_len = max([len(seq) for seq in input_ids])
            case _:
                raise ValueError(
                    f"Invalid padding option. Must be one of {PaddingOptions}"
                )
        if pad_len > self.model_max_len and truncation:
            pad_len = self.model_max_len
        return pad_len
