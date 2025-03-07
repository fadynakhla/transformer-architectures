from typing import Generic, Literal, Optional, TypeVar, overload

import pydantic
import torch

from transformer_architectures import tokenization

PaddingOptions = Literal["longest", "max"]

Enc = TypeVar("Enc", list[list[int]], torch.Tensor)
Dec = TypeVar("Dec", list[list[int]], torch.Tensor)
Mask = TypeVar("Mask", None, torch.Tensor)
DecMask = TypeVar("DecMask", None, torch.Tensor)


class BatchEncoding(pydantic.BaseModel, Generic[Enc, Dec, Mask, DecMask]):
    """Inspired by huggingface's approach to tokenizer output"""

    input_ids: Enc
    decoder_input_ids: Dec
    attention_mask: Mask
    decoder_attention_mask: DecMask

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


TensorBatchEncoding = BatchEncoding[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class Tokenizer(tokenization.BaseTokenizer):
    pad_token: str
    pad_token_id: int
    bos_token: str
    bos_token_id: int
    eos_token: str
    eos_token_id: int

    def __init__(
        self,
        base_encoding_name: str,
        model_max_len: int,
        pad_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        additional_special_tokens: Optional[set[str]] = None,
    ) -> None:
        pad_token = pad_token or "<pad>"
        bos_token = bos_token or "<bos>"
        eos_token = eos_token or "<eos>"
        super().__init__(
            base_encoding_name,
            model_max_len,
            pad_token,
            bos_token,
            eos_token,
            additional_special_tokens,
        )

    @overload
    def __call__(
        self,
        encoder_inputs: list[str],
        decoder_inputs: list[str],
        padding: Optional[PaddingOptions] = None,
        truncation: bool = False,
        return_tensors: Literal[False] = False,
    ) -> BatchEncoding[list[list[int]], list[list[int]], None, None]:
        ...

    @overload
    def __call__(
        self,
        encoder_inputs: list[str],
        decoder_inputs: list[str],
        padding: Optional[PaddingOptions] = None,
        truncation: bool = False,
        return_tensors: Literal[True] = True,
    ) -> BatchEncoding[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def __call__(
        self,
        encoder_inputs: list[str],
        decoder_inputs: list[str],
        padding: Optional[PaddingOptions] = None,
        truncation: bool = False,
        return_tensors: bool = False,
    ) -> BatchEncoding:
        return self.batch_encode(
            encoder_inputs, decoder_inputs, padding, truncation, return_tensors
        )

    @overload
    def batch_encode(
        self,
        encoder_inputs: list[str],
        decoder_inputs: list[str],
        padding: Optional[PaddingOptions] = None,
        truncation: bool = False,
        return_tensors: Literal[False] = False,
    ) -> BatchEncoding[list[list[int]], list[list[int]], None, None]:
        ...

    @overload
    def batch_encode(
        self,
        encoder_inputs: list[str],
        decoder_inputs: list[str],
        padding: Optional[PaddingOptions] = None,
        truncation: bool = False,
        return_tensors: Literal[True] = True,
    ) -> BatchEncoding[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    @overload
    def batch_encode(
        self,
        encoder_inputs: list[str],
        decoder_inputs: list[str],
        padding: Optional[PaddingOptions] = None,
        truncation: bool = False,
        return_tensors: bool = ...,
    ) -> BatchEncoding:
        ...

    def batch_encode(
        self,
        encoder_inputs: list[str],
        decoder_inputs: list[str],
        padding: Optional[PaddingOptions] = None,
        truncation: bool = False,
        return_tensors: bool = False,
    ) -> BatchEncoding:
        input_ids: list[list[int]] | torch.Tensor
        decoder_input_ids: list[list[int]] | torch.Tensor
        input_ids = self.encoding.encode_batch(encoder_inputs)
        decoder_input_ids = self.encoding.encode_batch(decoder_inputs)
        input_ids = [
            [self.bos_token_id] + input_ids + [self.eos_token_id]
            for input_ids in input_ids
        ]
        decoder_input_ids = [
            [self.bos_token_id] + input_ids + [self.eos_token_id]
            for input_ids in decoder_input_ids
        ]

        if padding:
            input_ids = self._pad_and_truncate(input_ids, padding, truncation)
            decoder_input_ids = self._pad_and_truncate(
                decoder_input_ids, padding, truncation
            )

        mask: Optional[torch.Tensor] = None
        decoder_mask: Optional[torch.Tensor] = None
        if return_tensors:
            input_ids, mask = self._tensorize(input_ids)
            decoder_input_ids, decoder_mask = self._tensorize(decoder_input_ids)

        return BatchEncoding(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=mask,
            decoder_attention_mask=decoder_mask,
        )

    def batch_decode(self, token_ids: list[list[int]] | torch.Tensor) -> list[str]:
        token_ids = (
            token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        )
        token_ids = [
            [tid for tid in seq if tid not in self.special_tokens.values()]
            for seq in token_ids
        ]
        return self.encoding.decode_batch(token_ids, errors="replace")

    def pad(
        self,
        batch: list[dict[str, list[int]]],
        padding: PaddingOptions,
        truncation: bool,
        pad_to_multiple_of: Optional[int],
    ) -> TensorBatchEncoding:
        """Pad and tensorize a batch.

        Args:
            batch (list[dict[str, list[int]]]): Batch of tokenized data.
            padding (PaddingOptions): How to pad, "longest" or "max".
            truncation (bool): Whether the result should be truncated.
            pad_to_multiple_of (Optional[int]): If provided, the resulting
              sequence lengths will be a multiple of the given value.

        Returns:
            TensorBatchEncoding: Padded and tensorized Batch Encoding.
        """
        padded_input = self._pad_and_truncate(
            [sample["input_ids"] for sample in batch],
            padding,
            truncation,
            pad_to_multiple_of,
        )
        input_ids, attention_mask = self._tensorize(padded_input)
        padded_dec_inp = self._pad_and_truncate(
            [sample["decoder_input_ids"] for sample in batch],
            padding,
            truncation,
            pad_to_multiple_of,
        )
        decoder_input_ids, decoder_attention_mask = self._tensorize(padded_dec_inp)
        return TensorBatchEncoding(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

    def _tensorize(
        self, input_ids_list: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.LongTensor(input_ids_list)
        attention_mask = (input_ids != self.pad_token_id).type(torch.uint8)
        return input_ids, attention_mask

    def _pad_and_truncate(
        self,
        input_ids_list: list[list[int]],
        padding: PaddingOptions,
        truncation: bool,
        pad_to_multiple_of: Optional[int] = None,
    ) -> list[list[int]]:
        pad_length = self._determine_pad_length(
            input_ids_list, padding, truncation, pad_to_multiple_of
        )
        padded_inputs = [
            (seq + [self.pad_token_id] * max(pad_length - len(seq), 0))[:pad_length]
            for seq in input_ids_list
        ]
        return padded_inputs

    def _determine_pad_length(
        self,
        input_ids: list[list[int]],
        padding: PaddingOptions,
        truncation: bool,
        pad_to_multiple_of: Optional[int],
    ) -> int:
        match padding:
            case "max":
                pad_len = self.model_max_len
            case "longest":
                pad_len = max(len(seq) for seq in input_ids)
            case _:
                raise ValueError(
                    f"Invalid padding option. Must be one of {PaddingOptions}"
                )
        if pad_to_multiple_of and (rem := pad_len % pad_to_multiple_of) != 0:
            assert (
                self.model_max_len % pad_to_multiple_of == 0
            ), "Model max len must be multiple of pad_to_multiple_of"
            pad_len += pad_to_multiple_of - rem

        if pad_len > self.model_max_len and truncation:
            pad_len = self.model_max_len
        return pad_len


if __name__ == "__main__":
    tokenizer = Tokenizer(
        "r50k_base", 512, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>"
    )
    enc = tokenizer(
        ["hello", "world"], ["to the world", "to the hello"], return_tensors=True
    )
    print(enc)

    print(tokenizer.encoding.special_tokens_set)
    print(tokenizer.batch_decode(enc.input_ids))
    print(tokenizer.batch_decode(enc.decoder_input_ids))
