from typing import Generic, Optional, TypeVar
import pydantic
import torch
from torch.utils import data as torchd

from transformer_architectures.architectures.vanilla import tokenization


IGNORE_ID = -100

Label = TypeVar("Label", None, torch.Tensor)
LabelMask = TypeVar("LabelMask", None, torch.Tensor)


class SourceTarget(pydantic.BaseModel):
    source: str
    target: str


class LabeledBatch(pydantic.BaseModel):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    target: torch.Tensor
    target_ignore_mask: torch.Tensor

    @classmethod
    def from_batch_encoding(
        cls, be: tokenization.TensorBatchEncoding, ignore_id: int
    ) -> "LabeledBatch":
        decoder_input_ids = be.decoder_input_ids[:, :-1]
        decoder_attention_mask = be.decoder_attention_mask[:, :-1]
        target = be.decoder_input_ids[:, 1:]
        target_ignore_mask = be.decoder_attention_mask[:, 1:]
        target_ignore_mask = torch.where(
            target_ignore_mask == 0, ignore_id, target_ignore_mask
        )
        return cls(
            input_ids=be.input_ids,
            attention_mask=be.attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            target=target,
            target_ignore_mask=target_ignore_mask,
        )


class TransformerDataset(torchd.Dataset):
    def __init__(
        self, data: list[SourceTarget], tokenizer: tokenization.Tokenizer
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.processed_data = self._process_data(data)

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.processed_data[index]

    def _process_data(self, data: list[SourceTarget]) -> list[dict[str, list[int]]]:
        tokenized_data = self.tokenizer(
            encoder_inputs=[dp.source for dp in data],
            decoder_inputs=[dp.target for dp in data],
        )
        return [
            {
                "input_ids": ids,
                "decoder_input_ids": dec_ids,
            }
            for ids, dec_ids in zip(
                tokenized_data.input_ids, tokenized_data.decoder_input_ids, strict=True
            )
        ]


class TransformerDataCollator:
    def __init__(
        self,
        tokenizer: tokenization.Tokenizer,
        padding: tokenization.PaddingOptions,
        label_pad_token_id: int = IGNORE_ID,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: list[dict[str, list[int]]]) -> LabeledBatch:
        batch_encoding = self.tokenizer.pad(
            batch,
            padding=self.padding,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return LabeledBatch.from_batch_encoding(batch_encoding, self.label_pad_token_id)
