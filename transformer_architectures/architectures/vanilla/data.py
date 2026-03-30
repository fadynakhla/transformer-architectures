from typing import Literal, Optional, TypeVar
import dataclasses
import multiprocessing

import numpy as np
import pydantic
import torch
from torch.utils import data as torchd

from transformer_architectures import samplers
from transformer_architectures.architectures.vanilla import tokenization
from transformer_architectures.training import data_utils

IGNORE_ID = -100

Label = TypeVar("Label", None, torch.Tensor)
LabelMask = TypeVar("LabelMask", None, torch.Tensor)


class SourceTarget(pydantic.BaseModel):
    source: str
    target: str


class DatasetConfig(pydantic.BaseModel):
    data_path: str
    num_samples: int
    val_split: float
    test_split: float


@dataclasses.dataclass
class LabeledBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    target: torch.Tensor

    # model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def to(self, device: torch.device) -> None:
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.decoder_input_ids = self.decoder_input_ids.to(device)
        self.decoder_attention_mask = self.decoder_attention_mask.to(device)
        self.target = self.target.to(device)

    @classmethod
    def from_batch_encoding(
        cls, be: tokenization.TensorBatchEncoding, ignore_id: int = IGNORE_ID
    ) -> "LabeledBatch":
        decoder_input_ids = be.decoder_input_ids[:, :-1]
        decoder_attention_mask = be.decoder_attention_mask[:, :-1]
        target = be.decoder_input_ids[:, 1:]
        target = torch.where(be.decoder_attention_mask[:, 1:] == 0, ignore_id, target)

        return cls(
            input_ids=be.input_ids,
            attention_mask=be.attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            target=target,
        )


class TransformerDataset(torchd.Dataset[dict[str, np.ndarray]]):
    def __init__(
        self, data: list[SourceTarget], tokenizer: tokenization.Tokenizer
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self._setup_arrays(data)

    def __len__(self) -> int:
        return len(self._input_ids_offsets) - 1

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        enc_s, enc_e = (
            self._input_ids_offsets[index],
            self._input_ids_offsets[index + 1],
        )
        dec_s, dec_e = (
            self._decoder_ids_offsets[index],
            self._decoder_ids_offsets[index + 1],
        )
        return {
            "input_ids": self._input_ids_flat[enc_s:enc_e],
            "decoder_input_ids": self._decoder_ids_flat[dec_s:dec_e],
        }

    def _setup_arrays(self, data: list[SourceTarget]) -> None:
        tokenized = self.tokenizer(
            encoder_inputs=[dp.source for dp in data],
            decoder_inputs=[dp.target for dp in data],
        )
        enc_flat: list[int] = []
        dec_flat: list[int] = []
        enc_offsets: list[int] = [0]
        dec_offsets: list[int] = [0]
        for enc_ids, dec_ids in zip(
            tokenized.input_ids, tokenized.decoder_input_ids, strict=True
        ):
            enc_flat.extend(enc_ids)
            dec_flat.extend(dec_ids)
            enc_offsets.append(len(enc_flat))
            dec_offsets.append(len(dec_flat))
        self._input_ids_flat = np.array(enc_flat, dtype=np.int32)
        self._decoder_ids_flat = np.array(dec_flat, dtype=np.int32)
        self._input_ids_offsets = np.array(enc_offsets, dtype=np.int64)
        self._decoder_ids_offsets = np.array(dec_offsets, dtype=np.int64)


class TransformerDataCollator:
    def __init__(
        self,
        tokenizer: tokenization.Tokenizer,
        padding: tokenization.PaddingOptions,
        label_pad_token_id: int = IGNORE_ID,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.padding: tokenization.PaddingOptions = padding
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: list[dict[str, np.ndarray]]) -> LabeledBatch:
        list_batch: list[dict[str, list[int]]] = [
            {
                "input_ids": sample["input_ids"].tolist(),
                "decoder_input_ids": sample["decoder_input_ids"].tolist(),
            }
            for sample in batch
        ]
        batch_encoding = self.tokenizer.pad(
            list_batch,
            padding=self.padding,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        return LabeledBatch.from_batch_encoding(batch_encoding, self.label_pad_token_id)
