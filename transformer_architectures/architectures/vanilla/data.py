from typing import Optional, Protocol, TypeVar, runtime_checkable
import math

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

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

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


class TransformerDataModule:
    """Inspired by data modules in torch lightning."""

    def __init__(
        self,
        data: list[SourceTarget],
        tokenizer: tokenization.Tokenizer,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        test_split: float = 0.2,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.generator = torch.Generator().manual_seed(seed)
        self.data_collator = TransformerDataCollator(
            tokenizer=tokenizer, padding="longest", pad_to_multiple_of=8
        )

    def setup(self) -> None:
        full_dataset = TransformerDataset(self.data, self.tokenizer)
        self.train_dataset, self.val_dataset, self.test_dataset = train_val_test_split(
            full_dataset, self.val_split, self.test_split, self.generator
        )

    def train_dataloader(self) -> torchd.DataLoader:
        return torchd.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            generator=self.generator,
        )

    def val_dataloader(self) -> torchd.DataLoader:
        return torchd.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

    def test_dataloader(self) -> torchd.DataLoader:
        return torchd.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )


@runtime_checkable
class HasLen(Protocol):
    def __len__(self) -> int:
        ...


DatasetType = TypeVar("DatasetType", bound=torchd.Dataset)
DataSplit = tuple[
    torchd.Subset[DatasetType],
    torchd.Subset[DatasetType],
    torchd.Subset[DatasetType],
]


def train_val_test_split(
    dataset: DatasetType,
    val_split: float,
    test_split: float,
    generator: torch.Generator,
) -> DataSplit:
    if not isinstance(dataset, HasLen):
        raise ValueError("Dataset must implement __len__")
    total_size = len(dataset)
    val_size = math.floor(total_size * val_split)
    test_size = math.floor(total_size * test_split)
    train_size = total_size - val_size - test_size
    subs = torchd.random_split(dataset, [train_size, val_size, test_size], generator)
    return subs[0], subs[1], subs[2]
