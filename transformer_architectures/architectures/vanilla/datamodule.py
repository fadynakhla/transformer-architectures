from typing import Optional
import multiprocessing

import loguru
import numpy as np
import torch
from torch.utils import data as torchd

from transformer_architectures import samplers
from transformer_architectures.architectures.vanilla import data, tokenization
from transformer_architectures.datasets import wmt_en_fr
from transformer_architectures.training import data_utils, distributed

logger = loguru.logger


class TransformerDataModule(distributed.DataModule):
    def __init__(
        self,
        dataset_config: data.DatasetConfig,
        tokenizer: tokenization.Tokenizer,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        seed: int = 42,
        token_budget: Optional[int] = None,
        sort_window: Optional[int] = None,
    ) -> None:
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.token_budget = token_budget
        self.sort_window = sort_window
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.train_batch_sampler: Optional[torchd.Sampler[list[int]]] = None
        self.data_collator = data.TransformerDataCollator(
            tokenizer=tokenizer, padding="longest", pad_to_multiple_of=8
        )

    def setup(self, ctx: distributed.DistributedContext) -> None:
        if ctx.is_head:
            logger.info("Loading dataset on rank 0 node.")
            full_data = load_data(
                self.dataset_config.num_samples, self.dataset_config.data_path
            )
            logger.info("Spliting dataset into train / val / test.")
            train, val, test = data_utils.train_val_test_split(
                full_data,
                self.dataset_config.val_split,
                self.dataset_config.test_split,
                self.seed,
            )
            logger.info(f"Chunking train data into {ctx.world_size} pieces.")
            train_chunks = data_utils.split_into_chunks(train, ctx.world_size)

        else:
            train_chunks, val, test = None, None, None

        local_train = distributed.scatter_objects(train_chunks, ctx)
        val = distributed.broadcast_objects(val, ctx)
        test = distributed.broadcast_objects(test, ctx)
        if ctx.is_head:
            logger.info("All data scattered/broadcasted.")
            logger.info("Creating datasets on all workers.")

        self._train_dataset = data.TransformerDataset(local_train, self.tokenizer)
        self._val_dataset = data.TransformerDataset(val, self.tokenizer)
        self._test_dataset = data.TransformerDataset(test, self.tokenizer)
        if ctx.is_head:
            logger.info("Datasets initialized.")

        if self.token_budget is not None:
            logger.info(f"Creating token budget sampler with budget: {self.token_budget}, sort window: {self.sort_window}")
            self.train_batch_sampler = samplers.TokenBudgetBatchSampler(
                dataset=self.train_dataset,
                token_budget=self.token_budget,
                sort_window=self.sort_window,
                generator=self.generator,
            )

    def train_dataloader(self) -> torchd.DataLoader[dict[str, np.ndarray]]:
        if self.train_batch_sampler is not None:
            logger.info("Creating train dataloader with token budget sampler")
            return torchd.DataLoader(
                dataset=self.train_dataset,
                batch_sampler=self.train_batch_sampler,
                collate_fn=self.data_collator,
                num_workers=multiprocessing.cpu_count(),
                pin_memory=False,
            )
        logger.info("Creating train dataloader with fixed batch sampling")
        return torchd.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            generator=self.generator,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=False,
        )

    def val_dataloader(self) -> torchd.DataLoader[dict[str, np.ndarray]]:
        return torchd.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

    def test_dataloader(self) -> torchd.DataLoader[dict[str, np.ndarray]]:
        return torchd.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )


def load_data(
    num_samples: int, data_path: str = "/data/datasets/wmt/en-fr"
) -> list[data.SourceTarget]:
    return [
        data.SourceTarget(source=en, target=fr)
        for en, fr in wmt_en_fr.load_parallel_sentences(
            data_path, num_samples=num_samples
        )
    ]
