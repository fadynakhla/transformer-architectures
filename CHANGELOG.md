# Changelog

## [Unreleased] — fix/distributed-sampler-rank-desync

### Fixed

- **DDP rank desync in TokenBudgetBatchSampler** (`bucket_sampler.py`, `datamodule.py`, `trainable_architecture.py`)

  When training across multiple ranks, `scatter_into_chunks` produces unequal dataset sizes per rank. Each rank's `TokenBudgetBatchSampler` uses the same initial seed but operates on a different number of sequences, causing the generator state to diverge after each epoch. Over long training runs, rank batch counts differ, leading to DDP rank desync and AllReduce hangs.

  Three changes:
  - `TokenBudgetBatchSampler.set_epoch(epoch)`: reseeds the generator to `base_seed + epoch` before each epoch, ensuring identical generator state across all ranks at epoch start.
  - `drop_last=True` on the distributed sampler: drops the final incomplete batch, preventing unequal batch counts from unequal chunk sizes.
  - `trainable_architecture.py`: calls `set_epoch(epoch)` on the sampler before each epoch's training loop.
