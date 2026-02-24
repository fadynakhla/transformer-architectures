from typing import Callable, Optional

import torch
from torch.utils import data as torchd


class TokenBudgetBatchSampler(torchd.Sampler[list[int]]):
    """Yields variable-size batches each containing ~token_budget real tokens.

    Sequences are grouped by similar length to minimize padding waste.
    sort_window controls the padding efficiency vs per-epoch diversity tradeoff:
      None  -> global sort (minimum padding, batches fixed across epochs)
      N     -> sort within random windows of N sequences, re-shuffled each epoch.
               Small N = more diversity, more padding; large N approaches global sort.

    Args:
        dataset: Dataset returning dicts of token id lists.
        token_budget: Target number of real tokens per batch. Controls batch size
            dynamically — set to batch_size * avg_seq_len to match a fixed-sample baseline.
        length_key: Dict key used to measure sequence length for bucketing.
        sort_window: Window size for windowed sort. None for global sort.
        generator: Optional RNG for reproducible shuffling.
        drop_last: If True, drop the final incomplete batch.
    """

    def __init__(
        self,
        dataset: torchd.Dataset[dict[str, list[int]]],
        token_budget: int,
        length_key: str = "input_ids",
        sort_window: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        drop_last: bool = False,
    ) -> None:
        self.token_budget = token_budget
        self.sort_window = sort_window
        self.generator = generator
        self.drop_last = drop_last
        n = len(dataset)  # type: ignore[arg-type]
        self._lengths = [len(dataset[i][length_key]) for i in range(n)]
        # Pre-build once with global sort for a stable __len__ approximation.
        self._static_batches = self._pack(sorted(range(n), key=lambda i: self._lengths[i]))

    def _pack(self, indices: list[int]) -> list[list[int]]:
        """Pack a length-sorted index list into token-budget batches."""
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for idx in indices:
            length = self._lengths[idx]
            new_max = max(current_max, length)
            if new_max * (len(current) + 1) > self.token_budget and current:
                batches.append(current)
                current, current_max = [idx], length
            else:
                current.append(idx)
                current_max = new_max
        if current and not self.drop_last:
            batches.append(current)
        return batches

    def _build_windowed_batches(self) -> list[list[int]]:
        assert self.sort_window is not None
        n = len(self._lengths)
        perm = torch.randperm(n, generator=self.generator).tolist()
        all_batches: list[list[int]] = []
        for start in range(0, n, self.sort_window):
            window = perm[start : start + self.sort_window]
            window.sort(key=lambda i: self._lengths[i])
            all_batches.extend(self._pack(window))
        batch_perm = torch.randperm(len(all_batches), generator=self.generator).tolist()
        return [all_batches[i] for i in batch_perm]

    def __iter__(self):
        if self.sort_window is None:
            # Global sort — fixed batches, shuffle order each epoch.
            for i in torch.randperm(len(self._static_batches), generator=self.generator).tolist():
                yield self._static_batches[i]
        else:
            yield from self._build_windowed_batches()

    def __len__(self) -> int:
        if self.sort_window is None:
            return len(self._static_batches)
        assert self.generator is not None, "generator required for exact __len__ with sort_window"
        state = self.generator.get_state()
        count = len(self._build_windowed_batches())
        self.generator.set_state(state)
        return count


def token_budget_sampler_factory(
    token_budget: int,
    length_key: str = "input_ids",
    sort_window: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    drop_last: bool = False,
) -> Callable[[torchd.Dataset], TokenBudgetBatchSampler]:
    def factory(dataset: torchd.Dataset) -> TokenBudgetBatchSampler:
        return TokenBudgetBatchSampler(
            dataset=dataset,
            token_budget=token_budget,
            length_key=length_key,
            sort_window=sort_window,
            generator=generator,
            drop_last=drop_last,
        )
    return factory
