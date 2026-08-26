from typing import Any, Literal, Mapping, NamedTuple
import abc
import contextlib
import datetime
import uuid
import zoneinfo

import loguru
import numpy as np
import pydantic
import torch
from torch import nn

logger = loguru.logger


GRAD_NORM = "grad_norm"
GRADS = "grads"
PARAMS = "params"


class RunMeta(pydantic.BaseModel):
    run_name: str
    run_id: str


class BaseRunLogger(abc.ABC):
    @abc.abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        ...

    @abc.abstractmethod
    def log_params(self, params: Mapping[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def log_dict(self, data: dict[str, Any], artifact_file: str) -> None:
        ...

    @abc.abstractmethod
    def log_text(self, text: str, artifact_file: str) -> None:
        ...

    @abc.abstractmethod
    def run(self) -> contextlib.AbstractContextManager[RunMeta]:
        ...

    @abc.abstractmethod
    def log_histogram(
        self, name: str, tag: str, counts: np.ndarray, bin_edges: np.ndarray, step: int
    ) -> None:
        ...

    @torch.no_grad()
    def log_grads(
        self, model: nn.Module, step: int, log_distributions: bool = False
    ) -> None:
        if not self._records_anything:
            return
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g) for g in grads.values()])
        ).item()
        self.log_metrics({GRAD_NORM: norm}, step=step)
        if log_distributions:
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            for tag, tensors in ((GRADS, grads), (PARAMS, params)):
                for name, t in tensors.items():
                    hist = make_hist(t)
                    if hist is None:
                        logger.warning(
                            f"Skipping {tag}/{name} histogram: no finite values."
                        )
                        continue
                    if hist.num_nonfinite:
                        logger.warning(
                            f"{tag}/{name} contains {hist.num_nonfinite} non-finite values."
                        )
                    self.log_histogram(name, tag, hist.counts, hist.bin_edges, step)

    @property
    def _records_anything(self) -> bool:
        return True


class NoOpLogger(BaseRunLogger):
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        return

    def log_params(self, params: Mapping[str, Any]) -> None:
        return

    def log_dict(self, data: dict[str, Any], artifact_file: str) -> None:
        return

    def log_text(self, text: str, artifact_file: str) -> None:
        return

    def log_histogram(
        self, name: str, tag: str, counts: np.ndarray, bin_edges: np.ndarray, step: int
    ) -> None:
        return

    def run(self) -> contextlib.nullcontext[RunMeta]:
        now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
        return contextlib.nullcontext(
            enter_result=RunMeta(
                run_name=f"noop_{now:%Y-%m-%d_%H-%M-%S}",
                run_id=uuid.uuid4().hex,
            )
        )


class BaseRunTrackingConfig(pydantic.BaseModel, abc.ABC):
    @abc.abstractmethod
    def build(
        self, world_rank: int | None = None, attach_run_id: str | None = None
    ) -> BaseRunLogger:
        ...


class NoOpConfig(BaseRunTrackingConfig):
    logger_type: Literal["noop"] = "noop"

    def build(
        self, world_rank: int | None = None, attach_run_id: str | None = None
    ) -> NoOpLogger:
        return NoOpLogger()


class Histogram(NamedTuple):
    counts: np.ndarray
    bin_edges: np.ndarray
    num_nonfinite: int


def make_hist(tensor: torch.Tensor, bins: int = 64) -> Histogram | None:
    """Bin a tensor on its own device; only bin counts cross to the host.

    Non-finite values (nan/inf, e.g. from overflowing grads) are excluded
    from the histogram and reported via `num_nonfinite`. Returns None if
    the tensor has no finite values to bin.
    """
    t = tensor.detach().float().flatten()
    if t.numel() == 0:
        return None
    finite_mask = torch.isfinite(t)
    num_nonfinite = t.numel() - int(finite_mask.sum().item())
    if num_nonfinite:
        t = t[finite_mask]
        if t.numel() == 0:
            return None
    lo = t.min().item()
    hi = t.max().item()
    if lo == hi:
        # histc needs min < max; widen the way np.histogram does.
        lo -= 0.5
        hi += 0.5
    counts = torch.histc(t, bins=bins, min=lo, max=hi).cpu().numpy()
    edges = np.linspace(lo, hi, num=bins + 1)
    return Histogram(counts=counts, bin_edges=edges, num_nonfinite=num_nonfinite)
