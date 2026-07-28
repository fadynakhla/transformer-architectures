from typing import Any, Iterator, Literal, Mapping
import contextlib
import socket
from concurrent import futures
from contextlib import AbstractContextManager

import loguru
import mlflow
import numpy as np
import pydantic
from matplotlib import figure

from ta.training.run_tracking import base_logger

logger = loguru.logger


class MissingRunIdError(RuntimeError):
    ...


class MLflowLogger(base_logger.BaseRunLogger):
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        world_rank: int | None = None,
        run_id: str | None = None,
        enable_system_metrics: bool = True,
        system_metrics_interval: int = 30,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.enable_system_metrics = enable_system_metrics
        self.system_metrics_interval = system_metrics_interval
        self.run_id = run_id
        self.world_rank = world_rank
        self._is_head = world_rank is None or world_rank == 0

    @contextlib.contextmanager
    def run(self) -> Iterator[base_logger.RunMeta]:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        if self.enable_system_metrics and self.world_rank is not None:
            node_id = f"{socket.gethostname()}-rank{self.world_rank}"
            mlflow.set_system_metrics_node_id(node_id)
            mlflow.config.enable_system_metrics_logging()  # pyright: ignore[reportPrivateImportUsage]
            mlflow.config.set_system_metrics_sampling_interval(  # pyright: ignore[reportPrivateImportUsage]
                self.system_metrics_interval
            )

        self._client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        self._executor = futures.ThreadPoolExecutor(max_workers=1)

        with mlflow.start_run(run_id=self.run_id) as active:
            if not self.run_id:
                self.run_id = active.info.run_id
            try:
                yield base_logger.RunMeta(
                    run_name=active.info.run_name, run_id=active.info.run_id
                )
            finally:
                self._executor.shutdown()

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        if self._is_head:
            mlflow.log_metrics(metrics=dict(metrics), step=step)

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self._is_head:
            mlflow.log_params(params=dict(params))

    def log_text(self, text: str, artifact_file: str) -> None:
        if self._is_head:
            mlflow.log_text(text=text, artifact_file=artifact_file)

    def log_dict(self, data: dict[str, Any], artifact_file: str) -> None:
        if self._is_head:
            mlflow.log_dict(data, artifact_file=artifact_file)

    def log_image(self, image: np.ndarray, artifact_file: str) -> None:
        if self._is_head:
            mlflow.log_image(image=image, artifact_file=artifact_file)

    def log_histogram(
        self, name: str, tag: str, counts: np.ndarray, bin_edges: np.ndarray, step: int
    ) -> None:
        if not self.run_id:
            raise MissingRunIdError(
                "All logging must be run from within a `.run()` context manager."
            )
        if self._is_head:
            fut = self._executor.submit(
                self._render_and_upload, name, tag, counts, bin_edges, step
            )
            fut.add_done_callback(_warn_on_failure)

    def _render_and_upload(
        self, name: str, tag: str, counts: np.ndarray, bin_edges: np.ndarray, step: int
    ) -> None:
        if not self.run_id:
            raise MissingRunIdError(
                "All logging must be run from within a `.run()` context manager."
            )

        path = name.replace(".", "/")
        fig = figure.Figure()
        ax = fig.subplots()
        ax.stairs(values=counts, edges=bin_edges)
        ax.set_title(name)
        self._client.log_figure(self.run_id, fig, f"{tag}/step_{step}/{path}.png")

    @property
    def _records_anything(self) -> bool:
        return self._is_head


class MLflowConfig(base_logger.BaseRunTrackingConfig):
    logger_type: Literal["mlflow"] = "mlflow"
    tracking_uri: str
    experiment_name: str
    enable_system_metrics: bool = True
    system_metrics_interval: int = 30

    def build(
        self, world_rank: int | None = None, attach_run_id: str | None = None
    ) -> MLflowLogger:
        return MLflowLogger(
            tracking_uri=self.tracking_uri,
            experiment_name=self.experiment_name,
            world_rank=world_rank,
            run_id=attach_run_id,
            enable_system_metrics=self.enable_system_metrics,
            system_metrics_interval=self.system_metrics_interval,
        )


def _warn_on_failure(fut: futures.Future) -> None:
    if not fut.cancelled() and (exc := fut.exception()) is not None:
        logger.warning(f"Histogram upload failed: {exc!r}")
