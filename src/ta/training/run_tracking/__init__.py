from typing import Annotated

import pydantic

from ta.training.run_tracking.base_logger import (
    BaseRunLogger,
    BaseRunTrackingConfig,
    NoOpConfig,
    NoOpLogger,
    RunMeta,
)
from ta.training.run_tracking.mlflow_logger import (
    MLflowConfig,
    MLflowLogger,
)


class RunTrackingConfig(
    pydantic.RootModel[
        Annotated[
            MLflowConfig | NoOpConfig, pydantic.Field(discriminator="logger_type")
        ]
    ]
):
    """Discriminated union over the backend configs, dispatched on `logger_type`.

    A RootModel rather than a bare Annotated alias so it is a real class and
    can be passed to `config.load_config` as `type[T]`.
    """

    def build(
        self, world_rank: int | None = None, attach_run_id: str | None = None
    ) -> BaseRunLogger:
        return self.root.build(world_rank=world_rank, attach_run_id=attach_run_id)


__all__ = [
    "BaseRunLogger",
    "BaseRunTrackingConfig",
    "MLflowConfig",
    "MLflowLogger",
    "NoOpConfig",
    "NoOpLogger",
    "RunMeta",
    "RunTrackingConfig",
]
