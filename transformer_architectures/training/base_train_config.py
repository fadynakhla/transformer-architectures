from typing import Literal

import pydantic


class BaseTrainConfig(pydantic.BaseModel):
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    grad_accumulation_steps: int
    epochs: int
    eval_metric: str
    comparator: Literal[">", "<"] = ">"
    log_interval: int = 25
    log_grad_distributions: bool = False
    precision: Literal["fp32", "bf16"] = "bf16"


class MLFlowConfig(pydantic.BaseModel):
    tracking_uri: str
    experiment_name: str
    enable_system_metrics: bool = True
    system_metrics_interval: int = 30


class RayConfig(pydantic.BaseModel):
    address: str = "auto"
    num_workers: int
    use_gpu: bool = True
    backend: str = "nccl"
