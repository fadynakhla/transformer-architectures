from typing import Literal

import pydantic


class BaseTrainConfig(pydantic.BaseModel):
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    grad_accumulation_steps: int
    epochs: int
    eval_metric: str
    maximize_eval: bool = True
    comparator: Literal[">", "<"] = ">"
    log_interval: int = 25
    log_grad_distributions: bool = False
    precision: Literal["fp32", "bf16"] = "bf16"
