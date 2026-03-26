from typing import Any, Callable, ContextManager, Literal, Optional
import math
from contextlib import nullcontext

import loguru
import mlflow
import pydantic
import torch
import torch.nn as nn
import torch.optim as optim


from transformer_architectures import config
from transformer_architectures.architectures import vanilla
from transformer_architectures.datasets import wmt_en_fr
from transformer_architectures.training import checkpointing, grad_logging


logger = loguru.logger


mlflow.set_tracking_uri("http://10.9.9.249:5000")
mlflow.set_experiment("Vanilla Transformer Large - ENFR Distributed")
mlflow.config.enable_system_metrics_logging()  # pyright: ignore[reportPrivateImportUsage]
mlflow.config.set_system_metrics_sampling_interval(  # pyright: ignore[reportPrivateImportUsage]
    30
)


CONFIG_PATH = "configs/vanilla_large.yaml"
