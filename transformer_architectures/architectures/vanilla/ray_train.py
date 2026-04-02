import multiprocessing
import sys

import mlflow
import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer

from transformer_architectures import config
from transformer_architectures.architectures.vanilla.distributed_training import (
    TrainableTransformer,
)
from transformer_architectures.training.base_train_config import MLFlowConfig, RayConfig

CONFIG_PATH = "configs/vanilla_large_distributed.yaml"


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH
    ray_config = config.load_config(
        config_path, section="Distributed", model_class=RayConfig
    )
    mlflow_config = config.load_config(
        config_path, section="MLFlow", model_class=MLFlowConfig
    )
    ray.init()
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.set_experiment(mlflow_config.experiment_name)
    with mlflow.start_run() as run:
        arch = TrainableTransformer.from_yaml_config(config_path, run.info.run_id)

        trainer = TorchTrainer(
            train_loop_per_worker=arch.distributed_train_loop,
            scaling_config=ScalingConfig(
                num_workers=ray_config.num_workers,
                use_gpu=ray_config.use_gpu,
                resources_per_worker={"GPU": 1, "CPU": 16},
            ),
            torch_config=TorchConfig(backend=ray_config.backend),
        )
        result = trainer.fit()
    print(f"Training finished. Result: {result}")


if __name__ == "__main__":
    main()
