import dataclasses

import ray.train
import torch


@dataclasses.dataclass
class DistributedContext:
    world_rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_head(self) -> bool:
        return self.world_rank == 0

    @classmethod
    def from_ray_context(cls) -> "DistributedContext":
        ray_context = ray.train.get_context()
        device = torch.device(
            f"cuda:{ray_context.get_local_rank()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        return DistributedContext(
            world_rank=ray_context.get_world_rank(),
            world_size=ray_context.get_world_size(),
            local_rank=ray_context.get_local_rank(),
            device=device,
        )
