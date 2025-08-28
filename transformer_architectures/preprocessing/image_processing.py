from torchvision.transforms import (  # pyright: ignore[reportMissingTypeStubs]
    v2 as transforms,
    InterpolationMode,
)
import torch

def build_transform(image_size: int, mode: str) -> transforms.Compose:
    if mode == "center_crop":
        resize = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.CenterCrop(image_size),
            ]
        )
    elif mode == "stretch":
        resize = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    else:
        raise ValueError(mode)

    return transforms.Compose(
        [
            resize,
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )


def build_transform_with_norm(image_size: int, mode: str, mean: torch.Tensor, std: torch.Tensor):
    if mode == "center_crop":
        resize = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.CenterCrop(image_size),
            ]
        )
    elif mode == "stretch":
        resize = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    else:
        raise ValueError(mode)

    return transforms.Compose(
        [
            resize,
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean.tolist(), std.tolist()  # pyright: ignore[reportUnknownArgumentType]
            ),
        ]
    )
