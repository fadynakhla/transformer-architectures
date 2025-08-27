import math

import torch
from torch import nn
from torch.nn import functional as F

from transformer_architectures import encoder


class VisionTransformer(nn.Module):
    """Implementation of the vision transformer from ViT paper."""

    def __init__(
        self,
        patch_size: int,
        image_size: int,
        num_stacks: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pre_layernorm = True
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = math.ceil(image_size / patch_size) ** 2

        # This is mathematically equivalent to reshaping the image to
        # (num_patches, patch_size**2) and multiplying by a linear layer
        self.patch_embedding = nn.Conv2d(
            in_channels=3,  # RGB
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )

        self.encoder = encoder.Encoder(
            num_stacks=num_stacks,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            pre_layernorm=self.pre_layernorm,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, "Input image batch should be 4d (b, c, h, w)"
        assert (
            images.shape[2] == images.shape[3] == self.image_size
        ), f"Invalid shape image size! Height and width should be: {self.image_size}"

        batch_size = images.shape[0]

        patch_embeddings: torch.Tensor = self.patch_embedding(images)
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        patch_embeddings = torch.cat([cls_token, patch_embeddings], dim=1)
        patch_embeddings = patch_embeddings + self.pos_embeddings

        return self.encoder(patch_embeddings, attention_mask=None)

    def convert_image_size(self, new_size: int) -> None:
        old_size = self.image_size

        cls_pe = self.pos_embeddings[:, 0:1, :]
        patch_pe = self.pos_embeddings[:, 1:, :]

        patch_pe = patch_pe.reshape((1, old_size, old_size, -1))
        patch_pe = patch_pe.permute(0, 3, 1, 2)

        new_patch_pe = F.interpolate(
            patch_pe, size=(new_size, new_size), mode="bilinear", align_corners=False
        )
        new_patch_pe = new_patch_pe.permute(0, 2, 3, 1)
        new_patch_pe = new_patch_pe.reshape(1, new_size * new_size, -1)
        self.pos_embeddings = nn.Parameter(torch.cat([cls_pe, new_patch_pe], dim=1))
        self.image_size = new_size
        self.num_patches = math.ceil(new_size / self.patch_size) ** 2



class VisionTransformerForImageClassification(VisionTransformer):
    def __init__(
        self,
        patch_size: int,
        image_size: int,
        num_stacks: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        num_classes: int
    ) -> None:
        super().__init__(
            patch_size, image_size, num_stacks, embed_dim, num_heads, ff_dim, dropout
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_embeddings = super().forward(images)
        cls_embeddings = patch_embeddings[:, 0]
        return self.head(cls_embeddings)
