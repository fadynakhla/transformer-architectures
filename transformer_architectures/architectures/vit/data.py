from typing import Literal, Protocol, TypeVar, runtime_checkable
import collections
import csv
import dataclasses
import math
import multiprocessing
import os
import pathlib

import loguru
import torch
from PIL import Image
from torch.utils import data as torchd
from torchvision.io import read_image  # pyright: ignore[reportMissingTypeStubs]
from torchvision.transforms import (
    v2 as transforms,
)  # pyright: ignore[reportMissingTypeStubs]

from transformer_architectures.preprocessing import image_processing

logger = loguru.logger


_T_co = TypeVar("_T_co", covariant=True)
Label = TypeVar("Label", None, torch.Tensor)
LabelMask = TypeVar("LabelMask", None, torch.Tensor)


@dataclasses.dataclass
class LabeledImage:
    image: torch.Tensor
    label: int


@dataclasses.dataclass
class MultiLabeledImageIndex:
    image_path: str
    pos_label_indices: list[int]
    neg_label_indices: list[int]
    image_id: str | None


@dataclasses.dataclass
class MultiLabeledImage:
    image: torch.Tensor
    labels: torch.Tensor  # multihot shape (n_classes,)
    mask: torch.Tensor
    image_id: str | None


@dataclasses.dataclass
class LabeledBatch:
    images: torch.Tensor
    labels: torch.Tensor
    masks: torch.Tensor

    def to(self, device: torch.device) -> None:
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)
        self.masks = self.masks.to(device)


# def _img_to_tensor_rgb01(path: str) -> torch.Tensor:
#     img = Image.open(path).convert("RGB")
#     arr = np.asarray(img, dtype=np.float32) / 255.0
#     return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class VisionTransformerDataset(torchd.Dataset[LabeledImage]):
    def __init__(
        self,
        data: list[LabeledImage],
        normalize: bool = True,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.processed_data = self._process_data(data)

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, index: int) -> LabeledImage:
        return self.processed_data[index]

    def _process_data(self, data: list[LabeledImage]) -> list[LabeledImage]:
        if not self.normalize:
            return data
        if self.mean is None or self.std is None:
            self.mean, self.std = self._compute_normalization_stats(data)

        processed: list[LabeledImage] = []
        for item in data:
            normalized_image = (item.image - self.mean) / self.std
            processed.append(LabeledImage(image=normalized_image, label=item.label))

        return processed

    def _compute_normalization_stats(
        self, data: list[LabeledImage]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([item.image for item in data])
        mean = images.mean(dim=(0, 2, 3), keepdim=True)
        std = images.std(dim=(0, 2, 3), keepdim=True)

        std = torch.clamp(std, min=1e-7)

        mean = mean.squeeze(0)
        std = std.squeeze(0)
        return mean, std


class OpenImagesDataset(torchd.Dataset[MultiLabeledImage]):
    def __init__(
        self,
        data_dir: str,
        annotations_dir: str,
        data_samples: int,
        image_size: int,
        resizing_strategy: Literal["stretch", "center_crop"] = "stretch",
        normalize: bool = True,
        norm_samples: int | None = None,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.mid_to_name = self._mid_to_name(annotations_dir)
        self.image_id_to_path = self._build_id_to_path(data_dir, data_samples)
        img_to_mids, img_to_neg_mids, used_mids = self._build_label_map(annotations_dir)
        self.class_mids = sorted(used_mids)
        self.mid_to_index = {m: i for i, m in enumerate(self.class_mids)}
        self.num_classes = len(self.class_mids)
        logger.info("Number of positive classes in dataset: {}", self.num_classes)
        logger.info("Building image index")
        self._image_index = self._build_index(img_to_mids, img_to_neg_mids)
        self.transform = self._get_transform(
            self._image_index,
            image_size,
            resizing_strategy,
            normalize,
            norm_samples,
            mean,
            std,
        )

    def __len__(self) -> int:
        return len(self._image_index)

    def __getitem__(self, index: int) -> MultiLabeledImage:
        image_data = self._image_index[index]
        image = read_image(image_data.image_path)
        image = self.transform(image)
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        labels[image_data.pos_label_indices] = 1.0
        mask = torch.zeros(self.num_classes, dtype=torch.float32)
        mask[image_data.pos_label_indices + image_data.neg_label_indices] = 1.0
        return MultiLabeledImage(image, labels, mask, image_data.image_id)

    def _build_id_to_path(self, data_dir: str, data_samples: int) -> dict[str, str]:
        exts = (".jpg", ".jpeg", ".png")
        img_paths = os.listdir(data_dir)
        logger.info("Images in data dir: {}", len(img_paths))
        id_to_path = {
            os.path.splitext(f)[0]: os.path.join(data_dir, f)
            for f in img_paths[:data_samples]
            if f.lower().endswith(exts)
        }
        if not id_to_path:
            raise RuntimeError("no images found")
        return id_to_path

    def _build_label_map(
        self, annotations_dir: str
    ) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str]]:
        labels_csv = os.path.join(
            annotations_dir, "train-annotations-human-imagelabels.csv"
        )

        img_to_mids: dict[str, set[str]] = collections.defaultdict(set)
        img_to_neg_mids: dict[str, set[str]] = collections.defaultdict(set)
        used_mids: set[str] = set()
        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["ImageID"]
                if img_id not in self.image_id_to_path:
                    continue
                mid = row["LabelName"]
                if (conf := float(row.get("Confidence", "1"))) == 1.0:
                    img_to_mids[img_id].add(mid)
                elif conf == 0:
                    img_to_neg_mids[img_id].add(mid)
                else:
                    continue
                used_mids.add(mid)

        return img_to_mids, img_to_neg_mids, used_mids

    def _build_index(
        self, img_to_mids: dict[str, set[str]], img_to_neg_mids: dict[str, set[str]]
    ) -> list[MultiLabeledImageIndex]:
        index: list[MultiLabeledImageIndex] = []
        for img_id, mids in img_to_mids.items():
            if not mids:
                continue
            pos_idx = [self.mid_to_index[m] for m in mids]
            neg_idx = [self.mid_to_index[m] for m in img_to_neg_mids[img_id]]
            index.append(
                MultiLabeledImageIndex(
                    self.image_id_to_path[img_id], pos_idx, neg_idx, img_id
                )
            )
        return index

    def _get_transform(
        self,
        data_index: list[MultiLabeledImageIndex],
        image_size: int,
        resizing_strategy: Literal["stretch", "center_crop"],
        normalize: bool,
        norm_samples: int | None,
        mean: torch.Tensor | None,
        std: torch.Tensor | None,
    ):
        transform = image_processing.build_transform(image_size, resizing_strategy)
        if not normalize:
            return transform

        if mean is None or std is None:
            logger.info(
                "Computing image set norm stats on {} images", norm_samples or "All"
            )
            mean, std = self._compute_normalization_stats(
                data_index, transform, norm_samples
            )
        return image_processing.build_transform_with_norm(
            image_size, resizing_strategy, mean, std
        )

    def _compute_normalization_stats(
        self,
        data_index: list[MultiLabeledImageIndex],
        transform: transforms.Compose,
        num_samples: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        samples = data_index
        if num_samples:
            samples = data_index[:num_samples]
        images = torch.stack([transform(read_image(dp.image_path)) for dp in samples])
        mean = images.mean(dim=(0, 2, 3), keepdim=True)
        std = images.std(dim=(0, 2, 3), keepdim=True)

        std = torch.clamp(std, min=1e-7)

        mean = mean.squeeze(0)
        std = std.squeeze(0)
        return mean, std

    @staticmethod
    def _mid_to_name(path: str) -> dict[str, str]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Annotations folder not found: {path}")
        class_csv = os.path.join(path, "class-descriptions.csv")
        if not os.path.exists(class_csv):
            raise FileNotFoundError(f"Missing {class_csv}")

        mid_to_names: dict[str, str] = {}
        with open(class_csv, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    mid_to_names[row[0]] = row[1]
        return mid_to_names


# class ImageDataCollator:
#     def __call__(self, batch: list[LabeledImage]) -> LabeledBatch:
#         images_list: list[torch.Tensor] = []
#         labels_list: list[int] = []
#         for item in batch:
#             images_list.append(item.image)
#             labels_list.append(item.label)

#         images = torch.stack(images_list)
#         labels = torch.LongTensor(labels_list)
#         return LabeledBatch(images=images, labels=labels)


class MultiLabelDataCollator:
    def __call__(self, batch: list[MultiLabeledImage]) -> LabeledBatch:
        images_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor] = []
        for item in batch:
            images_list.append(item.image)
            labels_list.append(item.labels)
            masks_list.append(item.mask)

        images = torch.stack(images_list)
        labels = torch.stack(labels_list)
        masks = torch.stack(masks_list)
        return LabeledBatch(images=images, labels=labels, masks=masks)


class TransformerDataModule:
    """Inspired by data modules in torch lightning."""

    def __init__(
        self,
        data_dir: str,
        annotations_dir: str,
        data_samples: int,
        image_size: int,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        test_split: float = 0.2,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.annotations_dir = annotations_dir
        self.data_samples = data_samples
        self.image_size = image_size
        # self.data = data
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.generator = torch.Generator().manual_seed(seed)
        self.data_collator = MultiLabelDataCollator()

    def setup(self) -> None:
        full_dataset = OpenImagesDataset(
            self.data_dir,
            self.annotations_dir,
            self.data_samples,
            self.image_size,
            norm_samples=10000,
        )
        self.num_classes = full_dataset.num_classes
        self.mid_to_name = full_dataset.mid_to_name
        self.mid_to_index = full_dataset.mid_to_index
        self.train_dataset, self.val_dataset, self.test_dataset = train_val_test_split(
            full_dataset, self.val_split, self.test_split, self.generator
        )

    def train_dataloader(self) -> torchd.DataLoader[MultiLabeledImage]:
        return torchd.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            generator=self.generator,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

    def val_dataloader(self) -> torchd.DataLoader[MultiLabeledImage]:
        return torchd.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

    def test_dataloader(self) -> torchd.DataLoader[MultiLabeledImage]:
        return torchd.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )


@runtime_checkable
class HasLen(Protocol):
    def __len__(self) -> int:
        ...


DataSplit = tuple[torchd.Subset[_T_co], torchd.Subset[_T_co], torchd.Subset[_T_co]]


def train_val_test_split(
    dataset: torchd.Dataset[_T_co],
    val_split: float,
    test_split: float,
    generator: torch.Generator,
) -> DataSplit[_T_co]:
    if not isinstance(dataset, HasLen):
        raise ValueError("Dataset must implement __len__")
    total_size = len(dataset)
    val_size = math.floor(total_size * val_split)
    test_size = math.floor(total_size * test_split)
    train_size = total_size - val_size - test_size
    subs = torchd.random_split(dataset, [train_size, val_size, test_size], generator)
    return subs[0], subs[1], subs[2]


def get_size(path: str | pathlib.Path):
    try:
        # Pillow only reads headers for .size; still slower than `imagesize`
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


if __name__ == "__main__":
    ds = OpenImagesDataset(
        data_dir="/data/datasets/downsampled-open-images-v4/512px/train",
        annotations_dir="/data/datasets/downsampled-open-images-v4/512px/annotations",
        data_samples=10,
        image_size=512,
    )
    print(ds[1])
    # from PIL import Image
    # from collections import Counter
    # from concurrent.futures import ThreadPoolExecutor, as_completed
    # from tqdm import tqdm

    # exts = {".jpg"}
    # p = pathlib.Path("/data/datasets/downsampled-open-images-v4/512px/train")
    # files = [p for p in p.iterdir() if p.suffix.lower() in exts]
    # sizes = Counter[tuple[int, int]]()
    # max_workers = min(16, (os.cpu_count() or 8) * 4)

    # with ThreadPoolExecutor(max_workers=max_workers) as ex:
    #     futures = [ex.submit(get_size, f) for f in files]
    #     for fut in tqdm(as_completed(futures), total=len(futures), unit="img"):
    #         s = fut.result()
    #         if s: sizes[s] += 1

    # print("unique sizes:", len(sizes))
    # for (w, h), n in sizes.most_common(15):
    #     print(f"{w}x{h}: {n}")
