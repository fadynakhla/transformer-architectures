from typing import Literal, Protocol, TypeVar, runtime_checkable
import collections
import csv
import dataclasses
import math
import multiprocessing
import os
import pathlib

import numpy as np
import torch
from PIL import Image
from torch.utils import data as torchd


IGNORE_ID = -100

_T_co = TypeVar("_T_co", covariant=True)
Label = TypeVar("Label", None, torch.Tensor)
LabelMask = TypeVar("LabelMask", None, torch.Tensor)



@dataclasses.dataclass
class LabeledImage:
    image: torch.Tensor
    label: int

@dataclasses.dataclass
class MultiLabeledImage:
    image: torch.Tensor
    labels: torch.Tensor # multihot shape (n_classes,)
    image_id: str | None


@dataclasses.dataclass
class LabeledBatch:
    images: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> None:
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)


def _img_to_tensor_rgb01(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


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


    def _compute_normalization_stats(self, data: list[LabeledImage]) -> tuple[torch.Tensor, torch.Tensor]:
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
        data_samples: int,
        image_size: tuple[int, int],
        resizing_strategy: Literal["skip"] = "skip",
        normalize: bool = True,
        normalization_samples: int | None = None,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.resizing_strategy = resizing_strategy
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.mid_to_name = self._mid_to_name(data_dir)
        data, self.mid_to_index = self._load_data(data_dir, data_samples)
        self.processed_data = self._process_data(data, normalization_samples)

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, index: int) -> MultiLabeledImage:
        return self.processed_data[index]

    def _load_data(self, data_dir: str, data_samples: int) -> tuple[list[MultiLabeledImage], dict[str, int]]:
        img_dir = os.path.join(data_dir, "train")
        ann_dir = os.path.join(data_dir, "annotations")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Images folder not found: {img_dir}")

        labels_csv  = os.path.join(ann_dir, "train-annotations-human-imagelabels.csv")
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"Missing {labels_csv}")

        exts = (".jpg", )
        id_to_path: dict[str, str] = {}
        samples_gathered = 0
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(exts):
                continue

            img_size = get_size(os.path.join(img_dir, fname))
            if img_size != self.image_size:
                continue

            img_id = os.path.splitext(fname)[0]
            id_to_path[img_id] = os.path.join(img_dir, fname)
            samples_gathered += 1
            if samples_gathered == data_samples:
                break
        present_ids = set(id_to_path.keys())
        if not present_ids:
            raise RuntimeError(f"No images found in {img_dir}")

        img_to_mids: dict[str, set[str]] = collections.defaultdict(set)
        used_mids: set[str] = set()

        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["ImageID"]
                if img_id not in present_ids:
                    continue
                try:
                    conf = float(row.get("Confidence", "1"))
                except ValueError:
                    conf = 1.0
                if conf != 1.0:
                    continue
                mid = row["LabelName"]
                img_to_mids[img_id].add(mid)
                used_mids.add(mid)
        class_mids = sorted(used_mids)
        mid_to_index = {m: i for i, m in enumerate(class_mids)}
        img_to_labels = {image: [mid_to_index[m] for m in mids] for image, mids in img_to_mids.items()}
        num_classes = len(mid_to_index)

        data: list[MultiLabeledImage] = []
        for img_id, labels in img_to_labels.items():
            y = torch.zeros(num_classes, dtype=torch.float32)
            y[labels] = 1.0
            x = _img_to_tensor_rgb01(id_to_path[img_id])
            data.append(MultiLabeledImage(image=x, labels=y, image_id=img_id))

        return data, mid_to_index



    def _process_data(self, data: list[MultiLabeledImage], normalization_samples: int | None) -> list[MultiLabeledImage]:
        if not self.normalize:
            return data
        if self.mean is None or self.std is None:
            self.mean, self.std = self._compute_normalization_stats(data, normalization_samples)

        processed: list[MultiLabeledImage] = []
        for item in data:
            normalized_image = (item.image - self.mean) / self.std
            processed.append(MultiLabeledImage(image=normalized_image, labels=item.labels, image_id=item.image_id))

        return processed


    def _compute_normalization_stats(self, data: list[MultiLabeledImage], samples: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        if samples:
            images = torch.stack([item.image for item in data[:samples]])
        else:
            images = torch.stack([item.image for item in data])
        mean = images.mean(dim=(0, 2, 3), keepdim=True)
        std = images.std(dim=(0, 2, 3), keepdim=True)

        std = torch.clamp(std, min=1e-7)

        mean = mean.squeeze(0)
        std = std.squeeze(0)
        return mean, std

    @staticmethod
    def _mid_to_name(path: str) -> dict[str, str]:
        ann_dir = os.path.join(path, "annotations")
        if not os.path.isdir(ann_dir):
            raise FileNotFoundError(f"Annotations folder not found: {ann_dir}")
        class_csv   = os.path.join(ann_dir, "class-descriptions.csv")
        if not os.path.exists(class_csv):
            raise FileNotFoundError(f"Missing {class_csv}")

        mid_to_names: dict[str, str] = {}
        with open(class_csv, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    mid_to_names[row[0]] = row[1]
        return mid_to_names


class ImageDataCollator:

    def __call__(self, batch: list[LabeledImage]) -> LabeledBatch:
        images_list: list[torch.Tensor] = []
        labels_list: list[int] = []
        for item in batch:
            images_list.append(item.image)
            labels_list.append(item.label)

        images = torch.stack(images_list)
        labels = torch.LongTensor(labels_list)
        return LabeledBatch(images=images, labels=labels)


class MultiLabelDataCollator:
    def __call__(self, batch: list[MultiLabeledImage]) -> LabeledBatch:
        images_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        for item in batch:
            images_list.append(item.image)
            labels_list.append(item.labels)

        images = torch.stack(images_list)
        labels = torch.stack(labels_list)
        return LabeledBatch(images=images, labels=labels)


class TransformerDataModule:
    """Inspired by data modules in torch lightning."""

    def __init__(
        self,
        data_dir: str,
        data_samples: int,
        image_size: tuple[int, int],
        # data: list[MultiLabeledImage],
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        test_split: float = 0.2,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
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
        full_dataset = OpenImagesDataset(self.data_dir, self.data_samples, self.image_size)
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


if __name__=="__main__":
    ds = OpenImagesDataset(data_dir="/data/datasets/downsampled-open-images-v4/512px", data_samples=10, image_size=(512, 512))
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
