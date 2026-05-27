"""CIFAR-10 data loading. Returns tensors normalized to [-1, 1]."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class CIFAR10Images(Dataset):
    """Wrapper for CIFAR-10 images, optionally including class labels."""

    def __init__(
        self,
        root: Path,
        train: bool = True,
        augment: bool = True,
        return_labels: bool = False,
    ) -> None:
        transform = _train_transform() if augment else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self._ds = CIFAR10(root=str(root), train=train, download=True, transform=transform)
        self.return_labels = return_labels

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        img, label = self._ds[idx]
        if self.return_labels:
            return img, int(label)
        return img


def build_loader(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    train: bool = True,
    augment: bool = True,
    return_labels: bool = False,
) -> DataLoader:
    ds = CIFAR10Images(data_dir, train=train, augment=augment, return_labels=return_labels)
    kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": train,
        "drop_last": train,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)
