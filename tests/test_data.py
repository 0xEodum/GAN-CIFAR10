"""Tests for CIFAR-10 data loading."""
import torch
import pytest
from pathlib import Path

from gan_cifar.data.cifar_dataset import CIFAR10Images


@pytest.fixture(scope="module")
def dataset():
    return CIFAR10Images(Path("data"), train=True, augment=False)


def test_dataset_length(dataset):
    assert len(dataset) == 50_000


def test_dataset_item_shape(dataset):
    img = dataset[0]
    assert img.shape == (3, 32, 32)


def test_dataset_item_range(dataset):
    img = dataset[0]
    assert img.min() >= -1.0 - 1e-5
    assert img.max() <=  1.0 + 1e-5


def test_dataset_returns_tensor(dataset):
    img = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.float32


def test_dataset_can_return_labels():
    ds = CIFAR10Images(Path("data"), train=True, augment=False, return_labels=True)
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(label, int)
    assert 0 <= label < 10
