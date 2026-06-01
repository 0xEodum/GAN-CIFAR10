"""Tests for training diagnostics helpers."""

import torch

from gan_cifar.training.trainer import _image_stats


def test_image_stats_reports_dynamic_range():
    x = torch.tensor([[[[-1.0, 0.0], [0.5, 1.0]]]])
    stats = _image_stats(x)
    assert stats["std"] > 0.0
    assert stats["min"] == -1.0
    assert stats["max"] == 1.0
