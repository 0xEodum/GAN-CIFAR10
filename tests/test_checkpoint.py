"""Tests for checkpoint naming and I/O helpers."""

from pathlib import Path

import pytest
import torch

from gan_cifar.training.checkpoint import (
    checkpoint_path_for_step,
    load_checkpoint,
    make_checkpoint_portable,
    save_checkpoint,
)


def test_checkpoint_path_for_step_is_zero_padded(tmp_path: Path):
    assert checkpoint_path_for_step(tmp_path, 5000) == tmp_path / "step_0005000.pt"


def test_checkpoint_path_for_step_rejects_negative_step(tmp_path: Path):
    with pytest.raises(ValueError):
        checkpoint_path_for_step(tmp_path, -1)


def test_save_checkpoint_roundtrip(tmp_path: Path):
    path = tmp_path / "nested" / "latest.pt"
    save_checkpoint(path, {"step": 7, "tensor": torch.tensor([1, 2, 3])})
    payload = load_checkpoint(path)
    assert payload["step"] == 7
    assert torch.equal(payload["tensor"], torch.tensor([1, 2, 3]))


def test_make_checkpoint_portable_converts_paths_without_touching_tensors():
    tensor = torch.tensor([1, 2, 3])
    payload = {
        "path": Path("checkpoints") / "latest.pt",
        "nested": {"paths": [Path("samples") / "grid.png"]},
        "tensor": tensor,
    }

    portable = make_checkpoint_portable(payload)

    assert portable == {
        "path": "checkpoints/latest.pt",
        "nested": {"paths": ["samples/grid.png"]},
        "tensor": tensor,
    }
    assert portable["tensor"] is tensor
