"""Tests for checkpoint naming and I/O helpers."""

from pathlib import Path

import pytest
import torch

from src.training.checkpoint import checkpoint_path_for_step, load_checkpoint, save_checkpoint


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
