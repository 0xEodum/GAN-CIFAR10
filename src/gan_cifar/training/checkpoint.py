from __future__ import annotations

from pathlib import Path, PurePath
from typing import Any

import torch


def checkpoint_path_for_step(ckpt_dir: Path, step: int) -> Path:
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    return ckpt_dir / f"step_{step:07d}.pt"


def make_checkpoint_portable(value: Any) -> Any:
    """Return a copy with pathlib objects converted to portable strings."""
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, dict):
        return {
            make_checkpoint_portable(key): make_checkpoint_portable(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [make_checkpoint_portable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(make_checkpoint_portable(item) for item in value)
    if isinstance(value, set):
        return {make_checkpoint_portable(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(make_checkpoint_portable(item) for item in value)
    return value


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(make_checkpoint_portable(payload), tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
