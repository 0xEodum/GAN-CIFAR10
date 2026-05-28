from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def checkpoint_path_for_step(ckpt_dir: Path, step: int) -> Path:
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    return ckpt_dir / f"step_{step:07d}.pt"


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
