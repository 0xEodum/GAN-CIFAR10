"""Image I/O helpers: tensor <-> PIL, grid saving."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert a CHW tensor in [-1, 1] to HWC uint8 numpy in [0, 255]."""
    if img.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")
    arr = (img.detach().clamp(-1.0, 1.0).add(1.0).mul(127.5)
           .to(torch.uint8).cpu().numpy())
    return np.transpose(arr, (1, 2, 0))


def save_image_tensor(img: torch.Tensor, path: Path) -> None:
    """Save a CHW [-1,1] tensor as PNG."""
    arr = tensor_to_uint8(img)
    if arr.shape[-1] == 1:
        Image.fromarray(arr.squeeze(-1), mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)


def make_grid_uint8(tensors: Iterable[torch.Tensor], ncol: int = 8) -> np.ndarray:
    """Stack CHW tensors (each in [-1,1]) into a single HWC uint8 grid."""
    imgs = [tensor_to_uint8(t) for t in tensors]
    if not imgs:
        raise ValueError("Empty tensor list")
    h, w, c = imgs[0].shape
    if any(im.shape != (h, w, c) for im in imgs):
        raise ValueError("All tensors must share shape")
    nrow = (len(imgs) + ncol - 1) // ncol
    grid = np.zeros((nrow * h, ncol * w, c), dtype=np.uint8)
    for idx, im in enumerate(imgs):
        r, col = divmod(idx, ncol)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = im
    return grid


def save_grid(tensors: Iterable[torch.Tensor], path: Path, ncol: int = 8) -> None:
    """Save a list of CHW [-1,1] tensors as a single grid PNG."""
    grid = make_grid_uint8(tensors, ncol=ncol)
    Image.fromarray(grid, mode="RGB").save(path)
