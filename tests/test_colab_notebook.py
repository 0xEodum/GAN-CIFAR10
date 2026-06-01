"""Smoke checks for the companion Colab notebook structure."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[2] / "colabs" / "p6_GAN_CIFAR10_Colab.ipynb"


def _cell_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources: list[str] = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        sources.append("".join(source) if isinstance(source, list) else source)
    return sources


def _cell_index_containing(cells: list[str], text: str) -> int:
    return next(index for index, source in enumerate(cells) if text in source)


def test_amp_context_is_available_without_running_training_cell():
    cells = _cell_sources()
    amp_cell = _cell_index_containing(cells, "def amp_ctx():")
    train_cell = _cell_index_containing(cells, "#@title 8. Train GAN")
    inference_cell = _cell_index_containing(cells, "#@title 11. Class-conditional inference")

    assert amp_cell < train_cell
    assert amp_cell < inference_cell
    assert "def amp_ctx():" not in cells[train_cell]
