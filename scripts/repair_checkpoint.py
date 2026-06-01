"""Rewrite a checkpoint so pathlib objects are portable across operating systems."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gan_cifar.training.checkpoint import make_checkpoint_portable


def _default_output(path: Path) -> Path:
    return path.with_name(f"{path.stem}.portable{path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pathlib.Path objects inside a torch checkpoint to strings."
    )
    parser.add_argument("checkpoint", type=Path, help="Input checkpoint path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to '<name>.portable.pt'.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the input checkpoint after writing a backup.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create '<checkpoint>.bak' when using --in-place.",
    )
    args = parser.parse_args()

    if args.output is not None and args.in_place:
        parser.error("--output cannot be combined with --in-place")
    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    destination = args.checkpoint if args.in_place else (args.output or _default_output(args.checkpoint))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    portable = make_checkpoint_portable(payload)

    if args.in_place and not args.no_backup:
        backup = args.checkpoint.with_suffix(args.checkpoint.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(args.checkpoint, backup)
            print(f"Backup: {backup}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(portable, tmp)
    tmp.replace(destination)
    print(f"Portable checkpoint: {destination}")


if __name__ == "__main__":
    main()
