"""Central configuration. Single source of truth for paths and hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GANConfig:
    # Data
    data_dir: Path = PROJECT_ROOT / "data"
    image_size: int = 32
    num_channels: int = 3
    num_classes: int = 10
    # Model
    z_dim: int = 128
    g_base: int = 64
    d_base: int = 64
    # Optimizer for class-conditional SN-GAN with hinge loss
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    d_steps: int = 2
    beta1: float = 0.0
    beta2: float = 0.9
    # Conditional-D safeguards: real/wrong-label negatives and image-only class head
    wrong_label_weight: float = 1.0
    aux_loss_weight: float = 0.5
    g_aux_loss_weight: float = 0.0
    # Optional lazy R1 gradient penalty. SN-GAN normally keeps this disabled.
    r1_gamma: float = 0.0
    r1_every: int = 16
    # Exponential moving average of G weights for sampling
    ema_decay: float = 0.999
    ema_warmup_steps: int = 500
    # Training
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 200
    max_steps: Optional[int] = None
    ckpt_dir_override: Optional[Path] = None
    sample_dir_override: Optional[Path] = None
    # Logging / checkpointing
    log_every: int = 100
    diagnostics_every: int = 500
    sample_every: int = 500
    ckpt_every: int = 5
    # Mixed precision
    amp_dtype: str = "bf16"
    seed: int = 42

    @property
    def ckpt_dir(self) -> Path:
        return self.ckpt_dir_override or PROJECT_ROOT / "checkpoints"

    @property
    def sample_dir(self) -> Path:
        return self.sample_dir_override or PROJECT_ROOT / "samples"


GAN = GANConfig()
