"""Loss functions for the GAN. Pure functions for easy testing."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ralsgan_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Relativistic average LSGAN discriminator loss (Jolicoeur-Martineau, 2018)."""
    real_centered = real_logits - fake_logits.mean()
    fake_centered = fake_logits - real_logits.mean()
    return 0.5 * (
        F.mse_loss(real_centered, torch.ones_like(real_centered))
        + F.mse_loss(fake_centered, -torch.ones_like(fake_centered))
    )


def ralsgan_g_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Relativistic average LSGAN generator loss (symmetric, swapped targets)."""
    real_centered = real_logits - fake_logits.mean()
    fake_centered = fake_logits - real_logits.mean()
    return 0.5 * (
        F.mse_loss(fake_centered, torch.ones_like(fake_centered))
        + F.mse_loss(real_centered, -torch.ones_like(real_centered))
    )


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """SN-GAN discriminator hinge loss."""
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def conditional_hinge_d_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    wrong_label_logits: torch.Tensor | None = None,
    wrong_label_weight: float = 1.0,
) -> torch.Tensor:
    """Hinge D loss with optional real-image/wrong-label negatives."""
    loss = hinge_d_loss(real_logits, fake_logits)
    if wrong_label_logits is not None:
        loss = loss + wrong_label_weight * F.relu(1.0 + wrong_label_logits).mean()
    return loss


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """SN-GAN generator hinge loss."""
    return -fake_logits.mean()
