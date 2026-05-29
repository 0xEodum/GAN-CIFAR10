"""Probe: does D score gray (low-amplitude) fakes higher than amplitude-scaled ones?"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import GAN
from src.models.dcgan import Discriminator, Generator


def main() -> None:
    dev = "cuda"
    ckpt = torch.load("checkpoints_conditional_v4/latest.pt", map_location=dev, weights_only=False)
    G = Generator(GAN.z_dim, GAN.num_channels, GAN.g_base, GAN.num_classes).to(dev)
    G.load_state_dict(ckpt["G_ema"])
    G.eval()
    D = Discriminator(GAN.num_channels, GAN.d_base, GAN.num_classes).to(dev)
    D.load_state_dict(ckpt["D"])
    D.eval()

    z = torch.randn(256, GAN.z_dim, device=dev)
    labels = torch.arange(256, device=dev) % GAN.num_classes
    with torch.no_grad():
        fake = G(z, labels)
        for scale in [1.0, 1.5, 2.0, 3.0, 5.0]:
            scaled = (fake * scale).clamp(-1, 1)
            score = D(scaled, labels).float().mean().item()
            print(f"  scale {scale:>3.1f}  D(scaled fake) {score:+.4f}  std {scaled.std():.3f}")


if __name__ == "__main__":
    main()
