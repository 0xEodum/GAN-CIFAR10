"""Diagnose generator output contrast/color collapse vs real CIFAR-10."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gan_cifar.config import GAN
from gan_cifar.data.cifar_dataset import build_loader
from gan_cifar.models.dcgan import Generator


def stats(name: str, x: torch.Tensor) -> None:
    # x: (N, 3, H, W) in [-1, 1]
    x = x.float()
    per_ch_mean = x.mean(dim=(0, 2, 3))
    per_ch_std = x.std(dim=(0, 2, 3))
    # saturation proxy: std across channels per pixel (0 = gray)
    chroma = x.std(dim=1).mean()
    print(f"== {name} ==")
    print(f"  overall  mean {x.mean():.4f}  std {x.std():.4f}  min {x.min():.4f}  max {x.max():.4f}")
    print(f"  per-ch mean R{per_ch_mean[0]:.3f} G{per_ch_mean[1]:.3f} B{per_ch_mean[2]:.3f}")
    print(f"  per-ch std  R{per_ch_std[0]:.3f} G{per_ch_std[1]:.3f} B{per_ch_std[2]:.3f}")
    print(f"  chroma (mean cross-channel std per pixel) {chroma:.4f}")


def main() -> None:
    dev = "cuda"
    ckpt = torch.load("checkpoints_conditional_v4/latest.pt", map_location=dev, weights_only=False)
    print("checkpoint step:", ckpt.get("step"))

    G = Generator(GAN.z_dim, GAN.num_channels, GAN.g_base, GAN.num_classes).to(dev)
    G.load_state_dict(ckpt["G_ema"])
    G.eval()

    z = torch.randn(640, GAN.z_dim, device=dev)
    labels = torch.arange(640, device=dev) % GAN.num_classes
    with torch.no_grad():
        fake = G(z, labels)
    stats("FAKE (G_ema, eval)", fake)

    # train-mode BN (uses batch stats) to compare
    G.train()
    with torch.no_grad():
        fake_tr = G(z, labels)
    stats("FAKE (G_ema, train-mode BN)", fake_tr)

    loader = build_loader(GAN.data_dir, batch_size=640, num_workers=0,
                          train=True, augment=False, return_labels=True)
    real, _ = next(iter(loader))
    stats("REAL CIFAR-10", real.to(dev))


if __name__ == "__main__":
    main()
