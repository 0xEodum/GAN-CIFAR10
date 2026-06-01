"""Entry point for GAN training on CIFAR-10."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gan_cifar.config import GAN
from gan_cifar.data.cifar_dataset import build_loader
from gan_cifar.training.trainer import GANTrainer
from gan_cifar.utils.seed import seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train class-conditional SN-GAN on CIFAR-10")
    parser.add_argument("--data-dir", type=Path, default=GAN.data_dir)
    parser.add_argument("--epochs", type=int, default=GAN.epochs)
    parser.add_argument("--batch-size", type=int, default=GAN.batch_size)
    parser.add_argument("--num-workers", type=int, default=GAN.num_workers)
    parser.add_argument("--z-dim", type=int, default=GAN.z_dim)
    parser.add_argument("--g-base", type=int, default=GAN.g_base)
    parser.add_argument("--d-base", type=int, default=GAN.d_base)
    parser.add_argument("--lr-g", type=float, default=GAN.lr_g)
    parser.add_argument("--lr-d", type=float, default=GAN.lr_d)
    parser.add_argument("--d-steps", type=int, default=GAN.d_steps)
    parser.add_argument("--adv-loss", default=GAN.adv_loss, choices=["ralsgan", "hinge"])
    parser.add_argument("--wrong-label-weight", type=float, default=GAN.wrong_label_weight)
    parser.add_argument("--aux-loss-weight", type=float, default=GAN.aux_loss_weight)
    parser.add_argument("--g-aux-loss-weight", type=float, default=GAN.g_aux_loss_weight)
    parser.add_argument("--r1-gamma", type=float, default=GAN.r1_gamma)
    parser.add_argument("--max-steps", type=int, default=GAN.max_steps)
    parser.add_argument("--ckpt-dir", type=Path, default=GAN.ckpt_dir_override)
    parser.add_argument("--sample-dir", type=Path, default=GAN.sample_dir_override)
    parser.add_argument("--amp-dtype", default=GAN.amp_dtype,
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--log-every", type=int, default=GAN.log_every)
    parser.add_argument("--diagnostics-every", type=int, default=GAN.diagnostics_every)
    parser.add_argument("--sample-every", type=int, default=GAN.sample_every)
    parser.add_argument("--keep-ckpt-every", type=int, default=GAN.keep_ckpt_every)
    parser.add_argument("--ckpt-every", type=int, default=GAN.ckpt_every)
    parser.add_argument("--seed", type=int, default=GAN.seed)
    args = parser.parse_args()

    cfg = replace(
        GAN,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        z_dim=args.z_dim,
        g_base=args.g_base,
        d_base=args.d_base,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        d_steps=args.d_steps,
        adv_loss=args.adv_loss,
        wrong_label_weight=args.wrong_label_weight,
        aux_loss_weight=args.aux_loss_weight,
        g_aux_loss_weight=args.g_aux_loss_weight,
        r1_gamma=args.r1_gamma,
        max_steps=args.max_steps,
        ckpt_dir_override=args.ckpt_dir,
        sample_dir_override=args.sample_dir,
        amp_dtype=args.amp_dtype,
        log_every=args.log_every,
        diagnostics_every=args.diagnostics_every,
        sample_every=args.sample_every,
        keep_ckpt_every=args.keep_ckpt_every,
        ckpt_every=args.ckpt_every,
        seed=args.seed,
    )

    seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True

    loader = build_loader(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train=True,
        augment=True,
        return_labels=True,
    )

    print(f"Dataset: CIFAR-10  batches/epoch: {len(loader)}", flush=True)
    print(f"z_dim={cfg.z_dim}  g_base={cfg.g_base}  d_base={cfg.d_base}", flush=True)
    print(
        f"lr_g={cfg.lr_g}  lr_d={cfg.lr_d}  d_steps={cfg.d_steps}  "
        f"adv_loss={cfg.adv_loss}  amp={cfg.amp_dtype}",
        flush=True,
    )
    print(
        f"wrong_label_weight={cfg.wrong_label_weight}  aux_loss_weight={cfg.aux_loss_weight}  "
        f"g_aux_loss_weight={cfg.g_aux_loss_weight}",
        flush=True,
    )
    print(f"conditional classes={cfg.num_classes}  max_steps={cfg.max_steps}", flush=True)
    print(f"keep_ckpt_every={cfg.keep_ckpt_every}  ckpt_every_epochs={cfg.ckpt_every}", flush=True)
    print(f"ckpt: {cfg.ckpt_dir}  samples: {cfg.sample_dir}", flush=True)

    trainer = GANTrainer(cfg, device="cuda")
    trainer.train(loader)


if __name__ == "__main__":
    main()
