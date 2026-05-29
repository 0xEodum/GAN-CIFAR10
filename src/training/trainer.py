"""Class-conditional SN-GAN trainer: hinge loss, projection D, EMA G, AMP."""
from __future__ import annotations

import copy
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import GAN, GANConfig
from ..models.dcgan import Discriminator, Generator
from ..utils.image_io import save_grid
from .checkpoint import checkpoint_path_for_step, save_checkpoint
from .losses import (
    class_consistency_loss,
    conditional_hinge_d_loss,
    conditional_ralsgan_d_loss,
    hinge_g_loss,
    ralsgan_g_loss,
)


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


@torch.no_grad()
def _ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for p_e, p in zip(ema.parameters(), model.parameters()):
        p_e.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for b_e, b in zip(ema.buffers(), model.buffers()):
        b_e.copy_(b)


def _r1_penalty(d_out: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """0.5 * ||∇_x D(x)||^2 averaged over the batch (in fp32 for stability)."""
    grad = torch.autograd.grad(
        outputs=d_out.sum(), inputs=real, create_graph=True, only_inputs=True,
    )[0]
    return 0.5 * grad.pow(2).flatten(1).sum(1).mean()


def _wrong_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    offsets = torch.randint(1, num_classes, labels.shape, device=labels.device)
    return (labels + offsets) % num_classes


def _image_stats(images: torch.Tensor) -> dict[str, float]:
    images = images.detach().float()
    return {
        "std": float(images.std().item()),
        "min": float(images.min().item()),
        "max": float(images.max().item()),
    }


class GANTrainer:
    def __init__(self, cfg: GANConfig = GAN, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.G = Generator(cfg.z_dim, cfg.num_channels, cfg.g_base, cfg.num_classes)
        self.D = Discriminator(cfg.num_channels, cfg.d_base, cfg.num_classes)
        self.G.to(self.device, memory_format=torch.channels_last)
        self.D.to(self.device, memory_format=torch.channels_last)

        # EMA copy of G — used for sampling only, not trained directly
        self.G_ema = copy.deepcopy(self.G).eval()
        for p in self.G_ema.parameters():
            p.requires_grad_(False)

        self.opt_g = torch.optim.Adam(
            self.G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2)
        )
        self.opt_d = torch.optim.Adam(
            self.D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2)
        )
        self.amp_dtype = _amp_dtype(cfg.amp_dtype)
        self.global_step = 0
        # Fixed noise and labels for consistent visual progress tracking
        self.fixed_z = torch.randn(64, cfg.z_dim, device=self.device)
        self.fixed_labels = (torch.arange(64, device=self.device) % cfg.num_classes).long()
        self.class_grid_z = torch.randn(cfg.num_classes * 8, cfg.z_dim, device=self.device)
        self.class_grid_labels = torch.arange(
            cfg.num_classes, device=self.device
        ).repeat_interleave(8).long()
        self.label_sweep_z = torch.randn(8, cfg.z_dim, device=self.device).repeat_interleave(
            cfg.num_classes, dim=0
        )
        self.label_sweep_labels = torch.arange(cfg.num_classes, device=self.device).repeat(8).long()

    def _autocast(self):
        if self.amp_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def train(self, loader: DataLoader) -> None:
        cfg = self.cfg
        if cfg.d_steps < 1:
            raise ValueError(f"d_steps must be >= 1, got {cfg.d_steps}")
        cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
        cfg.sample_dir.mkdir(parents=True, exist_ok=True)

        # Accumulate losses as GPU tensors to avoid CPU sync in the hot loop
        loss_g_acc = torch.zeros((), device=self.device)
        loss_d_acc = torch.zeros((), device=self.device)
        loss_aux_acc = torch.zeros((), device=self.device)
        loss_g_aux_acc = torch.zeros((), device=self.device)
        r1_acc = torch.zeros((), device=self.device)
        count = 0
        d_count = 0
        r1_count = 0

        for epoch in range(cfg.epochs):
            self.G.train()
            self.D.train()
            t0 = time.time()

            should_stop = False
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    real, labels = batch
                    labels = labels.to(self.device, non_blocking=True).long()
                else:
                    real = batch
                    labels = torch.randint(0, cfg.num_classes, (real.size(0),), device=self.device)
                real = real.to(self.device, non_blocking=True,
                               memory_format=torch.channels_last)
                bs = real.size(0)

                # ---- Discriminator step ----
                for _ in range(cfg.d_steps):
                    z = torch.randn(bs, cfg.z_dim, device=self.device)
                    fake_labels = torch.randint(0, cfg.num_classes, (bs,), device=self.device)
                    wrong_labels = _wrong_labels(labels, cfg.num_classes)
                    with self._autocast():
                        fake = self.G(z, fake_labels)
                        real_features = self.D.encode(real)
                        d_real = self.D.score_from_features(real_features, labels)
                        d_wrong = self.D.score_from_features(real_features, wrong_labels)
                        class_logits = self.D.classify_features(real_features)
                        d_fake = self.D(fake.detach(), fake_labels)
                        loss_aux = F.cross_entropy(class_logits.float(), labels)
                        d_loss_fn = (
                            conditional_ralsgan_d_loss
                            if cfg.adv_loss == "ralsgan"
                            else conditional_hinge_d_loss
                        )
                        loss_d = (
                            d_loss_fn(
                                d_real,
                                d_fake,
                                d_wrong,
                                wrong_label_weight=cfg.wrong_label_weight,
                            )
                            + cfg.aux_loss_weight * loss_aux
                        )
                    self.opt_d.zero_grad(set_to_none=True)
                    loss_d.backward()
                    self.opt_d.step()

                    loss_d_acc += loss_d.detach()
                    loss_aux_acc += loss_aux.detach()
                    d_count += 1

                # ---- Lazy R1 penalty (runs in fp32, every r1_every steps) ----
                if cfg.r1_gamma > 0.0 and self.global_step % cfg.r1_every == 0:
                    real_r1 = real.detach().float().requires_grad_(True)
                    d_real_r1 = self.D(real_r1, labels)
                    r1 = _r1_penalty(d_real_r1, real_r1) * (cfg.r1_gamma / 2.0) * cfg.r1_every
                    self.opt_d.zero_grad(set_to_none=True)
                    r1.backward()
                    self.opt_d.step()
                    r1_acc += r1.detach()
                    r1_count += 1

                # ---- Generator step ----
                z = torch.randn(bs, cfg.z_dim, device=self.device)
                fake_labels = torch.randint(0, cfg.num_classes, (bs,), device=self.device)
                with self._autocast():
                    fake = self.G(z, fake_labels)
                    d_fake_g = self.D(fake, fake_labels)
                    if cfg.adv_loss == "ralsgan":
                        # Real logits act as a fixed relativistic reference for G.
                        with torch.no_grad():
                            d_real_g = self.D(real, labels)
                        loss_g_main = ralsgan_g_loss(d_real_g, d_fake_g)
                    else:
                        loss_g_main = hinge_g_loss(d_fake_g)
                    fake_class_logits = self.D.classify(fake)
                    loss_g_aux = class_consistency_loss(fake_class_logits, fake_labels)
                    loss_g = loss_g_main + cfg.g_aux_loss_weight * loss_g_aux
                self.opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                self.opt_g.step()

                # ---- EMA update ----
                decay = cfg.ema_decay if self.global_step >= cfg.ema_warmup_steps else 0.0
                _ema_update(self.G_ema, self.G, decay)

                loss_g_acc += loss_g.detach()
                loss_g_aux_acc += loss_g_aux.detach()
                count += 1
                self.global_step += 1

                if self.global_step % cfg.log_every == 0:
                    g_avg = (loss_g_acc / count).item()
                    d_avg = (loss_d_acc / d_count).item()
                    aux_avg = (loss_aux_acc / d_count).item()
                    g_aux_avg = (loss_g_aux_acc / count).item()
                    r1_avg = (r1_acc / r1_count).item() if r1_count else 0.0
                    print(
                        f"  step {self.global_step:>7d}  "
                        f"G {g_avg:.4f}  D {d_avg:.4f}  Aux {aux_avg:.4f}  "
                        f"GAux {g_aux_avg:.4f}  R1 {r1_avg:.4f}",
                        flush=True,
                    )
                    loss_g_acc.zero_()
                    loss_d_acc.zero_()
                    loss_aux_acc.zero_()
                    loss_g_aux_acc.zero_()
                    r1_acc.zero_()
                    count = 0
                    d_count = 0
                    r1_count = 0

                if cfg.diagnostics_every > 0 and self.global_step % cfg.diagnostics_every == 0:
                    self._print_diagnostics(real, labels)

                if self.global_step % cfg.sample_every == 0:
                    self._dump_samples(self.global_step)

                if cfg.keep_ckpt_every > 0 and self.global_step % cfg.keep_ckpt_every == 0:
                    self._save(epoch, keep_step_snapshot=True)

                if cfg.max_steps is not None and self.global_step >= cfg.max_steps:
                    should_stop = True
                    break

            elapsed = time.time() - t0
            print(f"[epoch {epoch:03d}] {elapsed:.1f}s", flush=True)
            if (epoch + 1) % cfg.ckpt_every == 0:
                self._save(epoch)
            if should_stop:
                self._save(epoch, keep_step_snapshot=True)
                break

    def _dump_samples(self, step: int) -> None:
        self.G_ema.eval()
        with torch.no_grad(), self._autocast():
            fake = self.G_ema(self.fixed_z, self.fixed_labels).float()
        out_path = self.cfg.sample_dir / f"step_{step:07d}.png"
        save_grid(list(fake.cpu()), out_path, ncol=8)
        self._dump_diagnostic_samples(step)

    def _dump_diagnostic_samples(self, step: int) -> None:
        diag_dir = self.cfg.sample_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        self.G_ema.eval()
        with torch.no_grad(), self._autocast():
            by_class = self.G_ema(self.class_grid_z, self.class_grid_labels).float()
            label_sweep = self.G_ema(self.label_sweep_z, self.label_sweep_labels).float()
        save_grid(list(by_class.cpu()), diag_dir / f"by_class_step_{step:07d}.png", ncol=8)
        save_grid(
            list(label_sweep.cpu()),
            diag_dir / f"label_sweep_step_{step:07d}.png",
            ncol=self.cfg.num_classes,
        )

    @torch.no_grad()
    def _print_diagnostics(self, real: torch.Tensor, labels: torch.Tensor) -> None:
        self.G_ema.eval()
        self.D.eval()
        n = min(real.size(0), 64)
        real_eval = real[:n]
        labels_eval = labels[:n]
        z = torch.randn(n, self.cfg.z_dim, device=self.device)
        fake_labels = torch.randint(0, self.cfg.num_classes, (n,), device=self.device)
        wrong_labels = _wrong_labels(labels_eval, self.cfg.num_classes)

        with self._autocast():
            fake = self.G_ema(z, fake_labels)
            real_features = self.D.encode(real_eval)
            d_real = self.D.score_from_features(real_features, labels_eval).float()
            d_wrong = self.D.score_from_features(real_features, wrong_labels).float()
            class_logits = self.D.classify_features(real_features).float()
            d_fake = self.D(fake, fake_labels).float()

            div_z = torch.randn(self.cfg.num_classes * 8, self.cfg.z_dim, device=self.device)
            div_labels = torch.arange(self.cfg.num_classes, device=self.device).repeat_interleave(8)
            div_fake = self.G_ema(div_z, div_labels).float()

        class_acc = (class_logits.argmax(dim=1) == labels_eval).float().mean()
        feature_std = real_features.float().std(dim=0).mean()
        image_diversity = div_fake.view(self.cfg.num_classes, 8, *div_fake.shape[1:]).std(dim=1)
        image_diversity = image_diversity.mean()
        fake_stats = _image_stats(fake)
        print(
            f"  diag step {self.global_step:>7d}  "
            f"Dreal {d_real.mean().item():.4f}  Dfake {d_fake.mean().item():.4f}  "
            f"Dwrong {d_wrong.mean().item():.4f}  gap {(d_real - d_wrong).mean().item():.4f}  "
            f"cls_acc {class_acc.item():.3f}  feat_std {feature_std.item():.6f}  "
            f"ema_div {image_diversity.item():.4f}  fake_std {fake_stats['std']:.4f}  "
            f"fake_min {fake_stats['min']:.4f}  fake_max {fake_stats['max']:.4f}",
            flush=True,
        )
        self.D.train()

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "step": self.global_step,
            "G": self.G.state_dict(),
            "G_ema": self.G_ema.state_dict(),
            "D": self.D.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "config": self.cfg.__dict__,
        }

    def _save(self, epoch: int, keep_step_snapshot: bool = False) -> None:
        payload = self._checkpoint_payload(epoch)
        save_checkpoint(self.cfg.ckpt_dir / "latest.pt", payload)
        if keep_step_snapshot:
            save_checkpoint(checkpoint_path_for_step(self.cfg.ckpt_dir, self.global_step), payload)
