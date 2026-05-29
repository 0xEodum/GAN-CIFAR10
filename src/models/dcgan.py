"""Class-conditional ResNet GAN models for 32x32 RGB CIFAR-10 images.

The public class names stay ``Generator`` and ``Discriminator`` so older
scripts keep working, but the implementation is closer to SN-GAN than DCGAN:
residual up/down blocks, conditional batch norm in G, and a projection
discriminator in D.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


def _param_weight(module: nn.Module) -> torch.Tensor | None:
    if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
        return module.parametrizations.weight.original
    return getattr(module, "weight", None)


def _init_weights(module: nn.Module) -> None:
    # ConditionalBatchNorm2d sets its own affine embedding (gamma=1, beta=0) in
    # its constructor. Those embeddings must NOT be overwritten by the generic
    # normal_(0, 0.02) below — doing so collapses the BN gain to ~0 and forces
    # the generator to emit near-gray, low-contrast images.
    cbn_embeds = {
        id(m.embed) for m in module.modules() if isinstance(m, ConditionalBatchNorm2d)
    }
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight = _param_weight(m)
            if weight is not None:
                nn.init.orthogonal_(weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding) and id(m) not in cbn_embeds:
            weight = _param_weight(m)
            if weight is not None:
                nn.init.normal_(weight, 0.0, 0.02)


def _sn(layer: nn.Module) -> nn.Module:
    return spectral_norm(layer)


def _resolve_labels(
    labels: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if labels is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    labels = labels.to(device=device, dtype=torch.long)
    if labels.dim() != 1 or labels.size(0) != batch_size:
        raise ValueError(f"Expected labels with shape ({batch_size},), got {tuple(labels.shape)}")
    return labels


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        nn.init.ones_(self.embed.weight[:, :num_features])
        nn.init.zeros_(self.embed.weight[:, num_features:])

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.embed(labels).chunk(2, dim=1)
        return self.bn(x) * gamma[:, :, None, None] + beta[:, :, None, None]


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        residual = F.interpolate(x, scale_factor=2.0, mode="nearest")
        residual = self.skip(residual)

        out = self.bn1(x, labels)
        out = F.relu(out)
        out = F.interpolate(out, scale_factor=2.0, mode="nearest")
        out = self.conv1(out)
        out = self.bn2(out, labels)
        out = F.relu(out)
        out = self.conv2(out)
        return out + residual


class Generator(nn.Module):
    """z plus class label -> 3x32x32 image in [-1, 1]."""

    def __init__(
        self,
        z_dim: int = 128,
        num_channels: int = 3,
        base: int = 64,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.label_embed = nn.Embedding(num_classes, z_dim)
        self.project = nn.Linear(z_dim, base * 8 * 4 * 4)
        self.block1 = GeneratorBlock(base * 8, base * 4, num_classes)
        self.block2 = GeneratorBlock(base * 4, base * 2, num_classes)
        self.block3 = GeneratorBlock(base * 2, base, num_classes)
        self.bn = ConditionalBatchNorm2d(base, num_classes)
        self.out = nn.Conv2d(base, num_channels, 3, padding=1)
        _init_weights(self)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"Expected (B, z_dim), got {tuple(z.shape)}")
        labels = _resolve_labels(labels, z.size(0), z.device)
        h = z + self.label_embed(labels)
        h = self.project(h).view(z.size(0), -1, 4, 4)
        h = self.block1(h, labels)
        h = self.block2(h, labels)
        h = self.block3(h, labels)
        h = self.bn(h, labels)
        h = F.relu(h)
        return torch.tanh(self.out(h))


class OptimizedDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = _sn(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
        self.conv2 = _sn(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
        self.skip = _sn(nn.Conv2d(in_channels, out_channels, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.avg_pool2d(self.skip(x), 2)
        out = self.conv1(x)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = F.avg_pool2d(out, 2)
        return out + residual


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        self.downsample = downsample
        self.learned_skip = downsample or in_channels != out_channels
        self.conv1 = _sn(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
        self.conv2 = _sn(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
        self.skip = (
            _sn(nn.Conv2d(in_channels, out_channels, 1, bias=False))
            if self.learned_skip
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.learned_skip:
            residual = self.skip(residual)
        if self.downsample:
            residual = F.avg_pool2d(residual, 2)

        out = F.leaky_relu(x, 0.2)
        out = self.conv1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        return out + residual


class Discriminator(nn.Module):
    """3x32x32 image plus class label -> scalar logit per sample."""

    def __init__(
        self,
        num_channels: int = 3,
        base: int = 64,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.block1 = OptimizedDiscriminatorBlock(num_channels, base)
        self.block2 = DiscriminatorBlock(base, base * 2, downsample=True)
        self.block3 = DiscriminatorBlock(base * 2, base * 4, downsample=True)
        self.block4 = DiscriminatorBlock(base * 4, base * 8, downsample=False)
        self.linear = _sn(nn.Linear(base * 8, 1, bias=False))
        self.embed = _sn(nn.Embedding(num_classes, base * 8))
        self.aux_linear = _sn(nn.Linear(base * 8, num_classes, bias=False))
        _init_weights(self)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = F.leaky_relu(h, 0.2)
        return h.sum(dim=(2, 3))

    def score_from_features(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = _resolve_labels(labels, features.size(0), features.device)
        out = self.linear(features).squeeze(1)
        projection = (self.embed(labels) * features).sum(dim=1)
        return out + projection

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.aux_linear(features)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.encode(x))

    def score_and_classify(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = _resolve_labels(labels, x.size(0), x.device)
        features = self.encode(x)
        return self.score_from_features(features, labels), self.classify_features(features)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        labels = _resolve_labels(labels, x.size(0), x.device)
        return self.score_from_features(self.encode(x), labels)
