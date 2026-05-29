"""Tests for GAN loss functions."""
import torch
import pytest

from src.training.losses import (
    class_consistency_loss,
    conditional_hinge_d_loss,
    conditional_ralsgan_d_loss,
    hinge_d_loss,
    hinge_g_loss,
    ralsgan_d_loss,
    ralsgan_g_loss,
)


def _logits(n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(n), torch.randn(n)


def test_d_loss_finite():
    real, fake = _logits()
    loss = ralsgan_d_loss(real, fake)
    assert torch.isfinite(loss)


def test_g_loss_finite():
    real, fake = _logits()
    loss = ralsgan_g_loss(real, fake)
    assert torch.isfinite(loss)


def test_d_loss_non_negative():
    real, fake = _logits()
    assert ralsgan_d_loss(real, fake).item() >= 0.0


def test_g_loss_non_negative():
    real, fake = _logits()
    assert ralsgan_g_loss(real, fake).item() >= 0.0


def test_d_loss_at_optimum():
    """RaLSGAN D loss is 0 when real - mean(fake) = 1 and fake - mean(real) = -1."""
    real = torch.full((8,), 0.5)
    fake = torch.full((8,), -0.5)
    assert ralsgan_d_loss(real, fake).item() < 1e-5


def test_d_loss_has_gradient():
    real = torch.randn(8, requires_grad=True)
    fake = torch.randn(8, requires_grad=True)
    loss = ralsgan_d_loss(real, fake)
    loss.backward()
    assert real.grad is not None and fake.grad is not None


def test_hinge_d_loss_finite_and_non_negative():
    real, fake = _logits()
    loss = hinge_d_loss(real, fake)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_hinge_g_loss_finite_and_has_gradient():
    fake = torch.randn(8, requires_grad=True)
    loss = hinge_g_loss(fake)
    loss.backward()
    assert torch.isfinite(loss)
    assert fake.grad is not None


def test_conditional_hinge_d_loss_includes_wrong_label_term():
    real = torch.full((4,), 2.0)
    fake = torch.full((4,), -2.0)
    wrong_good = torch.full((4,), -2.0)
    wrong_bad = torch.full((4,), 2.0)
    loss_good = conditional_hinge_d_loss(real, fake, wrong_good)
    loss_bad = conditional_hinge_d_loss(real, fake, wrong_bad)
    assert loss_bad > loss_good


def test_conditional_hinge_d_loss_matches_base_without_wrong_labels():
    real, fake = _logits()
    assert torch.allclose(conditional_hinge_d_loss(real, fake), hinge_d_loss(real, fake))


def test_conditional_ralsgan_d_loss_matches_base_without_wrong_labels():
    real, fake = _logits()
    assert torch.allclose(conditional_ralsgan_d_loss(real, fake), ralsgan_d_loss(real, fake))


def test_conditional_ralsgan_d_loss_penalizes_wrong_label():
    real = torch.full((4,), 0.5)
    fake = torch.full((4,), -0.5)
    wrong_good = torch.full((4,), -0.5)  # wrong-labeled real scored low: good
    wrong_bad = torch.full((4,), 0.5)    # wrong-labeled real scored high: bad
    loss_good = conditional_ralsgan_d_loss(real, fake, wrong_good)
    loss_bad = conditional_ralsgan_d_loss(real, fake, wrong_bad)
    assert loss_bad > loss_good


def test_conditional_ralsgan_d_loss_has_gradient():
    real = torch.randn(8, requires_grad=True)
    fake = torch.randn(8, requires_grad=True)
    wrong = torch.randn(8, requires_grad=True)
    conditional_ralsgan_d_loss(real, fake, wrong).backward()
    assert real.grad is not None and fake.grad is not None and wrong.grad is not None


def test_class_consistency_loss_matches_cross_entropy():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.5, -0.5]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss = class_consistency_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
