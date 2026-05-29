"""Tests for Generator and Discriminator architectures."""
import torch
import pytest

from src.models.dcgan import ConditionalBatchNorm2d, Discriminator, Generator


def test_generator_output_shape():
    g = Generator(z_dim=64, num_channels=3, base=16, num_classes=10).eval()
    z = torch.randn(4, 64)
    labels = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        out = g(z, labels)
    assert out.shape == (4, 3, 32, 32)


def test_generator_output_range():
    g = Generator(z_dim=64, num_channels=3, base=16, num_classes=10).eval()
    z = torch.randn(4, 64)
    labels = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        out = g(z, labels)
    assert out.min() >= -1.0 - 1e-5
    assert out.max() <=  1.0 + 1e-5


def test_generator_uses_labels():
    g = Generator(z_dim=64, num_channels=3, base=16, num_classes=10).eval()
    z = torch.randn(2, 64).repeat_interleave(2, dim=0)
    labels_a = torch.tensor([0, 0, 1, 1])
    labels_b = torch.tensor([1, 1, 0, 0])
    with torch.no_grad():
        out_a = g(z, labels_a)
        out_b = g(z, labels_b)
    assert not torch.allclose(out_a, out_b)


def test_generator_rejects_wrong_rank():
    g = Generator(z_dim=64, num_channels=3, base=16, num_classes=10)
    with pytest.raises(ValueError):
        g(torch.randn(4, 64, 1, 1))


def test_generator_rejects_wrong_label_shape():
    g = Generator(z_dim=64, num_channels=3, base=16, num_classes=10)
    with pytest.raises(ValueError):
        g(torch.randn(4, 64), torch.tensor([0, 1]))


def test_discriminator_output_shape():
    d = Discriminator(num_channels=3, base=16, num_classes=10).eval()
    x = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        out = d(x, labels)
    assert out.shape == (4,)


def test_discriminator_score_and_classify_shapes():
    d = Discriminator(num_channels=3, base=16, num_classes=10).eval()
    x = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        score, class_logits = d.score_and_classify(x, labels)
    assert score.shape == (4,)
    assert class_logits.shape == (4, 10)


def test_discriminator_classifier_depends_on_image():
    d = Discriminator(num_channels=3, base=16, num_classes=10).eval()
    x_a = torch.randn(4, 3, 32, 32)
    x_b = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        logits_a = d.classify(x_a)
        logits_b = d.classify(x_b)
    assert not torch.allclose(logits_a, logits_b)


def test_discriminator_has_spectral_norm():
    """Every Conv2d in D must carry a spectral-norm parametrization."""
    d = Discriminator(num_channels=3, base=16, num_classes=10)
    conv_count = sn_count = 0
    for m in d.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_count += 1
            if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
                sn_count += 1
    assert conv_count > 0
    assert sn_count == conv_count, f"{sn_count}/{conv_count} Conv2d layers have spectral norm"


def test_discriminator_uses_labels():
    d = Discriminator(num_channels=3, base=16, num_classes=10).eval()
    x = torch.randn(4, 3, 32, 32)
    labels_a = torch.tensor([0, 1, 2, 3])
    labels_b = torch.tensor([1, 2, 3, 4])
    with torch.no_grad():
        out_a = d(x, labels_a)
        out_b = d(x, labels_b)
    assert not torch.allclose(out_a, out_b)


def test_discriminator_true_wrong_label_gap_is_image_conditioned():
    d = Discriminator(num_channels=3, base=16, num_classes=10).eval()
    x = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    wrong = (labels + 1) % 10
    with torch.no_grad():
        true_scores = d(x, labels)
        wrong_scores = d(x, wrong)
    assert true_scores.shape == wrong_scores.shape == (4,)
    assert not torch.allclose(true_scores, wrong_scores)


def test_conditional_bn_gamma_initialized_to_one():
    """Regression: _init_weights must NOT clobber CBN affine embeddings.

    The CBN gain (gamma) must start at 1.0 so the generator emits
    full-contrast images. A bug where the generic Embedding init overwrote
    gamma with normal_(0, 0.02) caused near-gray, low-contrast samples.
    """
    g = Generator(z_dim=64, num_channels=3, base=16, num_classes=10)
    for m in g.modules():
        if isinstance(m, ConditionalBatchNorm2d):
            nf = m.embed.weight.shape[1] // 2
            gamma = m.embed.weight[:, :nf]
            beta = m.embed.weight[:, nf:]
            assert torch.allclose(gamma, torch.ones_like(gamma)), "CBN gamma must init to 1.0"
            assert torch.allclose(beta, torch.zeros_like(beta)), "CBN beta must init to 0.0"


def test_generator_emits_full_contrast_at_init():
    """A freshly initialized generator must produce high-variance output.

    In train mode (BN normalizes to unit variance) the output std should be
    well above the collapsed-gray regime (~0.09). We require >0.25.
    """
    g = Generator(z_dim=64, num_channels=3, base=16, num_classes=10).train()
    z = torch.randn(64, 64)
    labels = torch.arange(64) % 10
    with torch.no_grad():
        out = g(z, labels)
    assert out.std().item() > 0.25, f"generator output std too low: {out.std().item():.4f}"


def test_generator_grad_flows():
    """Gradient must reach G's parameters when backpropagating through D."""
    g = Generator(z_dim=64, num_channels=3, base=16, num_classes=10)
    d = Discriminator(num_channels=3, base=16, num_classes=10)
    z = torch.randn(2, 64)
    labels = torch.tensor([0, 1])
    fake = g(z, labels)
    loss = d(fake, labels).sum()
    loss.backward()
    assert any(p.grad is not None for p in g.parameters())
