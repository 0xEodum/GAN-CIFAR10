"""Public package import smoke tests."""


def test_public_gan_cifar_imports():
    from gan_cifar.models.dcgan import Discriminator, Generator
    from gan_cifar.training.losses import hinge_g_loss
    from gan_cifar.utils.image_io import make_grid_uint8

    assert Generator.__name__ == "Generator"
    assert Discriminator.__name__ == "Discriminator"
    assert callable(hinge_g_loss)
    assert callable(make_grid_uint8)
