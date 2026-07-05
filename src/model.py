"""Generator, Discriminator, and weights_init for the DCGAN (Phase 2)."""

import torch.nn as nn

LATENT_DIM = 100
GEN_FEATURE_MAPS = 64
DISC_FEATURE_MAPS = 64
NUM_CHANNELS = 3


class Generator(nn.Module):
    """Maps a latent vector [B, 100, 1, 1] to an image [B, 3, 64, 64]."""

    def __init__(self, latent_dim: int = LATENT_DIM, feature_maps: int = GEN_FEATURE_MAPS,
                 num_channels: int = NUM_CHANNELS):
        super().__init__()
        self.main = nn.Sequential(
            # [B, 100, 1, 1] -> [B, fm*8, 4, 4]
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(inplace=True),
            # -> [B, fm*4, 8, 8]
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),
            # -> [B, fm*2, 16, 16]
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
            # -> [B, fm, 32, 32]
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),
            # -> [B, num_channels, 64, 64]
            nn.ConvTranspose2d(feature_maps, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """Maps an image [B, 3, 64, 64] to a real/fake probability [B]."""

    def __init__(self, feature_maps: int = DISC_FEATURE_MAPS, num_channels: int = NUM_CHANNELS):
        super().__init__()
        self.main = nn.Sequential(
            # [B, 3, 64, 64] -> [B, fm, 32, 32] — no BatchNorm on the input layer
            nn.Conv2d(num_channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> [B, fm*2, 16, 16]
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> [B, fm*4, 8, 8]
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> [B, fm*8, 4, 4]
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # -> [B, 1, 1, 1]
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.main(img).view(-1)


def weights_init(m):
    """DCGAN paper init: Normal(0, 0.02) on Conv/ConvTranspose/BatchNorm weights, BatchNorm bias 0."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    netG = Generator().to(device)
    netG.apply(weights_init)
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    z = torch.randn(8, LATENT_DIM, 1, 1, device=device)
    fake = netG(z)
    print("Generator output shape:", fake.shape)
    assert fake.shape == (8, NUM_CHANNELS, 64, 64)

    pred = netD(fake)
    print("Discriminator output shape:", pred.shape)
    assert pred.shape == (8,)

    # Spot-check weight init std
    first_conv = next(m for m in netG.modules() if isinstance(m, torch.nn.ConvTranspose2d))
    print("First ConvTranspose2d weight std:", first_conv.weight.data.std().item())

    print("Phase 2 shape checks passed.")
