import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
device = "cuda" if torch.cuda.is_available() else "cpu"


class GeneratorBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, upsample=True):
        super(GeneratorBlock, self).__init__()
        self.use_upsample = upsample
        if self.use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
        )

    def forward(self, x):
        h = self.upsample(x) if self.use_upsample else x
        return self.conv(h)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Length Latent space = 100 --> (25, 2, 2)
        self.gen = nn.Sequential(
                GeneratorBlock(25, 32),     # Output = 4x4
                GeneratorBlock(32, 64),     # Output = 8x8
                GeneratorBlock(64, 128),    # Output = 16x16
                GeneratorBlock(128, 256),   # Output = 32x32
                GeneratorBlock(256, 256))   # Output = 64x64

        self.final_conv = nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.gen(x)
        return self.final_conv(h)


class DiscriminatorBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, padding=1, stride=1, downsizing=True):
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=stride + downsizing, padding=padding, bias=False),
            nn.BatchNorm2d(c_out))

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            DiscriminatorBlock(3, 16, kernel_size=1, padding=0, downsizing=False),
            DiscriminatorBlock(16, 32),
            DiscriminatorBlock(32, 64),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        h = self.disc(x).squeeze()
        return self.classifier(h)