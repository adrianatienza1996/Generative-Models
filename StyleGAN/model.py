import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(c_in, c_out)
        self.activation = nn.ReLU()

    def forward(self, x):
        h = self.linear(x)
        return self.activation(h)


class MappingNoise(nn.Module):
    def __init__(self, c_in = 512, c_out=512):
        super(MappingNoise, self).__init__()
        self.net = nn.Sequential(
            LinearBlock(c_in, c_out),
            LinearBlock(c_in, c_out),
            LinearBlock(c_in, c_out),
            LinearBlock(c_in, c_out),
            LinearBlock(c_in, c_out),
            nn.Linear(c_in, c_out)
        )
    def forward(self, x):
        return self.net(x)


class NoiseInjector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(  # You use nn.Parameter so that these weights can be optimized
            torch.randn(1, channels, 1, 1))

    def forward(self, image):
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        noise = torch.randn(noise_shape, device=image.device)  # Creates the random noise
        return image + self.weight * noise


class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)

    def forward(self, image, w):
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image


class GeneratorMiniBlock(nn.Module):
    def __init__(self, in_chan=512, out_chan=512, w_dim=512, kernel_size=3, stride=1, padding=1):
        super(GeneratorMiniBlock, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding)
        self.inject_noise = NoiseInjector(out_chan)
        self.adain = AdaIN(out_chan, w_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        h = self.conv(x)
        h = self.inject_noise(h)
        h = self.activation(h)
        h = self.adain(h, w)
        return h


class GeneratorBlock(nn.Module):
    def __init__(self, c_in=512, c_out=512, use_upsample=True):
        super(GeneratorBlock, self).__init__()
        self.use_upsaple= use_upsample
        if self.use_upsaple:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.miniblock1 = GeneratorMiniBlock(in_chan=c_in, out_chan=c_out)
        self.miniblock2 = GeneratorMiniBlock(in_chan=c_out, out_chan=c_out)

    def upsample_to_match_size(self, smaller_image, bigger_image):
        return F.interpolate(smaller_image, size=bigger_image.shape[-2:], mode='bilinear')

    def forward(self, x, w, alpha):
        h = self.upsample(x) if self.use_upsaple else x
        h1 = self.miniblock1(h, w)
        h2 = self.miniblock2(h1, w)
        h1 = self.upsample_to_match_size(h1, h2)
        return h1*alpha + h2*(1 - alpha)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.alpha_values = [1, 1, 1, 1, 1, 1, 1]
        self.mapping_noise = MappingNoise()
        self.block1 = GeneratorBlock(use_upsample=False)
        self.block2 = GeneratorBlock()
        self.block3 = GeneratorBlock()
        self.block4 = GeneratorBlock(c_in=512, c_out=256)
        self.block5 = GeneratorBlock(c_in=256, c_out=128)
        self.block6 = GeneratorBlock(c_in=128, c_out=64)
        self.block7 = GeneratorBlock(c_in=64, c_out=32)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)

    def forward (self, x, w):
        w = self.mapping_noise(w)
        h = self.block1(x, w, self.alpha_values[0])
        h = self.block2(h, w, self.alpha_values[1])
        h = self.block3(h, w, self.alpha_values[2])
        h = self.block4(h, w, self.alpha_values[3])
        h = self.block5(h, w, self.alpha_values[4])
        h = self.block6(h, w, self.alpha_values[5])
        h = self.block7(h, w, self.alpha_values[6])
        return self.final_conv(h)

    def set_alpha(self, index, value):
        self.alpha_values[index] = value



