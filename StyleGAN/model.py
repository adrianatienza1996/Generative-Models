import torch
import torch.nn as nn


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


class ChannelPadding(nn.Module):
    def __init__(self):
        super(ChannelPadding, self).__init__()
        self.padding = nn.MaxPool3d(kernel_size=(2, 1, 1))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        return self.padding(x).squeeze()

class GeneratorBlock(nn.Module):
    def __init__(self, c_in=512, c_out=512, use_upsample=True):
        super(GeneratorBlock, self).__init__()
        self.use_upsaple = use_upsample
        self.pad_channels = c_in!=c_out
        if self.pad_channels:
            self.channel_padding = ChannelPadding()
        if self.use_upsaple:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.miniblock1 = GeneratorMiniBlock(in_chan=c_in, out_chan=c_out)
        self.miniblock2 = GeneratorMiniBlock(in_chan=c_out, out_chan=c_out)

    def forward(self, x, w, alpha):
        h = self.upsample(x) if self.use_upsaple else x
        h1 = self.miniblock1(h, w)
        h1 = self.miniblock2(h1, w)
        h = self.channel_padding(h) if self.pad_channels else h
        return h * alpha + h1 * (1 - alpha)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.alpha_values = [1, 1, 1, 1, 1, 1]
        self.mapping_noise = MappingNoise()
        self.block1 = GeneratorBlock(use_upsample=False)
        self.block2 = GeneratorBlock()
        self.block3 = GeneratorBlock(c_in=512, c_out=256)
        self.block4 = GeneratorBlock(c_in=256, c_out=128)
        self.block5 = GeneratorBlock(c_in=128, c_out=64)
        self.block6 = GeneratorBlock(c_in=64, c_out=32)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x, w):
        w = self.mapping_noise(w)
        h = self.block1(x, w, self.alpha_values[0])
        h = self.block2(h, w, self.alpha_values[1])
        h = self.block3(h, w, self.alpha_values[2])
        h = self.block4(h, w, self.alpha_values[3])
        h = self.block5(h, w, self.alpha_values[4])
        h = self.block6(h, w, self.alpha_values[5])
        #h = self.block7(h, w, self.alpha_values[6])
        return self.final_conv(h)

    def set_alpha(self, index, value):
        self.alpha_values[index] = min(value,  self.alpha_values[index])

class Upsample_Channels(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample_Channels, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x):
        h = torch.unsqueeze(x, dim=1)
        h = self.upsampler(h)
        return h.squeeze()


class Discriminator_Block(nn.Module):
    def __init__(self, ni, no, kernel_size=3, stride=1, padding=1, downsizing=True):
        super(Discriminator_Block, self).__init__()
        self.downsizing = downsizing
        self.upsample_channels = ni != no
        if self.downsizing:
            self.maxpool = nn.MaxPool2d(kernel_size=2)
        if self.upsample_channels:
            self.upsampler = Upsample_Channels(scale_factor=(no//ni, 1, 1))

        self.conv = nn.Sequential(
            nn.Conv2d(ni, no, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(no),
            nn.LeakyReLU(0.2),
            nn.Conv2d(no, no, kernel_size=kernel_size, stride=stride, padding=padding)
        )


    def forward(self, x, alpha):
        h1 = self.upsampler(x) if self.upsample_channels else x
        h = self.conv(x)
        h = h1 * alpha + h * (1 - alpha)
        h = self.maxpool(h) if self.downsizing else h
        return h


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.alpha_values = [1, 1, 1, 1, 1, 1]
        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2))
        self.block1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1))            # 128, 128
        self.upsampler0 = Upsample_Channels(scale_factor=(16 / 3, 1, 1))
        self.upsampler1 = Upsample_Channels(scale_factor=(32//16, 1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.block2 = Discriminator_Block(32, 64)                             # 64 x 64
        self.block3 = Discriminator_Block(64, 128)                            # 32 x 32
        self.block4 = Discriminator_Block(128, 256)                           # 16 x 16
        self.block5 = Discriminator_Block(256, 512)                           # 8 x 8
        self.maxpool7 = nn.MaxPool2d(kernel_size=4)
        self.classifier = nn.Linear(512, 1)

    def set_alpha(self, index, value):
            self.alpha_values[index] = min(value,  self.alpha_values[index])


    def forward (self, x):
        h = self.alpha_values[5] * self.upsampler0(x) + (1 - self.alpha_values[5]) * self.initial_block(x)
        h = self.alpha_values[4] * self.upsampler1(h) + (1 - self.alpha_values[4]) * self.block1(h)
        h = self.maxpool1(h)
        h = self.block2(h, self.alpha_values[3])
        h = self.block3(h, self.alpha_values[2])
        h = self.block4(h, self.alpha_values[1])
        h = self.block5(h, self.alpha_values[0])
        h = self.maxpool7(h)
        return self.classifier(h.squeeze())



