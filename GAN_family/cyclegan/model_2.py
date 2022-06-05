import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m : torch.nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ResNetBlock(nn.Module):
    def __init__(self, in_features: int):
        super(ResNetBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        identity = x

        out = self.block1(x)
        out = self.block2(out)

        out = out + identity
        return F.relu(out)


class Generator(nn.Module):
    def __init__(self, input_shape: int, num_resnet_block: int = 9) -> None:
        super(Generator, self).__init__()

        channels = input_shape[0]

        # initial block : 濃い緑部分
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]

        in_features = out_features # 次の層への入力

        # DownSampling ; 図の黄緑部分
        for _ in range(2):
            out_features *= 2

            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features # 次の層への入力

        # RsnetBlock
        for _ in range(num_resnet_block):
            model += [ResNetBlock(in_features)]
        
        # UpSampling : 図の右の方
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer : Final layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __inti__(self, input_shape):
        super(Discriminator, self).__init__()

        C, H, W = input_shape

        # caluculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, H // 2 ** 4, W // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(C, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
