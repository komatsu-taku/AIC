import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, images):
        super(Discriminator, self).__init__()

        B, C, H, W = images.shape

        self.conv1 = nn.Conv2d(in_channels=C, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.lrelu = nn.LeakyReLU(inplace=True)

        # calc image size after downsampling
        ds_size_H = H // 2 ** 4
        ds_size_W = W // 2 ** 4

        self.fc = nn.Linear(128*ds_size_H*ds_size_W, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.conv1(img)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.lrelu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.lrelu(out)
        
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out

class Generator(nn.Module):
    def __init__(self, latent_dim, init_size, out_channels):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim, 128*init_size**2)
        self.bn1 = nn.BatchNorm2d(128)
        self.upsamp1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu1 = nn.LeakyReLU(inplace=True)
        self.upsamp2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.lrelu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, out_channels, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.fc1(z)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.upsamp1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu1(out)
        out = self.upsamp2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.lrelu2(out)
        out = self.conv3(out)
        out = self.sigmoid(out)

        return out
