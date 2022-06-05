import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        pass
    
    def forward(self, x):
        pass


class Generator(nn.Module):
    def __init__(self, latent_dims, in_features):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(latent_dims 128, bias=False)
        self.fc2 = nn.Linear(128, 256, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc3 = nn.Linear(256, 512, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.fc4 = nn.Linear(512, 1024, bias=False)
        self.bn4 = nn.BatchNorm2d(1024)
        self.lrelu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        out = torch.sigmoid(x)

        return out


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        pass

    def forward(self, x):
        pass
