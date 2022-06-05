import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discrininator(nn.Module):
    def __init__(self):
        super(Discrininator, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, image: torch.Tensor):
        x = image.view(-1, 784)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return F.sigmoid(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 4096)
        self.fc3 = nn.Linear(4096, 784)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(-1, 1, 28, 28)

        return F.tanh(x)
