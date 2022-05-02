import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    # 参考 : https://qiita.com/MuAuan/items/a3b846b4cdae27189587
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.num_classes = num_classes

class VGG(nn.Module):
    # 参考 ; https://axa.biopapyrus.jp/deep-learning/object-classification/pytorch-vgg16.html
    def __init__(self):
        super(VGG, self).__init__()

        # block1
        self.conv01 = nn.Conv2d(3, 64, 3)
        self.conv02 = nn.Conv2d(64, 64, 3)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        # block2
        self.conv03 = nn.Conv2d(64, 128, 3)
        self.conv04 = nn.Conv2d(128, 128, 3)
        self.maxpool2 = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = F.relu(self.conv01(x))
        x = F.relu(self.conv02(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv03(x))
        x = F.relu(self.conv04(x))
        x = self.maxpool2(x)



