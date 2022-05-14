import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    # 参考 : https://qiita.com/MuAuan/items/a3b846b4cdae27189587
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        
        # block1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # block2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # block3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # block4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # block5
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512*2*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)    
        )

    def forward(self, x):
        # feature extractor
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # classifier
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class VGG(nn.Module):
    # 参考 ; https://axa.biopapyrus.jp/deep-learning/object-classification/pytorch-vgg16.html
    def __init__(self, num_classes):
        super(VGG, self).__init__()

        # block1
        self.conv01 = nn.Conv2d(3, 64, 3, 1, 1,)
        self.conv02 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        # block2
        self.conv03 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv04 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        # block3
        self.conv05 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv06 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv07 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, 2)

        # block4
        self.conv08 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv09 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, 2)

        # block5
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool5 = nn.MaxPool2d(2, 2)

        # classifier
        self.fc1 = nn.Linear(in_features=512*2*2, out_features=4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # block1
        x = F.relu(self.conv01(x))
        x = F.relu(self.conv02(x))
        x = self.maxpool1(x)

        # block2
        x = F.relu(self.conv03(x))
        x = F.relu(self.conv04(x))
        x = self.maxpool2(x)

        # block3
        x = F.relu(self.conv05(x))
        x = F.relu(self.conv06(x))
        x = F.relu(self.conv07(x))
        x = self.maxpool3(x)

        # block4
        x = F.relu(self.conv08(x))
        x = F.relu(self.conv09(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool4(x)

        # block5
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool5(x)

        # classefier
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x



