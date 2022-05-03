# 公式 : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# 参考 : https://qiita.com/TaiseiYamana/items/b3e97da112d912c66563
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from cv2 import norm
from numpy import place
from sklearn.model_selection import GroupShuffleSplit


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(
    in_planes: int, out_planes: int, stride: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expantion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        super(BasicBlock, self).__init__()

        if norm_layer is None:
            """
            正規化層が指定されていなかったらBatch Norm層にする
            """
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x # 残差接続用に入力xを保持しておく

        # 1個目のblock
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        # 2個目のblock
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            # サイズが入力と変わっている場合のみdownsample
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expantion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # TODO : * groupsが意味わからん
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        width = 
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(planes)

        # TODO : width != planes * expantionてことか
        self.conv3 = conv1x1(width, planes * self.expantion)
        self.bn3 = norm_layer(planes * self.expantion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 1個目のblock
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 2個目のblock
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)

        # 3個目のblock
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_groups: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_groups

        # conv1 : 普段通り
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, paddin=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, paddinf=1)

        # conv2_x : size不変のため stride = 1
        self.layer1 = self._make_layer(block, 64, layers[0])

        # conv3_x : size変更するためこれ以降 stride = 2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expantion, num_classes)

        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode-"fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            # 残差接続後のbnの重みを0で初期化する : 精度が上がったそう
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,

    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        layers = []
        # 1層目
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                    self.base_width, norm_layer=norm_layer)
        )

        self.inplanes = place * block.expantion # 次の層のチャンネル数
        # 残りの層の追加
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, 
                        base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer)
            )
        
        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # conv_1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # conv_2 ~ xonv_5
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def _resnet(
    block: type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
)
