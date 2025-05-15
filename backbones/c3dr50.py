import math
import os
import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from einops import rearrange

from utils.registry import BACKBONE

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, max_pool=False, conv_alter=False):
        super().__init__()

        self.downsample = downsample

        self.conv1 = conv3x1x1(in_planes, planes) if not conv_alter else conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv1x1x1(planes, planes, stride)
        if max_pool:
            self.bn2 = nn.Sequential(
                        nn.BatchNorm3d(planes),
                        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
                    )
        else:
            self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@BACKBONE.register_module(module_name="c3dr50")
class C3DResNet50(nn.Module):
    def __init__(self, in_channels=1,
                frames_per_clip=25,
                block=Bottleneck,
                layers=[3, 4, 6, 3],
                block_inplanes=[64, 128, 256, 512],):
        super(C3DResNet50, self).__init__()
        """ Constructor
        Args:
            c3dr50: configuration file with the dict format
        """
        self.in_planes = block_inplanes[0]

        self.conv1 = nn.Conv3d(in_channels,
                               self.in_planes,
                               kernel_size=(5, 1, 1),
                               stride=(1, 1, 1),
                               padding=(2, 0, 0),
                               bias=False)
        # self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.bn1 = nn.Sequential(
            nn.BatchNorm3d(self.in_planes),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0])
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], max_pool=True, conv_alter=True)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], max_pool=True, conv_alter=True)
        # self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], max_pool=True)

        self.avgpool = nn.Sequential(
            # nn.AdaptiveAvgPool3d((frames_per_clip//2, 1, 1)),
            nn.AvgPool3d(kernel_size=(1, 14, 14), stride=(1, 14, 14)),
            nn.Flatten(2, 4), )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, stride=1, max_pool=False, conv_alter=False):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if max_pool:
                downsample = nn.Sequential(
                        conv1x1x1(self.in_planes, planes * block.expansion, stride),
                        nn.BatchNorm3d(planes * block.expansion),
                        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0) if max_pool else nn.Identity()),)
            else:
                downsample = nn.Sequential(
                        conv1x1x1(self.in_planes, planes * block.expansion, stride),
                        nn.BatchNorm3d(planes * block.expansion),)


        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  max_pool=max_pool,))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, conv_alter=conv_alter and i%2))

        return nn.Sequential(*layers)

    def features(self, x):
        pass

    def end_points(self, x):
        pass

    def classifier(self, x):
        pass

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)
        x = self.maxpool2(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        # x = self.layer4(x)
        # print(x.shape)

        x = self.avgpool(x)
        # print(x.shape)
        x = rearrange(x, 'b d t -> b t d')
        # print(x.shape)
        return x


"""
Code from https://github.com/aiiu-lab/AVSSDeepfakeDet/blob/main/src/models/c3dr50.py
"""

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv3x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=(3, 1, 1),
                     stride=stride,
                     padding=(1, 0, 0),
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


