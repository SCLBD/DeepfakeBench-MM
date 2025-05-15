import math
import os
import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from einops import rearrange

from utils.registry import BACKBONE

@BACKBONE.register_module(module_name="c3dr18")
class C3DResNet18(nn.Module):
    def __init__(self, in_dim=1, last_dim=128, relu_type='prelu'):
        super(C3DResNet18, self).__init__()
        """ Constructor
        Args:
            c3dr18: configuration file with the dict format
        """
        self.v_conv3D = nn.Sequential(
            nn.Conv3d(in_dim, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.PReLU(num_parameters=64) if relu_type == 'prelu' else nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.v_conv2D = ResNet(BasicBlock, [2, 2, 2, 2],
                               num_filters=[last_dim // 8, last_dim // 4, last_dim // 2, last_dim], relu_type=relu_type)

    def features(self, x):
        pass

    def end_points(self, x):
        pass

    def classifier(self, x):
        pass

    def forward(self, x):
        x = self.v_conv3D(x)
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x = self.v_conv2D(x)
        x = rearrange(x, '(b t) d -> b t d', b=b)

        return x


"""
Code from https://github.com/aiiu-lab/AVSSDeepfakeDet/blob/main/src/models/c3d_resnet18.py
"""

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_basic_block_v2(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(outplanes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="relu"):
        super(BasicBlock, self).__init__()

        assert relu_type in ["relu", "prelu"]

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception("relu type not implemented")
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_filters=[64, 128, 256, 512], strides=[1, 2, 2, 2], relu_type="relu",
                 gamma_zero=False, avg_pool_downsample=False):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(self._make_layer(block, num_filters[i], layers[i], stride=strides[i]))
        # self.layer1 = self._make_layer(block, num_filters[0], layers[0], stride=strides[0])
        # self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=strides[1])
        # self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=strides[2])
        # self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=strides[3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes, outplanes=planes * block.expansion, stride=stride
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        for i in range(4):
            x = self.layers[i](x)
        x = self.avgpool(x)
        x = rearrange(x, 'b d 1 1 -> b d')
        return x

