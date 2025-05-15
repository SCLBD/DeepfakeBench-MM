import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch import nn
from einops import rearrange

from utils.registry import BACKBONE

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

@BACKBONE.register_module(module_name="resnet34")
class ResNet34(nn.Module):
    def __init__(self, block=BasicBlock, in_channels=3, layers=[3, 4, 6, 3], num_filters=[64, 128, 256, 512], strides=[2, 2, 2, 2], relu_type="relu", gamma_zero=False, avg_pool_downsample=False):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters[0], kernel_size=7, stride=2, padding=0, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.inplanes = num_filters[0]
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(self._make_layer(block, num_filters[i], layers[i], stride=strides[i]))

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1, 3), )

        # self.mlp_head = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(num_filters[-1], num_filters[-1]),
        #     nn.ReLU(),
        #     nn.Linear(num_filters[-1], num_filters[-1]),
        #     nn.ReLU(),
        #     nn.Linear(num_filters[-1], num_filters[-1])
        # )

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

    def features(self, x):
        pass

    def end_points(self, x):
        pass

    def classifier(self, x):
        pass

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        x = self.pool1(x)
        for i in range(4):
            x = self.layers[i](x)
        # average spatial dimension
        x = self.avgpool(x)
        # average temporal dimension
        x = rearrange(x, '(b t) d -> b t d', t=t)
        # x = torch.mean(x, 1)
        # x = self.mlp_head(x)
        return x


"""
Code from https://github.com/aiiu-lab/AVSSDeepfakeDet/blob/main/src/models/resnet34.py
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

