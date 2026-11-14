import math
import os
import sys
import torch
import torch.nn as nn

from einops import rearrange, repeat

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from detectors.abstract_detector import AbstractDetector
from tools.registry import DETECTOR, LOSSFUNC


@DETECTOR.register_module(module_name='C3DResNet18+SEResNet+MLP')
class C3DResNet18_SEResNet_MLP(AbstractDetector):
    """Baseline with 3D ResNet video, SE-ResNet audio and an MLP classifier."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_backbone = SEResnet(layers=[2, 2, 2, 2], num_filters=[64, 128, 256, 512])
        self.video_backbone = C3dResnet18(in_dim=3, last_dim=512, relu_type='prelu')
        self.head = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 2))
        self.loss_func_dict, self.loss_weight_dict = self.build_loss(config['losses'])

    def build_backbone(self, config):
        """Unused interface hook; both encoders are constructed in ``__init__``."""
        pass

    def build_loss(self, losses):
        """Construct named loss modules and return their modules and configured weights."""
        loss_func_dict, loss_weight_dict = {}, {}
        for loss in losses:
            loss_func_dict[loss['name']] = LOSSFUNC[loss['func']](**loss.get('params', {}))
            loss_weight_dict[loss['name']] = loss.get('weight', 1.0)

        return loss_func_dict, loss_weight_dict

    def features(self, data_dict):
        """Encode RGB video and MFCC audio into temporal feature sequences.

        Adds the singleton audio channel to ``data_dict`` in place.
        """
        data_dict['audio'] = data_dict['audio'].unsqueeze(1)  # [B, 1, T, F]
        video_feature = self.video_backbone(data_dict['video'])
        audio_feature = self.audio_backbone(data_dict['audio'])
        return {'video': video_feature, 'audio': audio_feature}  # [B, 16, 512], [B, 16, 512]

    def classifier(self, features):
        """Mean-pool both streams and return MLP logits plus pooled/fusion features."""
        video_feat = torch.mean(features['video'], dim=1)  # [B, 512]
        audio_feat = torch.mean(features['audio'], dim=1)  # [B, 512]
        fusion_feat = torch.cat((video_feat, audio_feat), dim=-1)  # [B, 1024]
        logits = self.head(fusion_feat)

        return {'logit': logits, 'fusion_feature': fusion_feat, 'audio_feature': audio_feat, 'video_feature': video_feat}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """Return overall-label cross-entropy as ``overall`` and ``CE``."""
        loss = self.loss_func_dict['CE'](pred_dict['cls'], data_dict['label'])
        return {'overall': loss, 'CE': loss}

    def forward(self, data_dict: dict, inference=False) -> dict:
        """Return two-class ``cls``, fake-class ``prob`` and modality/fusion features."""
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred['logit'], dim=1)[:, 1]
        pred_dict = {
            'video_feature': pred['video_feature'],
            'audio_feature': pred['audio_feature'],
            'fusion_feature': pred['fusion_feature'],
            'cls': pred['logit'],
            'prob': prob,
        }

        return pred_dict


"""
The following code is from https://github.com/aiiu-lab/AVSSDeepfakeDet/blob/main/src/models/audio_encoder.py
"""


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="relu", reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if relu_type == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu = nn.PReLU(num_parameters=planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEResnet(nn.Module):
    def __init__(self, layers, num_filters, **kwargs):
        super(SEResnet, self).__init__()
        block = SEBasicBlock
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=7, stride=(1, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))
        out_dim = num_filters[3] * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=-1)
        x = rearrange(x, 'b d t -> b t d')

        return x


"""
The following code is from https://github.com/aiiu-lab/AVSSDeepfakeDet/blob/main/src/models/c3d_resnet18.py
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
    def __init__(
        self,
        block,
        layers,
        num_filters=[64, 128, 256, 512],
        strides=[1, 2, 2, 2],
        relu_type="relu",
        gamma_zero=False,
        avg_pool_downsample=False,
    ):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(self._make_layer(block, num_filters[i], layers[i], stride=strides[i]))
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
            downsample = self.downsample_block(inplanes=self.inplanes, outplanes=planes * block.expansion, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        for i in range(4):
            x = self.layers[i](x)
        x = self.avgpool(x)
        x = rearrange(x, 'b d 1 1 -> b d')
        return x


class C3dResnet18(nn.Module):
    def __init__(self, in_dim=1, last_dim=128, relu_type='prelu'):
        super().__init__()
        self.v_conv3D = nn.Sequential(
            nn.Conv3d(in_dim, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.PReLU(num_parameters=64) if relu_type == 'prelu' else nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.v_conv2D = ResNet(
            BasicBlock, [2, 2, 2, 2], num_filters=[last_dim // 8, last_dim // 4, last_dim // 2, last_dim], relu_type=relu_type
        )

    def forward(self, x):
        x = self.v_conv3D(x)
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x = self.v_conv2D(x)
        x = rearrange(x, '(b t) d -> b t d', b=b)

        return x
