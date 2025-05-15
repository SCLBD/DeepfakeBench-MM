# Functions in the Class are summarized as:
# 1. __init__: Initialization
# 2. build_backbone: Backbone-building
# 3. build_loss: Loss-function-building
# 4. features: Feature-extraction
# 5. classifier: Classification
# 6. get_losses: Loss-computation
# 7. forward: Forward-propagation

# Reference:
# @inproceedings{10.1145/3394171.3413700,
#     author = {Chugh, Komal and Gupta, Parul and Dhall, Abhinav and Subramanian, Ramanathan},
#     title = {Not made for each other- Audio-Visual Dissonance-based Deepfake Detection and Localization},
#     year = {2020},
#     isbn = {9781450379885},
#     publisher = {Association for Computing Machinery},
#     address = {New York, NY, USA},
#     url = {https://doi.org/10.1145/3394171.3413700},
#     doi = {10.1145/3394171.3413700},
#     abstract = {We propose detection of deepfake videos based on the dissimilarity between the audio and visual modalities, termed as the Modality Dissonance Score (MDS). We hypothesize that manipulation of either modality will lead to dis-harmony between the two modalities, e.g., loss of lip-sync, unnatural facial and lip movements, etc. MDS is computed as the mean aggregate of dissimilarity scores between audio and visual segments in a video. Discriminative features are learnt for the audio and visual channels in a chunk-wise manner, employing the cross-entropy loss for individual modalities, and a contrastive loss that models inter-modality similarity. Extensive experiments on the DFDC and DeepFake-TIMIT Datasets show that our approach outperforms the state-of-the-art by up to 7\%. We also demonstrate temporal forgery localization, and show how our technique identifies the manipulated video segments.},
#     booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
#     pages = {439–447},
#     numpages = {9},
#     keywords = {neural networks, modality dissonance, deepfake detection and localization, contrastive loss},
#     location = {Seattle, WA, USA},
#     series = {MM '20}
# }
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from metrics.base_metrics_class import calculate_metrics_for_train

from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from losses import LOSSFUNC


@DETECTOR.register_module(module_name='MDS')
class MDSDetector(AbstractDetector):
    def __init__(self, config, num_layers_in_fc_layers=1024, dropout=0.5):
        super().__init__()
        self.config = config
        self.video_backbone, self.audio_backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.train_stage_idx = 0

        # From https://github.com/abhinavdhall/deepfake/blob/main/ACM_MM_2020/model.py
        self.final_bn_lip = nn.BatchNorm1d(num_layers_in_fc_layers)
        self.final_bn_lip.weight.data.fill_(1)
        self.final_bn_lip.bias.data.zero_()
        self.final_fc_lip = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
        initialize_weights(self.final_fc_lip)

        self.final_bn_aud = nn.BatchNorm1d(num_layers_in_fc_layers)
        self.final_bn_aud.weight.data.fill_(1)
        self.final_bn_aud.bias.data.zero_()
        self.final_fc_aud = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, 2))
        initialize_weights(self.final_fc_aud)

    def build_backbone(self, config):
        audio_backbone = Audio_RNN()
        video_backbone = ResNet2d3d_full(block=[BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d],
                                         layers=[2, 2, 2, 2], track_running_stats=False)
        return video_backbone, audio_backbone

    def build_loss(self, config):
        # prepare the loss function
        contrastive_loss = LOSSFUNC[config['loss_func'][0]]
        video_cross_entropy = LOSSFUNC[config['loss_func'][1]]
        audio_cross_entropy = LOSSFUNC[config['loss_func'][2]]

        return {'L1': contrastive_loss(), 'L2': video_cross_entropy(), 'L3': audio_cross_entropy()}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        # pred = pred_dict['cls']
        prob = pred_dict['prob']
        # auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), prob.detach())

        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def features(self, data_dict: dict) -> torch.tensor:
        video_feature = self.video_backbone(data_dict['video'])
        audio_feature = self.audio_backbone(data_dict['audio'])
        return {'video': video_feature, 'audio': audio_feature}

    def classifier(self, features: torch.tensor) -> torch.tensor:
        video_output = self.final_bn_lip(features['video'])
        video_output = self.final_fc_lip(video_output)
        audio_output = self.final_bn_aud(features['audio'])
        audio_output = self.final_fc_aud(audio_output)

        return {'video': video_output, 'audio': audio_output}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        loss1 = self.loss_func['L1'](pred_dict['mds'], data_dict['label'])
        loss2 = self.loss_func['L2'](pred_dict['cls']['video'], data_dict['label'])
        loss3 = self.loss_func['L2'](pred_dict['cls']['audio'], data_dict['label'])

        return {'overall': loss1 + loss2 + loss3, 'L1': loss1, 'L2': loss2, 'L3': loss3}


    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)

        mds_score = torch.norm(features['video'] - features['audio'], p=2, dim=1, keepdim=True)
        prob = (mds_score > 0.6).float()
        # we use this in most cases!!!
        # # get the probability of the pred
        # prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {
            'cls': pred,
            'prob': prob,
            # 'feat': features,
            'mds': mds_score}
        return pred_dict


"""
The following code is from https://github.com/abhinavdhall/deepfake/blob/main/ACM_MM_2020/model.py
"""
def initialize_weights(module):
    for m in module:
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Dropout):
            pass
        else:
            m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None: m.bias.data.zero_()


class Audio_RNN(nn.Module):
    def __init__(self, num_layers_in_fc_layers=1024):
        super().__init__()
        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),  # [B, 64, 13, 99]

            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),  # [B, 192, 11, 49]

            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),  # [B, 384, 11, 49]

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [B, 256, 11, 49]

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # [B, 256, 5, 24]

            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),  # [B, 512, 1, 21]
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.netfcaud = nn.Sequential(
            nn.Linear(512 * 21, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, num_layers_in_fc_layers),
        )
        initialize_weights(self.netcnnaud)
        initialize_weights(self.netfcaud)

    def forward(self, audio):
        # Audio is loaded as [B, 1, 1, 13, 99]. In this benchmark, audio is loaded as [B, 99, 13]
        audio = audio.permute(0, 2, 1).unsqueeze(1)
        mid = self.netcnnaud(audio)
        mid = mid.view(audio.shape[0], -1)
        audio_feature = self.netfcaud(mid)
        return audio_feature


"""
The following code is from https://github.com/abhinavdhall/deepfake/blob/main/ACM_MM_2020/resnet_2d3d.py
"""


class ResNet2d3d_full(nn.Module):
    def __init__(self, block, layers, track_running_stats=True, num_layers_in_fc_layers=1024):
        super(ResNet2d3d_full, self).__init__()
        self.inplanes = 64
        self.track_running_stats = track_running_stats
        bias = False
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=bias)
        self.bn1 = nn.BatchNorm3d(64, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        if not isinstance(block, list):
            block = [block] * 4

        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block[3], 256, layers[3], stride=2, is_final=True)
        # modify layer4 from exp=512 to exp=256
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # [Modified] Merge FC layers into backbone network class
        self.netfclip = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, num_layers_in_fc_layers),
        )
        initialize_weights(self.netfclip)

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # customized_stride to deal with 2d or 3d residual blocks
            if (block == Bottleneck2d) or (block == BasicBlock2d):
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride

            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=customized_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion, track_running_stats=self.track_running_stats)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, track_running_stats=self.track_running_stats))
        self.inplanes = planes * block.expansion
        if is_final:  # if is final block, no ReLU in the final output
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes, track_running_stats=self.track_running_stats))
            layers.append(
                block(self.inplanes, planes, track_running_stats=self.track_running_stats, use_final_relu=False))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, track_running_stats=self.track_running_stats))

        return nn.Sequential(*layers)

    def _initialize_weights(self, module):
        for m in module:
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Dropout):
                pass
            else:
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # [Modified] Merge FC layers into backbone network class
        x = F.avg_pool3d(x, (7, 1, 1), stride=(1, 1, 1))
        x = x.reshape(x.shape[0], -1)
        x = self.netfclip(x)

        return x


class Bottleneck2d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(Bottleneck2d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1),
                               bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4, track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batchnorm: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm: out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batchnorm: out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(Bottleneck3d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4, track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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
        if self.use_final_relu: out = self.relu(out)

        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(BasicBlock2d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv1x3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

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
        if self.use_final_relu: out = self.relu(out)

        return out


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_final_relu=True):
        super(BasicBlock3d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv3x3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

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
        if self.use_final_relu: out = self.relu(out)

        return out


# 3x3x3 convolution with padding
def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias)


# 1x3x3 convolution with padding
def conv1x3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=bias)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out
