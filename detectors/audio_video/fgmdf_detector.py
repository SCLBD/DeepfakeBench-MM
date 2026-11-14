import math
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from detectors.abstract_detector import AbstractDetector
from tools.registry import DETECTOR, LOSSFUNC


@DETECTOR.register_module(module_name='FGMDF')
class FGMDFDetector(AbstractDetector):
    """Graph fusion with video, audio and cross-modal classification heads."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.get('model_params', {}).get('num_classes', 2)
        self.backbone = GAT_video_audio(num_classes=self.num_classes)
        self.loss_func_dict, self.loss_weight_dict = self.build_loss(config['losses'])

        # From https://github.com/yinql1995/Fine-grained-Multimodal-DeepFake-Classification/blob/main/network/graph_video_audio_model.py
        self.video_mlp = nn.Linear(192, 2)
        self.audio_mlp = nn.Linear(64, 2)  # 64
        self.cross_mlp = nn.Linear(256, self.num_classes)  # 256   # Note: We standardize the number of classes as 2.

    def build_backbone(self, config):
        """Unused interface hook; the graph network is constructed in ``__init__``."""
        pass

    def build_loss(self, losses):
        """Construct named loss modules and return their modules and configured weights."""
        loss_func_dict, loss_weight_dict = {}, {}
        for loss in losses:
            loss_func_dict[loss['name']] = LOSSFUNC[loss['func']](**loss.get('params', {}))
            loss_weight_dict[loss['name']] = loss.get('weight', 1.0)

        return loss_func_dict, loss_weight_dict

    def features(self, data_dict):
        """Encode prepared channel-stacked video and raw audio into graph features."""
        video_feature, audio_feature = self.backbone(data_dict['video'], data_dict['audio'])
        return {'video_feature': video_feature, 'audio_feature': audio_feature}

    def classifier(self, features):
        """Return named video, audio and cross-modal logit tensors."""
        video_logit = self.video_mlp(features['video_feature'])
        audio_logit = self.audio_mlp(features['audio_feature'])
        cross_logit = self.cross_mlp(torch.cat([features['video_feature'], features['audio_feature']], dim=1))

        return {'video_logit': video_logit, 'audio_logit': audio_logit, 'cross_logit': cross_logit}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """Sum video/audio/overall cross-entropies using their corresponding labels."""
        loss1 = self.loss_func_dict['video_modality_CE'](pred_dict['cls']['video_logit'], data_dict['video_label'])
        loss2 = self.loss_func_dict['audio_modality_CE'](pred_dict['cls']['audio_logit'], data_dict['audio_label'])
        loss3 = self.loss_func_dict['cross_modality_CE'](pred_dict['cls']['cross_logit'], data_dict['label'])

        return {
            'overall': loss1 + loss2 + loss3,
            'video_modality_CE': loss1,
            'audio_modality_CE': loss2,
            'cross_modality_CE': loss3,
        }

    def forward(self, data_dict, inference=False):
        """Sample 40 of 100 input frames and return logits, fake scores and features.

        Training uses random frame offsets; inference uses uniform indices. The
        selected video is reshaped and replaced in ``data_dict`` in place.
        """
        # The input is 100 frames (4 second length). To sample 40 frames, the strategy is
        # 1) In training mode, use the same strategy in Line 74-81 of https://github.com/yinql1995/Fine-grained-Multimodal-DeepFake-Classification/blob/main/dataset/dataset.py
        # 2) In testing and validation mode, sample frame uniformly.
        if inference:
            # [0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, ..., 90, 92, 95, 98]
            slice_index = np.round(np.linspace(0, 100, 41)).astype(np.int32).tolist()[:-1]
        else:
            slice_index = np.arange(0, 10, 1)
            random.shuffle(slice_index)
            slice_index = slice_index[:4]  # randomly sample 4 frames from each 10 frame segment
            slice_index.sort()
            slice_index = slice_index.repeat(10)
            slice_index = slice_index.reshape(4, 10).transpose(1, 0)
            a = np.arange(0, 100, 10).reshape(10, 1)
            slice_index = slice_index + a
            slice_index = slice_index.reshape(-1).tolist()

        data_dict['video'] = data_dict['video'][:, :, slice_index]
        data_dict['video'] = data_dict['video'].permute(0, 2, 1, 3, 4)
        B, T, C, H, W = data_dict['video'].shape
        data_dict['video'] = data_dict['video'].reshape(B, -1, H, W)
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred['cross_logit'], dim=1)[:, 1]

        pred_dict = {
            'cls': pred,
            'prob': prob,
            'fusion_feature': torch.cat([features['video_feature'], features['audio_feature']], dim=1),
            'video_feature': features['video_feature'],
            'audio_feature': features['audio_feature'],
        }

        return pred_dict


"""
The following code is from https://github.com/yinql1995/Fine-grained-Multimodal-DeepFake-Classification/blob/main/network/graph_video_audio_model.py
"""


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        # self.att_proj = nn.Linear(in_dim, out_dim)
        # self.att_weight = self._init_new_params(out_dim, 1)
        self.att_q = nn.Linear(in_dim, out_dim)
        self.att_k = nn.Linear(in_dim, out_dim)
        self.att_v = nn.Linear(in_dim, out_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

        # project
        # self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1, groups=out_dim)

        # batch norm
        # self.bn1 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        # self.input_drop = nn.Dropout(p=0.5)

        # activate
        self.act = nn.SELU(inplace=True)

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        # x = self.input_drop(x)
        # x = self.proj_without_att(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm

        x = self._apply_BN(x, self.bn2)

        # apply activation
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''
        nb_nodes = x.size()[1]
        device = x.device

        q = F.normalize(self.att_q(x), dim=-1)  # b,n,c
        k = F.normalize(self.att_k(x), dim=-1).permute(0, 2, 1)  # b,c,n
        att = torch.matmul(q, k)  # b, n, n

        a = torch.arange(0, nb_nodes, 1).to(device)
        b = a.unsqueeze(0).expand(nb_nodes, -1)
        b = b - a.unsqueeze(1) + (torch.eye(nb_nodes).to(device) * 0.5)
        return torch.div(att + 1, b.unsqueeze(0))

        # return att

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        # att_map = torch.tanh(self.att_proj(att_map))
        # # size: (#bs, #node, #node, 1)
        # att_map = torch.matmul(att_map, self.att_weight)
        att_map = F.softmax(att_map, dim=-1)  # b, n, n

        return att_map

    def _project(self, x, att_map):
        v = self.att_v(x)  # b, n, c
        v = torch.matmul(att_map, v)

        x = self.proj_without_att(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1).contiguous()

        return x + self.gamma * v

    def _apply_BN(self, x, bn):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class Pool(nn.Module):  # new pooling

    def __init__(self, k: float, in_dim: int, p, permute=False):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim
        self.flag = permute

    def forward(self, h, adj):
        Z = self.drop(h)
        # weights = self.proj(Z)
        # scores = self.sigmoid(weights)
        scores = torch.matmul(Z, Z.permute(0, 2, 1))
        scores = torch.sum(scores, dim=-1, keepdim=False)

        new_h, adj = self.top_k_graph(scores, h, self.k, adj)

        return new_h, adj

    def top_k_graph(self, scores, h, k, adj):
        """
        args
        ====
        scores: attention-based weights (#bs,#node,1)
        h: graph (#bs,#node,#dim)
        k: ratio of remaining nodes, (float)

        """
        num_nodes = h.shape[1]
        batch_size = h.shape[0]

        # first reflect the weights and then rank them
        # H = h * scores
        # scores = scores.squeeze(-1)
        _, idx = torch.topk(scores, max(1, int(k * num_nodes)), largest=False, dim=1)
        idx = torch.sort(idx, dim=1)[0]
        new_g = []
        new_adj = []
        if self.flag:
            adj = adj.permute(0, 2, 1)

        for i in range(batch_size):
            new_g.append(h[i, idx[i][: int(len(idx[i]))], :])
            new_adj.append(adj[i, idx[i][: int(len(idx[i]))], :])
        new_g = torch.stack(new_g, dim=0)
        new_adj = torch.stack(new_adj, dim=0)

        if self.flag:
            new_adj = new_adj.permute(0, 2, 1)

        return new_g, new_adj


class DualCrossGraphAttention(nn.Module):
    def __init__(self, in_channels, vid_nodes, aud_nodes, ratio=4):
        super(DualCrossGraphAttention, self).__init__()

        self.key_conv1 = nn.Linear(in_channels, in_channels // ratio)
        self.key_conv2 = nn.Linear(in_channels, in_channels // ratio)
        self.key_conv_share = nn.Linear(in_channels // ratio, in_channels // ratio)

        self.aud_to_vid_linear = nn.Linear(aud_nodes, aud_nodes)
        self.vid_to_aud_linear = nn.Linear(vid_nodes, vid_nodes)

        # separated value conv
        self.value_conv1 = nn.Linear(in_channels, in_channels)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.value_conv2 = nn.Linear(in_channels, in_channels)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, adj):

        def _get_att(a, b, adj):

            proj_key1 = self.key_conv_share(self.key_conv1(a))  # B, vid_n, C
            proj_key2 = self.key_conv_share(self.key_conv2(b)).permute(0, 2, 1)  # b, c,aud_n
            energy = torch.bmm(proj_key1, proj_key2)  # B, vid_n, aud_n

            attention1 = self.aud_to_vid_linear(energy)  # b, vid_n, aud_n
            attention2 = self.vid_to_aud_linear(energy.permute(0, 2, 1))  # b, aud_n, vid_n

            adj_re = torch.where(
                adj > 0.0, torch.full_like(adj, 0.0).to(adj.device), torch.full_like(adj, -9e15 * 1.0).to(adj.device)
            )
            adj_re = adj_re.to(adj.device)
            attention1 = self.softmax(attention1 * adj + adj_re)
            attention2 = self.softmax(attention2 * adj.permute(0, 2, 1) + adj_re.permute(0, 2, 1))

            return attention1, attention2

        att_y_on_x, att_x_on_y = _get_att(x, y, adj)  # b, vid_n, aud_n; b, aud_n, vid_n
        proj_value_y_on_x = self.value_conv2(y)  # b, aud_n, c
        out_y_on_x = torch.bmm(att_y_on_x, proj_value_y_on_x)  # b, vid_n, c
        out_x = self.gamma1 * out_y_on_x + x

        proj_value_x_on_y = self.value_conv1(x)  # b, vid_n, c
        out_x_on_y = torch.bmm(att_x_on_y, proj_value_x_on_y)  # b, aud_n, c
        out_y = self.gamma2 * out_x_on_y + y

        return out_x, out_y


class GAT_video_audio(nn.Module):
    def __init__(self, num_classes=2):
        super(GAT_video_audio, self).__init__()
        self.num_classes = num_classes
        self.video_encoder = video_model()
        self.audio_encoder = audio_model()
        self.init_adj = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        self.cross_att1 = DualCrossGraphAttention(512, 10, 4)  # 8

        # Graph attention and pooling layer for Spectral-RawGAT
        self.GAT_layer1 = GraphAttentionLayer(512, 256)
        self.pool1 = Pool(0.75, 256, 0.3, permute=False)
        self.audio_GAT_layer1 = GraphAttentionLayer(512, 256)
        self.audio_pool1 = Pool(0.75, 256, 0.3, permute=True)
        self.cross_att2 = DualCrossGraphAttention(256, 7, 3)  # 6

        # Graph attention and pooling layer for Temporal-RawGAT
        self.GAT_layer2 = GraphAttentionLayer(256, 128)
        self.pool2 = Pool(0.75, 128, 0.3, permute=False)
        self.audio_GAT_layer2 = GraphAttentionLayer(256, 128)
        self.audio_pool2 = Pool(0.75, 128, 0.3, permute=True)
        self.cross_att3 = DualCrossGraphAttention(128, 5, 2)  # 4

        # Graph attention and pooling layer for Spectro-Temporal RawGAT
        self.GAT_layer3 = GraphAttentionLayer(128, 64)
        self.pool3 = Pool(0.75, 64, 0.3, permute=False)
        self.audio_GAT_layer3 = GraphAttentionLayer(128, 64)
        self.audio_pool3 = Pool(0.75, 64, 0.3, permute=True)
        self.cross_att4 = DualCrossGraphAttention(64, 3, 1)  # 3

        # Note: Move this part to our benchmark as model classifier
        # self.video_pre = nn.Linear(192, 2)
        # self.audio_pre = nn.Linear(64, 2)  # 64
        # self.mix_pre = nn.Linear(256, self.num_classes)  # 256

    def video_processing(self, model, inp):
        inp = inp.view(
            (
                inp.size()[0],
                10,
                -1,
            )
            + inp.size()[2:]
        )
        x = []
        for i in range(10):
            x.append(model(inp[:, i, :, :, :].contiguous()))
        x = torch.stack(x, dim=1)

        return x

    def forward(self, vid_inp, aud_inp):
        adj = self.init_adj.unsqueeze(0).expand(vid_inp.size(0), -1, -1).permute(0, 2, 1).to(vid_inp.device)
        x = self.video_processing(self.video_encoder, vid_inp)
        y = self.audio_encoder(aud_inp)

        x, y = self.cross_att1(x, y, adj)

        # first layer
        x = self.GAT_layer1(x)  # (#bs,#node(F),feat_dim(C)) --> [#b, 10, 512]
        x, adj = self.pool1(x, adj)  # [#b, 10, 512] --> [#b, 7, 1, 512]
        y = self.audio_GAT_layer1(y)
        y, adj = self.audio_pool1(y, adj)
        x, y = self.cross_att2(x, y, adj)

        # second layer
        x = self.GAT_layer2(x)  # [#b, 7, 512] --> [#b, 7, 256]
        x, adj = self.pool2(x, adj)  # [#b, 7, 256] --> [#b, 5, 1, 256]
        y = self.audio_GAT_layer2(y)
        y, adj = self.audio_pool2(y, adj)
        x, y = self.cross_att3(x, y, adj)

        # third layer
        x = self.GAT_layer3(x)  # [#b, 7, 256] --> [#b, 5, 128]
        x, adj = self.pool3(x, adj)  # [#b, 5, 128] --> [#b, 2, 1, 64]
        y = self.audio_GAT_layer3(y)
        y, adj = self.audio_pool3(y, adj)
        x, y = self.cross_att4(x, y, adj)
        x = x.flatten(1)
        y = y.flatten(1)

        # Note: Move this part to our benchmark as model classifier
        # video_out = self.video_pre(x)
        # audio_out = self.audio_pre(y)
        # fusion_out = torch.cat([x, y], dim=1)
        # mix_out = self.mix_pre(fusion_out)
        #
        # return mix_out, video_out, audio_out, fusion_out
        return x, y


"""
The following code is from https://github.com/yinql1995/Fine-grained-Multimodal-DeepFake-Classification/blob/main/network/video_processing_model.py
"""

num_frames = 4


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048 * num_frames, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False), nn.BatchNorm2d(out_chan), nn.ReLU())
        self.ca = ChannelAttention(out_chan, ratio=16)

    def forward(self, x):
        fuse_fea = self.convblk(x)
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea


class Horizontal_Vertical_att(nn.Module):
    def __init__(self, in_filters, ratio=4):
        super(Horizontal_Vertical_att, self).__init__()

        self.horizontal_conv = nn.Sequential(
            SeparableConv2d(in_filters, in_filters // ratio, (3, 1), 1, (1, 0)),
            nn.BatchNorm2d(in_filters // ratio),
        )
        self.vertical_conv = nn.Sequential(
            SeparableConv2d(in_filters, in_filters // ratio, (1, 3), 1, (0, 1)),
            nn.BatchNorm2d(in_filters // ratio),
        )

        self.avg_pool_row = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.avg_pool_row2 = nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.avg_pool_col = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.avg_pool_col2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        self.horizontal_conv_down = nn.Sequential(
            SeparableConv2d(in_filters, in_filters // ratio, (3, 1), 1, (1, 0)),
            nn.BatchNorm2d(in_filters // ratio),
        )
        self.vertical_conv_down = nn.Sequential(
            SeparableConv2d(in_filters, in_filters // ratio, (1, 3), 1, (0, 1)),
            nn.BatchNorm2d(in_filters // ratio),
        )

        self.horizontal_conv_down2 = nn.Sequential(
            SeparableConv2d(in_filters, in_filters // ratio, (3, 1), 1, (1, 0)),
            nn.BatchNorm2d(in_filters // ratio),
        )
        self.vertical_conv_down2 = nn.Sequential(
            SeparableConv2d(in_filters, in_filters // ratio, (1, 3), 1, (0, 1)),
            nn.BatchNorm2d(in_filters // ratio),
        )

        self.horizontal_conv_merge = nn.Sequential(
            nn.Conv2d(in_filters // ratio, in_filters, 1),
            nn.BatchNorm2d(in_filters),
        )
        self.vertical_conv_merge = nn.Sequential(
            nn.Conv2d(in_filters // ratio, in_filters, 1),
            nn.BatchNorm2d(in_filters),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x.view(
            (
                x.size(0) // num_frames,
                num_frames,
            )
            + x.size()[1:]
        )
        B, T, C, H, W = input.size()

        input_col = input.permute(0, 4, 2, 3, 1).contiguous()  # B W C H T
        input_col = input_col.view((-1,) + input_col.size()[2:])
        input_col_down = self.avg_pool_col(input_col)
        input_col_down2 = self.avg_pool_col2(input_col)

        input_row = input.permute(0, 3, 2, 1, 4).contiguous()  # B H C T W
        input_row = input_row.view((-1,) + input_row.size()[2:])
        input_row_down = self.avg_pool_row(input_row)
        input_row_down2 = self.avg_pool_row2(input_row)

        input_col = self.vertical_conv(input_col)
        input_row = self.horizontal_conv(input_row)

        input_col_down = self.vertical_conv_down(input_col_down)
        input_row_down = self.horizontal_conv_down(input_row_down)

        input_col_down2 = self.vertical_conv_down2(input_col_down2)
        input_row_down2 = self.horizontal_conv_down2(input_row_down2)

        input_col_down = F.interpolate(input_col_down, input_col.size()[2:])
        input_row_down = F.interpolate(input_row_down, input_row.size()[2:])

        input_col_down2 = F.interpolate(input_col_down2, input_col.size()[2:])
        input_row_down2 = F.interpolate(input_row_down2, input_row.size()[2:])

        col = self.vertical_conv_merge(1 / 3 * input_col + 1 / 3 * input_col_down + 1 / 3 * input_col_down2)
        row = self.horizontal_conv_merge(1 / 3 * input_row + 1 / 3 * input_row_down + 1 / 3 * input_row_down2)

        col = self.sigmoid(col) - 0.5
        row = self.sigmoid(row) - 0.5

        col = (
            col.view(
                (
                    col.size(0) // W,
                    W,
                )
                + col.size()[1:]
            )
            .permute(0, 4, 2, 3, 1)
            .contiguous()
        )
        row = (
            row.view(
                (
                    row.size(0) // H,
                    H,
                )
                + row.size()[1:]
            )
            .permute(0, 3, 2, 1, 4)
            .contiguous()
        )

        att = col * 0.5 + row * 0.5
        att = att.view((-1,) + att.size()[2:])

        return x + self.gamma * att * x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=False, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip
        return x


class Block_1(nn.Module):
    def __init__(
        self, in_filters, out_filters, reps, strides=1, start_with_relu=False, grow_first=True, relu=True, sim=False, hv=False
    ):
        super(Block_1, self).__init__()
        self.neg_sim = spatial_self_attention(in_filters)
        self.block = Block(in_filters, out_filters, reps, strides, start_with_relu, grow_first)
        self.relu = nn.ReLU(inplace=True)
        self.flag = relu
        self.hv_att = Horizontal_Vertical_att(in_filters)
        self.sim = sim
        self.hv = hv

    def forward(self, x):
        if self.flag:
            x = self.relu(x)
        if self.sim:
            x = self.neg_sim(x)
        if self.hv:
            x = self.hv_att(x)
        x = self.block(x)

        return x


class spatial_self_attention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(spatial_self_attention, self).__init__()
        self.q_conv = nn.Conv2d(in_channels, in_channels // ratio, 1, 1, bias=False)
        self.k_conv = nn.Conv2d(in_channels, in_channels // ratio, 1, 1, bias=False)
        self.v_conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # BT, C, H, W
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = q.view(
            (
                -1,
                num_frames,
            )
            + q.size()[1:]
        )  # B, T, C, H, W
        B, T, C, H, W = q.size()
        q = q.view(B, T, C, -1).permute(0, 3, 1, 2).contiguous()  # B, HW, T, C
        q = F.normalize(q, dim=-1)

        k = k.view(
            (
                -1,
                num_frames,
            )
            + k.size()[1:]
        )
        k = k.view(B, T, C, -1).permute(0, 3, 1, 2).contiguous()  # B, HW, T, C
        k = F.normalize(k, dim=-1)

        att = -torch.matmul(q, k.permute(0, 1, 3, 2))  # B, HW, T, T
        att = att.sum(-1)  # B, HW, T

        att_var = torch.var(att, dim=2)  # B, HW
        att_var_k = torch.topk(att_var, k=H, largest=True, sorted=True)[0]
        att_var_k = att_var_k[:, -1].unsqueeze(-1)  # B, 1
        att_mask = torch.le(att_var, att_var_k)  # B, HW
        att_mask = att_mask.expand(T, att_mask.size(0), att_mask.size(1)).permute(1, 2, 0).contiguous()
        att[att_mask] = -float(num_frames)  # B, HW, T

        temp_min = torch.min(att, dim=1, keepdim=True)[0]
        temp_max = torch.max(att, dim=1, keepdim=True)[0]
        att = (att - temp_min) / (temp_max - temp_min + 1e-8)

        att = att.permute(0, 2, 1).contiguous().unsqueeze(2)  # B, T, 1, HW
        att = att.view(B, T, 1, H, W)
        att = self.gamma * v * att.view(-1, 1, H, W)  # BT, C H, W
        x = x + att

        return x


class video_model(nn.Module):
    def __init__(self, num_classes=2):
        """Constructor
        Args:
            num_classes: number of classes
        """
        super(video_model, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block_1(64, 64, 2, 2, start_with_relu=False, grow_first=True, relu=False, sim=True, hv=True)
        self.block2 = Block_1(64, 128, 2, 2, start_with_relu=False, grow_first=True, sim=True, hv=True)
        self.block3 = Block_1(128, 128, 2, 2, start_with_relu=False, grow_first=True, sim=True, hv=True)

        self.conv3 = SeparableConv2d(128, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)

        # do relu here
        self.conv4 = SeparableConv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fea_fusion = FeatureFusionModule(128 * num_frames, 512)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = x.view(
            (
                -1,
                int(x.size()[1] // num_frames),
            )
            + x.size()[2:]
        )

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = x.view(x.size(0) // num_frames, -1, x.size(2), x.size(3))

        x = self.fea_fusion(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        fea = x.view(x.size(0), -1)

        return fea


"""
The following code is from https://github.com/yinql1995/Fine-grained-Multimodal-DeepFake-Classification/blob/main/network/video_processing_model.py
"""


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        in_channels=1,
        sample_rate=16000,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        mask=False,
    ):
        super(CONV, self).__init__()
        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x, mask=False):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = torch.Tensor(np.hamming(self.kernel_size)) * torch.Tensor(hideal)
        device = x.device
        band_pass_filter = self.band_pass.to(device)

        # Frequency masking: We randomly mask (1/5)th of no. of sinc filters channels (70)
        if mask == True:
            for i1 in range(1):
                A = np.random.uniform(0, 14)
                A = int(A)
                A0 = random.randint(0, band_pass_filter.shape[0] - A)
                band_pass_filter[A0 : A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=None, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
            self.conv1 = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1], kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.selu = nn.SELU(inplace=True)

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=nb_filts[1], kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1], out_channels=nb_filts[1], kernel_size=(2, 3), padding=(0, 1), stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0], out_channels=nb_filts[1], padding=(0, 1), kernel_size=(1, 3), stride=1
            )
        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x

        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
            out = self.conv1(out)
        else:
            x = x
            out = self.conv_1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class audio_model(nn.Module):
    def __init__(self, num_nodes=4):
        super(audio_model, self).__init__()
        '''
        Sinc conv. layer
        '''
        self.conv_time = CONV(out_channels=50, kernel_size=128, in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Note that here you can also use only one encoder to reduce the network parameters which is jsut half of the 0.44M (mentioned in the paper). I was doing some subband analysis and forget to remove the use of two encoders.  I also checked with one encoder and found same results.
        self.encoder1 = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=[32, 32], first=True)),
            nn.Sequential(Residual_block(nb_filts=[32, 32])),
            nn.Sequential(Residual_block(nb_filts=[32, 64])),
            nn.Sequential(Residual_block(nb_filts=[64, 64])),
            nn.Sequential(Residual_block(nb_filts=[64, 64])),
            nn.Sequential(Residual_block(nb_filts=[64, 32])),
        )

        self.num_nodes = num_nodes

    def forward(self, x, Freq_aug=False):
        """
        x= (#bs,samples)
        """
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)
        # Freq masking during training only

        if Freq_aug == True:
            x = self.conv_time(x, mask=True)  # (#bs,sinc_filt(70),64472)
        else:
            x = self.conv_time(x, mask=False)
        """
        Different with the our RawNet2 model, we interpret the output of sinc-convolution layer as 2-dimensional image with one channel (like 2-D representation).
        """
        x = x.unsqueeze(dim=1)  # 2-D (#bs,1,sinc-filt(70),64472)
        x = F.max_pool2d(torch.abs(x), (3, 3))  # [#bs, C(1),F(23),T(21490)]
        x = self.first_bn(x)
        x = self.selu(x)

        temp = torch.chunk(x, self.num_nodes, dim=3)
        out = []

        for i in range(self.num_nodes):
            # encoder structure for spectral GAT
            t = self.encoder1(temp[i])
            # max-pooling along time with absolute value  (Attention in spectral part)
            x_max, _ = torch.max(torch.abs(t), dim=3)  # [#bs, C(64), F(23)]
            out.append(x_max)
        out = torch.stack(out, dim=1)
        out = out.view(out.size(0), out.size(1), -1)

        return out

    def _make_layer(self, nb_blocks, nb_filts, first=False):
        layers = []

        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts=nb_filts, first=first))
            if i == 0:
                nb_filts[0] = nb_filts[1]

        return nn.Sequential(*layers)
