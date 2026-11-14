import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from einops import rearrange, repeat
from pytorchvideo.models.resnet import Net, create_bottleneck_block, create_res_stage
from pytorchvideo.models.stem import create_res_basic_stem
from torch import Tensor, einsum, nn
from typing import Callable, Iterable, Optional, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from detectors.abstract_detector import AbstractDetector
from tools.registry import DETECTOR, LOSSFUNC


@DETECTOR.register_module(module_name='FauForensics')
class FauForensicsDetector(AbstractDetector):
    """Audio, video and facial-action fusion with joint and unimodal supervision."""

    def __init__(self, config, num_layers_in_fc_layers=1024, dropout=0.5):
        super().__init__()
        self.config = config
        self.video_backbone, self.audio_backbone, self.fau_backbone = self.build_backbone(config)
        self.loss_func_dict, self.loss_weight_dict = self.build_loss(config['losses'])

        feat_dim = 512
        frame_num = 25

        # fau
        self.au_pool_norm = LayerNorm(feat_dim)

        # video
        self.v_pool = nn.Linear(2048, feat_dim, bias=False)
        self.v_pool_norm = LayerNorm(feat_dim)

        # query-shared self-attention
        q_num_queries = frame_num
        self.queries = nn.Parameter(torch.randn(q_num_queries, feat_dim))
        self.aud_attn_pool = CrossAttention(dim=feat_dim, context_dim=feat_dim, norm_context=True)
        self.aud_attn_pool_norm = LayerNorm(feat_dim)
        self.v_attn_pool = CrossAttention(dim=feat_dim, context_dim=feat_dim, norm_context=True)
        self.v_attn_pool_norm = LayerNorm(feat_dim)

        # MLP
        self.num_cls = config.get('num_classes', 2)
        self.frame_num = frame_num
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_v = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.mlp_joint = nn.Linear(frame_num**2, self.num_cls, bias=False)
        self.mlp_v = nn.Linear(frame_num**2, 2, bias=False)
        self.mlp_a = nn.Linear(frame_num**2, 2, bias=False)

    def build_backbone(self, config):
        """Load Whisper, CSN and MEFARG initializations; freeze MEFARG parameters."""
        params = ModelDimensions(
            **{
                'n_mels': 80,
                'n_vocab': 51865,
                'n_audio_ctx': 50,
                'n_audio_state': 512,
                'n_audio_head': 8,
                'n_audio_layer': 6,
                'n_text_ctx': 448,
                'n_text_state': 512,
                'n_text_head': 8,
                'n_text_layer': 6,
            }
        )
        audio_backbone = Whisper(params)
        state_dict = torch.load('pretrained/whisper/whisper_base.pt', map_location='cpu')['model_state_dict']
        del state_dict['encoder.positional_embedding']
        audio_backbone.load_state_dict(state_dict, strict=False)

        video_backbone = csn_temporal_no_head()
        state_dict = torch.load('pretrained/CSN/CSN_32x2_R101.pyth', map_location='cpu')['model_state']
        video_backbone.load_state_dict(state_dict, strict=False)

        fau_backbone = MEFARG(num_classes=12, backbone='resnet50', metric='dots')
        state_dict = torch.load('pretrained/MEFARG/au_112_12cls2.pth', map_location='cpu')['state_dict']
        fau_backbone.load_state_dict(state_dict, strict=False)
        for parameter in fau_backbone.parameters():
            parameter.requires_grad = False

        return video_backbone, audio_backbone, fau_backbone

    def build_loss(self, losses):
        """Construct named loss modules and return their modules and configured weights."""
        loss_func_dict, loss_weight_dict = {}, {}
        for loss in losses:
            loss_func_dict[loss['name']] = LOSSFUNC[loss['func']](**loss.get('params', {}))
            loss_weight_dict[loss['name']] = loss.get('weight', 1.0)

        return loss_func_dict, loss_weight_dict

    def v_att_p(self, au_tokens):
        """Pool visual/action tokens with learned shared queries."""
        au_queries = repeat(self.queries, 'n d -> b n d', b=au_tokens.shape[0])
        au_queries = self.v_attn_pool(au_queries, au_tokens)
        au_queries = self.v_attn_pool_norm(au_queries)
        return au_queries

    def a_att_p(self, audio_tokens):
        """Pool audio tokens with learned shared queries."""
        audio_queries = repeat(self.queries, 'n d -> b n d', b=audio_tokens.shape[0])
        audio_queries = self.aud_attn_pool(audio_queries, audio_tokens)
        audio_queries = self.aud_attn_pool_norm(audio_queries)
        return audio_queries

    def get_unimodal_ap(self, feat, logit_scale):
        """Flatten the normalized within-modality token similarity matrix."""
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat = logit_scale * torch.matmul(feat, feat.permute(0, 2, 1))
        feat = torch.nn.functional.normalize(feat)
        return feat.view(feat.size(0), -1)

    def get_TAPooler(self, visual_features, audio_features):
        """Return cross-modal, audio and visual token-pair similarity features."""
        v_feat_intra = self.get_unimodal_ap(visual_features, self.logit_scale_v.exp())
        a_feat_intra = self.get_unimodal_ap(audio_features, self.logit_scale_a.exp())
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        feat_inter = logit_scale * torch.matmul(visual_features, audio_features.permute(0, 2, 1))
        feat_inter = torch.nn.functional.normalize(feat_inter)
        feat_inter = feat_inter.view(feat_inter.size(0), -1)
        return feat_inter, a_feat_intra, v_feat_intra

    def features(self, data_dict):
        """Return token-pair features from 25 video frames and 80-bin audio spectrograms."""
        audio_feat = self.audio_backbone.embed_audio(data_dict['audio'][:, :100, :].permute(0, 2, 1))

        video_feat = self.video_backbone(data_dict['video'])
        video_feat = self.v_pool(video_feat.permute(0, 2, 1))
        video_feat = self.v_pool_norm(video_feat)

        B, C, T, H, W = data_dict['video'].size()
        video = data_dict['video'].permute(0, 2, 1, 3, 4)
        video = video.contiguous().view(-1, C, H, W)
        fau_feat = self.fau_backbone(video)
        fau_feat = fau_feat.mean(-1)
        fau_feat = fau_feat.view(B, T, -1)

        feat_v_p = self.v_att_p(fau_feat + video_feat)
        feat_a_p = self.a_att_p(audio_feat)
        feat_inter, a_feat_intra, v_feat_intra = self.get_TAPooler(feat_v_p, feat_a_p)
        return {'feat_inter': feat_inter, 'v_feat_intra': v_feat_intra, 'a_feat_intra': a_feat_intra}

    def classifier(self, features):
        """Return audio, visual and joint two-class logits from token-pair features."""
        pred_a = self.mlp_a(features['a_feat_intra'])
        pred_v = self.mlp_v(features['v_feat_intra'])
        pred_joint = self.mlp_joint(features['feat_inter'])
        return {'audio': pred_a, 'visual': pred_v, 'total': pred_joint}

    def get_losses(self, data_dict, pred_dict):
        """Combine weighted joint, audio and video cross-entropies.

        Uses ``label``, ``audio_label`` and ``video_label`` respectively.
        """
        ce_av_loss = self.loss_func_dict['CE_AV'](pred_dict['cls'], data_dict['label'])
        ce_av_weight = self.loss_weight_dict['CE_AV']
        ce_a_loss = self.loss_func_dict['CE_A'](pred_dict['a_logits'], data_dict['audio_label'])
        ce_a_weight = self.loss_weight_dict['CE_A']
        ce_v_loss = self.loss_func_dict['CE_V'](pred_dict['v_logits'], data_dict['video_label'])
        ce_v_weight = self.loss_weight_dict['CE_V']

        return {
            'overall': ce_av_loss * ce_av_weight + ce_a_loss * ce_a_weight + ce_v_loss * ce_v_weight,
            'CE_AV': ce_av_loss,
            'CE_A': ce_a_loss,
            'CE_V': ce_v_loss,
        }

    def forward(self, data_dict, inference=False):
        """Return joint ``cls``/``prob``, unimodal logits and token-pair feature tensors."""
        feats = self.features(data_dict)
        preds = self.classifier(feats)

        prob = torch.softmax(preds['total'], dim=1)[:, 1]
        return {
            'cls': preds['total'],
            'prob': prob,
            'a_logits': preds['audio'],
            'v_logits': preds['visual'],
            'av_logits': preds['total'],
            'fusion_feature': feats['feat_inter'],
            'audio_feature': feats['a_feat_intra'],
            'video_feature': feats['v_feat_intra'],
        }


"""
The following code is from https://github.com/JWang-cs21/FauForensics/blob/main/utils/models/backbone/WhisperAEncoder.py
"""


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        q = self.query(x)
        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        bs, t, c = x.size()
        x = x.view(bs, 25, -1, c)
        x = x.mean(2)
        return x


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)


"""
The following code is from https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/csn.py
"""


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def create_csn(
    *,
    # Input clip configs.
    input_channel: int = 3,
    # Model configs.
    # model_depth: int = 50,
    model_depth: int = 101,  ## changed by wj
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_out: int = 64,
    # stem_dim_out: int = 24,        ## changed by wj
    stem_conv_kernel_size: Tuple[int] = (3, 7, 7),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    stem_pool: Callable = None,
    stem_pool_kernel_size: Tuple[int] = (1, 3, 3),
    stem_pool_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_a_kernel_size: Tuple[int] = (1, 1, 1),
    stage_conv_b_kernel_size: Tuple[int] = (3, 3, 3),
    stage_conv_b_width_per_group: int = 1,
    stage_spatial_stride: Tuple[int] = (1, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 2, 2, 2),
    bottleneck: Callable = create_bottleneck_block,
    bottleneck_ratio: int = 4,
) -> nn.Module:

    torch._C._log_api_usage_once("PYTORCHVIDEO.model.create_csn")

    # Number of blocks for different stages given the model depth.
    _MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}

    assert model_depth in _MODEL_STAGE_DEPTH.keys(), f"{model_depth} is not in {_MODEL_STAGE_DEPTH.keys()}"
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]

    blocks = []
    stem = create_res_basic_stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        pool=stem_pool,
        pool_kernel_size=stem_pool_kernel_size,
        pool_stride=stem_pool_stride,
        pool_padding=[size // 2 for size in stem_pool_kernel_size],
        norm=norm,
        activation=activation,
    )
    blocks.append(stem)
    stage_dim_in = stem_dim_out
    stage_dim_out = stage_dim_in * 4
    for idx in range(len(stage_depths)):
        stage_dim_inner = stage_dim_out // bottleneck_ratio
        depth = stage_depths[idx]
        stage_conv_b_stride = (
            stage_temporal_stride[idx],
            stage_spatial_stride[idx],
            stage_spatial_stride[idx],
        )
        stage = create_res_stage(
            depth=depth,
            dim_in=stage_dim_in,
            dim_inner=stage_dim_inner,
            dim_out=stage_dim_out,
            bottleneck=bottleneck,
            conv_a_kernel_size=stage_conv_a_kernel_size,
            conv_a_stride=(1, 1, 1),
            conv_a_padding=[size // 2 for size in stage_conv_a_kernel_size],
            conv_b_kernel_size=stage_conv_b_kernel_size,
            conv_b_stride=stage_conv_b_stride,
            conv_b_padding=[size // 2 for size in stage_conv_b_kernel_size],
            conv_b_num_groups=(stage_dim_inner // stage_conv_b_width_per_group),
            conv_b_dilation=(1, 1, 1),
            norm=norm,
            activation=activation,
        )

        blocks.append(stage)
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

    blocks.append(LambdaModule(lambda x: x.mean(dim=(3, 4))))

    return Net(blocks=nn.ModuleList(blocks))


def csn_temporal_no_head(**kwargs):
    model = create_csn(stem_pool=nn.MaxPool3d, stage_temporal_stride=(1, 1, 1, 1), **kwargs)
    return model


"""
The following code is from https://github.com/JWang-cs21/FauForensics/tree/main/utils/models/backbone/MEFarg
"""


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2.0 / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x


models_dir = os.path.expanduser('pretrained')
# if not os.path.exists(models_dir):
#     # models_dir = os.path.expanduser('/Users/wangjian/project/FAU/ME-GraphAU-main/checkpoints')
#     # models_dir = os.path.expanduser('/opt/data/private/code/test/ME-GraphAU-main/checkpoints')
#     models_dir

model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)

        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet34'])))
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet50'])))
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet101'])))
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet152'])))
    return model


class MEFARG(nn.Module):
    def __init__(self, num_classes=8, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(MEFARG, self).__init__()
        if backbone == 'resnet18':
            self.backbone = resnet18()
        elif backbone == 'resnet101':
            self.backbone = resnet101()
        else:
            self.backbone = resnet50(pretrained=False)  # Use this
        self.in_channels = self.backbone.fc.weight.shape[1]
        self.out_channels = self.in_channels // 4
        self.backbone.fc = None

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_linear(x)

        return x.permute(0, 2, 1)


"""
The following code is from https://github.com/JWang-cs21/FauForensics/blob/main/utils/models/base.py
"""


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class CrossAttention(nn.Module):
    def __init__(self, dim, *, context_dim=None, dim_head=64, heads=8, parallel_ff=False, ff_mult=4, norm_context=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, 512, bias=False)

        # whether to have parallel feedforward
        ff_inner_dim = ff_mult * dim
        self.ff = (
            nn.Sequential(nn.Linear(dim, ff_inner_dim * 2, bias=False), SwiGLU(), nn.Linear(ff_inner_dim, dim, bias=False))
            if parallel_ff
            else None
        )

    def forward(self, x, context):
        x = self.norm(x)  # pre-layernorm, for queries and context
        context = self.context_norm(context)
        q = self.to_q(x)  # get queries
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale  # scale
        k, v = self.to_kv(context).chunk(2, dim=-1)  # get key / values
        sim = einsum('b h i d, b j d -> b h i j', q, k)  # query / key similarity

        # attention
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # add parallel feedforward (for multimodal layers)
        if exists(self.ff):
            out = out + self.ff(x)
        return out
