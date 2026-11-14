import copy
import logging
import math
import os
import sys
import torch
import torch.nn as nn

from einops import rearrange, repeat
from timm import create_model
from timm.models.layers import DropPath, Mlp

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from detectors.abstract_detector import AbstractDetector
from tools.registry import DETECTOR, LOSSFUNC


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='FRADE')
class FRADEDetector(AbstractDetector):
    """ViT adapter fusion with classification, contrastive and center-cluster losses."""

    def __init__(self, config):
        super().__init__()
        self.backbone = self.build_backbone(config)
        self.loss_func_dict, self.loss_weight_dict = self.build_loss(config['losses'])

    def build_backbone(self, config):
        """Construct the ViT adapter head with its fixed relative initialization path."""
        return AudioVisual_Transformer_with_Adapter_Head()

    def build_loss(self, losses):
        """Construct named loss modules and return their modules and configured weights."""
        loss_func_dict, loss_weight_dict = {}, {}
        for loss in losses:
            loss_func_dict[loss['name']] = LOSSFUNC[loss['func']](**loss.get('params', {}))
            loss_weight_dict[loss['name']] = loss.get('weight', 1.0)

        return loss_func_dict, loss_weight_dict

    def features(self, data_dict: dict):
        """Return logits and fusion/video/audio features from the adapter head.

        Takes the first 64 audio steps, repeats them across three channels and
        reorders video axes, replacing both tensors in ``data_dict`` in place.
        """
        data_dict['audio'] = data_dict['audio'][:, :64, :].unsqueeze(1).repeat(1, 3, 1, 1)
        data_dict['video'] = data_dict['video'].permute(0, 2, 1, 3, 4)

        return self.backbone(data_dict['audio'], data_dict['video'])

    def classifier(self, features):
        """Unused interface hook; ``features`` includes classification."""
        pass

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """Combine weighted CE, cross-modal contrastive and center-cluster losses."""
        loss1 = self.loss_func_dict['CE'](pred_dict['cls'], data_dict['label'])
        lambda1 = self.loss_weight_dict['CE']
        loss2 = self.loss_func_dict['CMC'](pred_dict['video_feature'], pred_dict['audio_feature'], data_dict['label'])
        lambda2 = self.loss_weight_dict['CMC']
        loss3 = self.loss_func_dict['CC'](pred_dict['fusion_feature'], data_dict['label'])
        lambda3 = self.loss_weight_dict['CC']

        return {'overall': loss1 * lambda1 + loss2 * lambda2 + loss3 * lambda3, 'CE': loss1, 'CMC': loss2, 'CC': loss3}

    def forward(self, data_dict, inference=False):
        """Return two-class logits, fake probability and audio/video/fusion features."""
        pred, fusion_out, fv_head, fa_head = self.features(data_dict)
        prob = torch.softmax(pred, dim=1)[:, 1]

        pred_dict = {'cls': pred, 'prob': prob, 'fusion_feature': fusion_out, 'audio_feature': fa_head, 'video_feature': fv_head}

        return pred_dict


"""
The following code is from https://github.com/niefengxxx/FRADE/blob/main/frade/frade_model.py
"""


class Attention_qkv_Bias(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, bias=None, pad=False, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        if bias is not None:
            qbias, kbias, vbias = bias

        padded = torch.zeros([B, self.num_heads, 1, C // self.num_heads], device=x.device)

        if pad is True:
            q = q + torch.cat([padded, qbias], dim=2)
            k = k + torch.cat([padded, kbias], dim=2)
            v = v + torch.cat([padded, vbias], dim=2)
        elif bias is not None:
            q = q + qbias
            k = k + kbias
            v = v + vbias

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn is True:
            return x, attn
        else:
            return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block_qkv_Bias(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_qkv_Bias(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, bias=None, pad=False, return_attn=False):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), bias, pad, return_attn)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class QKV_Adapter_ViT_spatio_filter(nn.Module):
    def __init__(self, input_dims, nums_head=12, reduction=2):  # 768, 12, 4
        super().__init__()

        self.input_dims = input_dims
        self.nums_head = nums_head

        lpf_mask = torch.zeros((14, 14))

        for x in range(14):
            for y in range(14):
                if ((x - (14 - 1) / 2) ** 2 + (y - (14 - 1) / 2) ** 2) < 3:
                    lpf_mask[y, x] = 1

        self.mask = lpf_mask

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dims, out_channels=input_dims // reduction, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels=input_dims // reduction, out_channels=input_dims, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

        self.q = nn.Conv2d(in_channels=input_dims, out_channels=input_dims, kernel_size=1, bias=False)
        self.k = nn.Conv2d(in_channels=input_dims, out_channels=input_dims, kernel_size=1, bias=False)
        self.v = nn.Conv2d(in_channels=input_dims, out_channels=input_dims, kernel_size=1, bias=False)

    def filter(self, x, H, W):
        # print(x.shape, x.dtype)
        out = torch.fft.fftn(x, dim=(2, 3))
        # print(out.shape, out.dtype)
        out = torch.roll(out, (H // 2, W // 2), dims=(2, 3))
        # print(out.shape, out.dtype)
        out_low = out * self.mask.to(x.device)
        # print(out_low.shape, out_low.dtype)
        # if not torch.isfinite(out_low).all():
        #     print("NaNs/Infs detected in `out_low`:", out_low)
        # print(torch.fft.ifftn(out_low, dim= (2,3)).shape, torch.fft.ifftn(out_low, dim= (2,3)).dtype)
        # out = torch.fft.ifftn(out_low, dim= (2,3))
        # out = out.to('cpu')
        # out = torch.abs(out)
        # out = out.cuda()
        out = torch.abs(torch.fft.ifftn(out_low, dim=(2, 3)))

        return out

    def forward(self, x):
        batchsize, n, dims = x.shape  # [16B, 197, 768]

        H = W = int(math.sqrt(n - 1))  # 14
        images = x[:, 1:, :].view(batchsize, H, W, -1).permute(0, 3, 1, 2)  # [16B, 768, 14, 14]

        out = self.conv(images)

        out = self.filter(out, H, W)

        # [16B, 768, 14, 14] -> [16B, 768, 196] -> [16B, 196, 768] -> [16B, 196, 12, 64] -> [16B, 12, 196, 64]
        q = (
            self.q(out)
            .view(batchsize, self.input_dims, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(batchsize, n - 1, self.nums_head, dims // self.nums_head)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(out)
            .view(batchsize, self.input_dims, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(batchsize, n - 1, self.nums_head, dims // self.nums_head)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(out)
            .view(batchsize, self.input_dims, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(batchsize, n - 1, self.nums_head, dims // self.nums_head)
            .permute(0, 2, 1, 3)
        )

        return [q, k, v]


# AFI - Audio
class QKV_Adapter_ViT_audio_filter(nn.Module):
    def __init__(self, input_dims, nums_head=12, reduction=2):
        super().__init__()

        self.input_dims = input_dims
        self.nums_head = nums_head

        lpf_mask = torch.zeros((4, 5))

        for x in range(5):
            for y in range(4):
                if ((x - (5 - 1) / 2) ** 2 + (y - (4 - 1) / 2) ** 2) < 1:
                    lpf_mask[y, x] = 1

        self.mask = lpf_mask

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dims, out_channels=input_dims // reduction, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels=input_dims // reduction, out_channels=input_dims, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

        self.q = nn.Conv2d(in_channels=input_dims, out_channels=input_dims, kernel_size=1, bias=False)
        self.k = nn.Conv2d(in_channels=input_dims, out_channels=input_dims, kernel_size=1, bias=False)
        self.v = nn.Conv2d(in_channels=input_dims, out_channels=input_dims, kernel_size=1, bias=False)

    def filter(self, x, H, W):
        out = torch.fft.fftn(x, dim=(2, 3))
        out = torch.roll(out, (H // 2, W // 2), dims=(2, 3))
        out_low = out * self.mask.to(x.device)
        out = torch.abs(torch.fft.ifftn(out_low, dim=(2, 3)))
        # out = torch.fft.ifftn(out_low, dim= (2,3))
        # out = out.to('cpu')
        # out = torch.abs(out)
        # out = out.cuda()

        return out

    def forward(self, x, h):
        batchsize, n, dims = x.shape

        H = h // 16
        W = (n - 1) // H
        images = x[:, 1:, :].view(batchsize, H, W, -1).permute(0, 3, 1, 2)

        out = self.conv(images)
        # print(out.shape)
        out = self.filter(out, H, W)

        q = (
            self.q(out)
            .view(batchsize, self.input_dims, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(batchsize, n - 1, self.nums_head, dims // self.nums_head)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(out)
            .view(batchsize, self.input_dims, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(batchsize, n - 1, self.nums_head, dims // self.nums_head)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(out)
            .view(batchsize, self.input_dims, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(batchsize, n - 1, self.nums_head, dims // self.nums_head)
            .permute(0, 2, 1, 3)
        )

        return [q, k, v]


class Audio_Queried_CrossModal_Inconsistency_Enhancer(nn.Module):
    def __init__(self, num_tokens=4, input_dims=768, reduction=4):
        super().__init__()

        self.input_dims = input_dims
        self.num_tokens = num_tokens

        self.cross_token = nn.Parameter(torch.randn(num_tokens, input_dims))  # [4, 768]

        self.proj_head = nn.Sequential(
            nn.Linear(input_dims, input_dims // reduction), nn.GELU(), nn.Linear(input_dims // reduction, input_dims)
        )
        self.norm = nn.LayerNorm(input_dims)

    def forward(self, audio, video):  # audio: [b, ma, dims]; video: [b*mv, t, dims]
        batchsize, nums_audio_with_head, _ = audio.shape  # [B, 4*5+1, 768]
        batch_and_mv, frames, _ = video.shape  # [197B, 16, 768]

        nums_video_with_head = batch_and_mv // batchsize  # 197

        video_nohead = video.view(batchsize, nums_video_with_head, frames, -1)[:, 1:].clone()  # [B, 196, 16, 768]
        video_nohead = video_nohead.view(-1, frames, self.input_dims)  # [b*(mv-1), t, dims]     # [196B, 16, 768]

        audio_nohead = audio[:, 1:]  # [b, ma-1, dims]    # [B, 20, 768]
        audio_query = repeat(audio_nohead, 't m d -> (t k) m d', k=nums_video_with_head - 1)  # [196B, 20, 768]

        rep_token = repeat(self.cross_token, 't d -> b t d', b=(nums_video_with_head - 1) * batchsize)  # [196B, 4, 768]
        cross_attn = torch.bmm(rep_token, audio_query.permute(0, 2, 1)).softmax(dim=-1)  # [196B, 4, 20]
        rep_token_res = torch.bmm(cross_attn, audio_query)  # [196B, 4, 768]

        rep_token = rep_token + rep_token_res

        video_attn = torch.bmm(video_nohead, rep_token.permute(0, 2, 1)).softmax(dim=-1)  # [196B, 16, 4]
        video_res = torch.bmm(video_attn, rep_token)  # [196B, 16, 768]

        video_res = video_res.view(batchsize, nums_video_with_head - 1, frames, self.input_dims)  # [B, 196, 16, 768]
        padded = torch.zeros([batchsize, 1, frames, self.input_dims], device=audio.device)  # [B, 1, 16, 768]

        out = torch.cat([padded, video_res], dim=1).view(batch_and_mv, frames, -1)  # [197B, 16, 768]

        out = torch.add(out, video)
        out = self.norm(out)
        out = self.proj_head(out)

        return out


class AudioVisual_Transformer_with_Adapter_Head(nn.Module):
    def __init__(self, num_tokens=4, input_dims=768, use_bn=False, reduction=4, frames=16):
        super().__init__()

        self.input_dims = input_dims
        self.frames = frames
        # timm.models.set_download_progress(True)
        vit_base = create_model(
            'vit_base_patch16_224',
            pretrained=False,
            checkpoint_path='pretrained/jx_vit_base_p16_224-80ecf9dd.pth',
            dynamic_img_size=True,
        )
        self.vit_base = copy.deepcopy(vit_base)

        for i in range(len(vit_base.blocks)):
            vit_block = self.vit_base.blocks[i]
            my_block = Block_qkv_Bias(dim=input_dims, num_heads=12, qkv_bias=True)
            my_block.load_state_dict(vit_block.state_dict(), strict=True)
            self.vit_base.blocks[i] = my_block

        for param in self.vit_base.parameters():
            param.require_grad = False

        hidden_list = []
        for idx, blk in enumerate(self.vit_base.blocks):
            hidden_d_size = blk.mlp.fc1.in_features
            hidden_list.append(hidden_d_size)

        self.ViT_Video_Spatio_Attn_Adapter = nn.ModuleList(
            [
                QKV_Adapter_ViT_spatio_filter(input_dims=input_dims, nums_head=12, reduction=reduction)
                for i in range(len(vit_base.blocks))
            ]
        )

        self.Video_ffn_Adapter = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dims, input_dims // reduction), nn.GELU(), nn.Linear(input_dims // reduction, input_dims)
                )
                for i in range(len(vit_base.blocks))
            ]
        )

        self.ViT_Audio_Attn_Adapter = nn.ModuleList(
            [
                QKV_Adapter_ViT_audio_filter(input_dims=input_dims, nums_head=12, reduction=reduction)
                for i in range(len(vit_base.blocks))
            ]
        )

        self.Audio_ffn_adapter = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dims, input_dims // reduction), nn.GELU(), nn.Linear(input_dims // reduction, input_dims)
                )
                for i in range(len(vit_base.blocks))
            ]
        )

        self.query_enhancer = nn.ModuleList(
            [
                Audio_Queried_CrossModal_Inconsistency_Enhancer(num_tokens, input_dims, reduction)
                for i in range(len(vit_base.blocks))
            ]
        )

        self.norm_video = nn.LayerNorm(input_dims)
        self.norm_audio = nn.LayerNorm(input_dims)

        self.fusion_proj = nn.Linear(input_dims * 2, input_dims)
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(input_dims, 2))

    def forward_patchEmbed(self, x):
        x = self.vit_base.patch_embed(x)  # [B, 197, 768]
        x = self.vit_base._pos_embed(x)
        x = self.vit_base.patch_drop(x)
        x = self.vit_base.norm_pre(x)

        return x

    def forward(self, audio, video):  # [B, 3, 64, 80], [B, 16, 3, 224, 224]
        b, c, h, w = audio.shape
        with torch.no_grad():
            fa = self.forward_patchEmbed(audio)
            fv = self.forward_patchEmbed(rearrange(video, 'b t c w h -> (b t) c w h'))
        b, ma, dims = fa.shape  # [B, 4*5+1, 768]
        bt, mv, dims = fv.shape  # [16B, 14*14+1, 768]

        for idx, blk in enumerate(self.vit_base.blocks):
            spatio_attn = self.ViT_Video_Spatio_Attn_Adapter[idx](fv)
            fv_res = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(fv), spatio_attn, True)))
            fv = fv + fv_res

            audio_attn = self.ViT_Audio_Attn_Adapter[idx](fa, h)
            fa_res = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(fa), audio_attn, True)))
            fa = fa + fa_res

            # [16B, 197, 768] -> [B, 16, 197, 768] -> [B, 197, 16, 768] -> [197B, 16, 768]
            fv_temporal = (
                fv.view(b, self.frames, mv, dims).permute(0, 2, 1, 3).contiguous().view(-1, self.frames, dims)
            )  # [b*mv, t, dims]
            fv_cross_audio = self.query_enhancer[idx](fa, fv_temporal)
            fv_temporal = fv_temporal + fv_cross_audio
            fv = fv_temporal.view(b, mv, self.frames, dims).permute(0, 2, 1, 3).contiguous().view(-1, mv, dims)

            fv_norm = blk.norm2(fv)
            fv = fv + blk.drop_path2(blk.ls2(blk.mlp(fv_norm))) + self.Video_ffn_Adapter[idx](fv_norm)

            fa_norm = blk.norm2(fa)
            fa = fa + blk.drop_path2(blk.ls2(blk.mlp(fa_norm))) + self.Audio_ffn_adapter[idx](fa_norm)

        fv = self.norm_video(fv)  # [b*t, nums, dims]
        fa = self.norm_audio(fa)  # [b, nums, dims]

        fv_head = fv[:, 0].view(b, self.frames, dims).mean(dim=1)
        fa_head = fa[:, 0]

        out = torch.cat([fv_head, fa_head], dim=-1)
        fusion_out = self.fusion_proj(out)
        logits = self.fc(fusion_out)
        return logits, fusion_out, fv_head, fa_head
