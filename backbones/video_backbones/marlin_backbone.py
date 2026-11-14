import math
import os
import sys
import torch
import warnings

from einops import rearrange
from einops.layers.torch import Rearrange
from functools import partial
from torch import Size, Tensor, nn
from torch.nn import functional as F
from typing import Callable, List, Optional, Sequence, Tuple, Union

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.registry import BACKBONE


Shape = Union[Size, List[int], Tuple[int, ...]]
ModuleFactory = Union[Callable[[], nn.Module], Callable[[int], nn.Module]]

"""
The following code is from https://github.com/ControlNet/MARLIN/blob/master/src/marlin_pytorch/model/positional_embedding.py
"""


class PositionalEmbedding(nn.Module):
    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5, trainable: bool = True):
        super().__init__()
        self.input_shape = input_shape
        self.emb = nn.Parameter(torch.zeros(1, *input_shape), requires_grad=trainable)
        self.use_dropout = dropout_rate is not None and dropout_rate != 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.emb
        if self.use_dropout:
            x = self.dropout(x)
        return x

    @property
    def trainable(self):
        return self.emb.requires_grad

    @trainable.setter
    def trainable(self, value: bool):
        self.emb.requires_grad = value


class SinCosPositionalEmbedding(PositionalEmbedding):
    def __init__(self, input_shape: Shape, dropout_rate: float = 0.5):
        super().__init__(input_shape, dropout_rate, trainable=False)
        self.emb.data = self.make_embedding().unsqueeze(0)

    def make_embedding(self) -> Tensor:
        n_position, d_hid = self.input_shape

        def get_position_angle_vec(position):
            return position / torch.tensor(10000).pow(2 * torch.div(torch.arange(d_hid), 2, rounding_mode='trunc') / d_hid)

        sinusoid_table = torch.stack([get_position_angle_vec(pos_i) for pos_i in range(n_position)], 0)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return sinusoid_table.float()


"""
The following code is from https://github.com/ControlNet/MARLIN/blob/master/src/marlin_pytorch/model/modules.py
"""


class PatchEmbedding3d(nn.Module):
    def __init__(
        self,
        input_size: Shape,
        patch_size: Union[int, Shape],
        embedding: int,
        strides: Optional[Union[int, Shape]] = None,
        build_normalization: Optional[ModuleFactory] = None,
    ):
        super().__init__()
        # channel, time, height, width
        c, t, h, w = input_size
        # patch_time, patch_height, patch_width
        pt, ph, pw = (patch_size, patch_size, patch_size) if type(patch_size) is int else patch_size

        # configure the strides for conv3d
        if strides is None:
            # no specified means no overlap and gap between patches
            strides = (pt, ph, pw)
        elif type(strides) is int:
            # transform the side length of strides to 3D
            strides = (strides, strides, strides)

        self.projection = nn.Conv3d(c, embedding, kernel_size=(pt, ph, pw), stride=strides)
        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
        self.rearrange = Rearrange("b d nt nh nw -> b (nt nh nw) d")

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        x = self.rearrange(x)
        if self.has_norm:
            x = self.normalization(x)
        return x


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        build_activation: Optional[ModuleFactory] = None,
        build_normalization: Optional[ModuleFactory] = None,
        normalization_after_activation: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        self.has_act = build_activation is not None
        if self.has_act:
            self.activation = build_activation()
        else:
            self.activation = None

        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
            self.norm_after_act = normalization_after_activation
        else:
            self.normalization = None

        self.has_dropout = dropout_rate > 0
        if self.has_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.has_act and self.has_norm:
            if self.norm_after_act:
                x = self.activation(x)
                x = self.normalization(x)
            else:
                x = self.normalization(x)
                x = self.activation(x)
        elif self.has_act and not self.has_norm:
            x = self.activation(x)
        elif not self.has_act and self.has_norm:
            x = self.normalization(x)

        if self.has_dropout:
            x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, neurons: Sequence[int], build_activation: Optional[ModuleFactory] = None, dropout_rate: float = 0.0):
        super().__init__()
        n_features = neurons[1:]
        self.layers: nn.ModuleList[Linear] = nn.ModuleList(
            [
                Linear(neurons[i], neurons[i + 1], True, build_activation, None, False, dropout_rate)
                for i in range(len(n_features) - 1)
            ]
            + [Linear(neurons[-2], neurons[-1], True)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(neurons=[dim, mlp_hidden_dim, dim], build_activation=act_layer, dropout_rate=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            x = x + (self.gamma_1 * self.attn(self.norm1(x)))
            x = x + (self.gamma_2 * self.mlp(self.norm2(x)))
        return x


"""
The following code is from https://github.com/ControlNet/MARLIN/blob/master/src/marlin_pytorch/model/encoder.py
"""


class MarlinEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        n_frames=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer="LayerNorm",
        init_values=0.0,
        tubelet_size=2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding3d(
            input_size=(3, n_frames, img_size, img_size), patch_size=(tubelet_size, patch_size, patch_size), embedding=embed_dim
        )
        num_patches = (img_size // patch_size) * (img_size // patch_size) * (n_frames // tubelet_size)

        # sine-cosine positional embeddings
        self.pos_embedding = SinCosPositionalEmbedding((num_patches, embed_dim), dropout_rate=0.0)

        if norm_layer == "LayerNorm":
            self.norm_layer = nn.LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError("Only LayerNorm is supported")

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=self.norm_layer,
                    init_values=init_values,
                )
                for _ in range(depth)
            ]
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # mask: (B, T, N) with boolean values, 0 -> masked, 1 -> visible
        assert len(x.shape) == 5, "x must be 5D"
        emb = self.patch_embedding(x)
        emb = self.pos_embedding(emb)
        b, _, c = emb.shape
        emb = emb[mask].view(b, -1, c)  # only visible patches are used
        emb = self.forward_features(emb)
        return emb

    def extract_features(self, x: Tensor, seq_mean_pool: bool) -> Tensor:
        x = self.patch_embedding(x)
        x = self.pos_embedding(x)
        for block in self.blocks:
            x = block(x)

        if seq_mean_pool:
            x = x.mean(dim=1)
        x = self.norm(x)
        return x


def no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


"""
The following code is from https://github.com/ControlNet/MARLIN/blob/master/src/marlin_pytorch/model/decoder.py
"""


class MarlinDecoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        n_frames=16,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer="LayerNorm",
        init_values=1.0,
        tubelet_size=2,
    ):
        super().__init__()
        output_dim = 3 * tubelet_size * patch_size * patch_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.n_patch_h = img_size // patch_size
        self.n_patch_w = img_size // patch_size
        self.embed_dim = embed_dim
        if norm_layer == "LayerNorm":
            self.norm_layer = nn.LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError("Only LayerNorm is supported")

        # sine-cosine positional embeddings
        self.pos_embedding = SinCosPositionalEmbedding(
            (self.n_patch_h * self.n_patch_w * (n_frames // tubelet_size), embed_dim), dropout_rate=0.0
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=self.norm_layer,
                    init_values=init_values,
                )
                for _ in range(depth)
            ]
        )

        self.head = Linear(embed_dim, output_dim)
        self.apply(self._init_weights)
        no_grad_trunc_normal_(self.mask_token, mean=0.0, std=0.02, a=-0.02, b=0.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatch_to_img(self, x: Tensor) -> Tensor:
        # x: (Batch, No. batches, Prod of cube size * C)
        x = rearrange(x, "b n (c p) -> b n p c", c=3)
        # x: (Batch, No. batches, Prod of cube size, C)
        x = rearrange(
            x,
            "b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)",
            p0=self.tubelet_size,
            p1=self.patch_size,
            p2=self.patch_size,
            h=self.n_patch_h,
            w=self.n_patch_w,
        )
        # x: (B, C, T, H, W)
        return x

    def forward_features(self, x, return_token_num=0):
        for block in self.blocks:
            x = block(x)

        if return_token_num > 0:
            x = x[:, -return_token_num:]

        x = self.norm(x)
        x = self.head(x)
        # x: (B, N_mask, C)
        return x

    def forward(self, x, mask):
        # mask: 0 -> masked, 1 -> visible
        b, n, c = x.shape
        expand_pos_embed = self.pos_embedding.emb.data.expand(b, -1, -1)
        pos_emb_vis = expand_pos_embed[mask].view(b, -1, c)
        pos_emb_mask = expand_pos_embed[~mask].view(b, -1, c)
        x = torch.cat([x + pos_emb_vis, self.mask_token + pos_emb_mask], dim=1)

        mask_num = pos_emb_mask.shape[1]

        x = self.forward_features(x, return_token_num=mask_num)
        return x


@BACKBONE.register_module(module_name='MARLIN')
class MARLINBackbone(nn.Module):
    """Registered MARLIN video encoder with an optional reconstruction decoder."""

    predefined_versions = {
        'marlin_vit_small': {
            'img_size': 224,
            'patch_size': 16,
            'n_frames': 16,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'qk_scale': None,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'norm_layer': 'LayerNorm',
            'init_values': 0.0,
            'tubelet_size': 2,
            'encoder_embed_dim': 384,
            'encoder_depth': 12,
            'encoder_num_heads': 6,
            'decoder_embed_dim': 192,
            'decoder_depth': 4,
            'decoder_num_heads': 3,
        },
        'marlin_vit_base': {
            'img_size': 224,
            'patch_size': 16,
            'n_frames': 16,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'qk_scale': None,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'norm_layer': 'LayerNorm',
            'init_values': 0.0,
            'tubelet_size': 2,
            'encoder_embed_dim': 768,
            'encoder_depth': 12,
            'encoder_num_heads': 12,
            'decoder_embed_dim': 384,
            'decoder_depth': 4,
            'decoder_num_heads': 6,
        },
        'marlin_vit_large': {
            'img_size': 224,
            'patch_size': 16,
            'n_frames': 16,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'qk_scale': None,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'norm_layer': 'LayerNorm',
            'init_values': 0.0,
            'tubelet_size': 2,
            'encoder_embed_dim': 1024,
            'encoder_depth': 24,
            'encoder_num_heads': 16,
            'decoder_embed_dim': 512,
            'decoder_depth': 12,
            'decoder_num_heads': 8,
        },
    }

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        n_frames: int,
        encoder_embed_dim: int,
        encoder_depth: int,
        encoder_num_heads: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: Optional[float],
        drop_rate: float,
        attn_drop_rate: float,
        norm_layer: str,
        init_values: float,
        tubelet_size: int,
        as_feature_extractor: bool = True,
    ):
        """
        MARLIN backbone structure, referred from https://github.com/ControlNet/MARLIN/blob/master/src/marlin_pytorch/model/marlin.py
        Args:
            img_size: (int) input images size
            patch_size: (int) patch size
            n_frames: (int) the number of input frames
            encoder_embed_dim: (int) embedding dimension of encoder
            encoder_depth: (int) the number of blocks in encoder
            encoder_num_heads: (int) the number of heads in encoder attention
            decoder_embed_dim: (int) embedding dimension of decoder
            decoder_depth: (int) the number of blocks in decoder
            decoder_num_heads: (int) the number of heads in decoder attention
            mlp_ratio: (float) the ratio of dimension expansion in MLP, used in attention blocks
            qkv_bias: (bool) whether to use qkv bias
            qk_scale: (int) qk scaler if provided
            drop_rate: (float) dropout rate of feed forward layer
            attn_drop_rate: (float) dropout rate of self-attention layer
            norm_layer: (nn.Module) normalization layer type
            init_values: (float) if larger than 0, use learnable scaler between shortcut connection in attention block
            tubelet_size: (int) temporal patch size
            as_feature_extractor: (bool) whether to instantiate as a feature extract, i.e., only containing encoder
        """
        super().__init__()
        self.encoder = MarlinEncoder(
            img_size=img_size,
            patch_size=patch_size,
            n_frames=n_frames,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
        )
        self.as_feature_extractor = as_feature_extractor
        self.clip_frames = n_frames
        if as_feature_extractor:
            self.enc_dec_proj = None
            self.decoder = None
        else:
            self.decoder = MarlinDecoder(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=decoder_embed_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                tubelet_size=tubelet_size,
            )
            self.enc_dec_proj = Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

    @property
    def device(self):
        """Return the device of the encoder normalization parameters."""
        return self.encoder.norm.weight.device

    def extract_features(self, x: Tensor, keep_seq: bool = True):
        """Return clip token features, or their mean when ``keep_seq=False``.

        Input is ``(B, C, T, H, W)``. Evaluation mode disables gradient recording.
        """
        if self.training:
            return self.encoder.extract_features(x, seq_mean_pool=not keep_seq)
        else:
            with torch.no_grad():
                return self.encoder.extract_features(x, seq_mean_pool=not keep_seq)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Reconstruct masked video patches in decoder mode.

        Requires ``as_feature_extractor=False`` and a supplied boolean patch mask.
        """
        if self.as_feature_extractor:
            raise RuntimeError("For feature extraction, please use `extract_features` or `extract_video`.")
        else:
            assert mask is not None
            x = self.encoder(x, mask)
            x = self.enc_dec_proj(x)
            x = self.decoder(x, mask)
        return x
