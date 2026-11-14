import einops
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backbones.video_backbones.marlin_backbone import MLP
from detectors.abstract_alternative_detector import AbstractAlternativeDetector
from detectors.abstract_detector import AbstractDetector
from tools.registry import BACKBONE, DETECTOR, LOSSFUNC


@DETECTOR.register_module(module_name='AVFF-Pretrain-Adv')
class AVFFPretrainAdv(AbstractAlternativeDetector):
    """AVFF reconstruction pretraining with generator/discriminator parameter groups."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.video_backbone, self.audio_backbone = self.build_backbone(config['backbone'])
        self.loss_func_dict, self.loss_weight_dict = self.build_loss(config['losses'])

        self.a2v = A2VNetwork(audio_dim=64 * 8 // 2, visual_dim=196 * 8 // 2, embed_dim=768, num_heads=12, batch_first=True)
        self.v2a = V2ANetwork(audio_dim=64 * 8 // 2, visual_dim=196 * 8 // 2, embed_dim=768, num_heads=12, batch_first=True)

        self.audio_discriminator = MLP([16 * 16, 512, 1], build_activation=nn.LeakyReLU)
        self.video_discriminator = MLP([2 * 16 * 16 * 3, 2048, 1], build_activation=nn.LeakyReLU)

        self.stage_parameter_dict = {
            'generator': [self.video_backbone, self.audio_backbone, self.a2v, self.v2a],
            'discriminator': [self.audio_discriminator, self.video_discriminator],
        }

    @staticmethod
    def _load_backbone_weights(backbone, weight_path):
        """
        Load raw backbone weights or a model/state_dict checkpoint container.
        Args:
            backbone (nn.Module): Encoder/decoder module to initialize.
            weight_path (str): Trusted initialization checkpoint selected by configuration.
        Returns:
            incompatible_keys: Missing/unexpected keys reported by PyTorch's non-strict loader.
                Shape mismatches still raise; parameter prefixes are not guessed or rewritten.
        """
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        if not isinstance(state_dict, dict) or not set(backbone.state_dict()).intersection(state_dict):
            raise ValueError(f'No matching backbone parameters found in {weight_path}. Check the checkpoint format.')
        return backbone.load_state_dict(state_dict, strict=False)

    def build_backbone(self, backbone_config):
        """Build reconstruction-capable MARLIN and AudioMAE with optional weights."""
        video_backbone_config = backbone_config['video']
        name = video_backbone_config['name']
        version = video_backbone_config.get('version')
        params = {'as_feature_extractor': False}
        if version in BACKBONE[name].predefined_versions:
            params.update(BACKBONE[name].predefined_versions[version])
        params.update(video_backbone_config.get('params', {}))
        video_backbone = BACKBONE[name](**params)
        if video_backbone_config.get('weight'):
            self._load_backbone_weights(video_backbone, video_backbone_config['weight'])

        audio_backbone_config = backbone_config['audio']
        name = audio_backbone_config['name']
        version = audio_backbone_config.get('version')
        params = {'in_chans': 1, 'audio_exp': True, 'img_size': (1024, 128)}
        if version in BACKBONE[name].predefined_versions:
            params.update(BACKBONE[name].predefined_versions[version])
        params.update(audio_backbone_config.get('params', {}))
        audio_backbone = BACKBONE[name](**params)
        if audio_backbone_config.get('weight'):
            self._load_backbone_weights(audio_backbone, audio_backbone_config['weight'])

        return video_backbone, audio_backbone

    def get_stage_parameters(self, stage_name):
        """Return parameters for ``generator`` or ``discriminator`` optimization."""
        params = []
        for module in self.stage_parameter_dict[stage_name]:
            params.extend(list(module.parameters()))

        return params

    def build_loss(self, losses):
        """Construct named loss modules and return their modules and configured weights."""
        loss_func_dict, loss_weight_dict = {}, {}
        for loss in losses:
            loss_func_dict[loss['name']] = LOSSFUNC[loss['func']](**loss.get('params', {}))
            loss_weight_dict[loss['name']] = loss.get('weight', 1.0)

        return loss_func_dict, loss_weight_dict

    def features(self, data_dict):
        """Return complementary reconstructions without modifying the input batch.

        Audio is padded/truncated locally to 1024 steps. Encoded audio patches
        are restored to temporal order before complementary masks are applied.
        """
        embed_dim = 768
        target_Ta = 1024
        P = 16
        audio = data_dict['audio']
        B, Ta, F = audio.shape
        _, C, Tv, H, W = data_dict['video'].shape

        # complementary mask
        audio_mask = torch.ones((B, 8), dtype=torch.bool, device=audio.device)
        video_mask = torch.zeros((B, 8), dtype=torch.bool, device=data_dict['video'].device)
        idxs = random.sample([i for i in range(8)], 4)
        for idx in idxs:
            audio_mask[:, idx] = False
            video_mask[:, idx] = True

        # AudioMAE encoding
        if Ta > target_Ta:
            audio = audio[:, :target_Ta, :]
        elif Ta < target_Ta:
            # Pad 0 here. But data_dict['audio'] is retrieved after normalization. To pad 0, we need pad (0 - mean) / std
            zero_pad = (
                torch.ones((B, target_Ta - Ta, 128), device=audio.device, dtype=audio.dtype)
                * 4.2677393
                / 9.1379948
            )
            audio = torch.cat([audio, zero_pad], dim=1)
        Ta = target_Ta
        num_audio_patches = Ta // P * F // P
        audio = audio.unsqueeze(1)
        audio_feat, _, ids_restore, _ = self.audio_backbone.forward_encoder(audio, mask_ratio=0)
        audio_embed = audio_feat[:, 1:, :]  # remove cls token
        # AudioMAE also shuffles at mask_ratio=0; restore time before grouping.
        audio_embed = torch.gather(audio_embed, 1, ids_restore.unsqueeze(-1).expand(-1, -1, embed_dim))
        audio_embed = audio_embed.reshape(B, 8, -1, embed_dim)
        audio_embed_vis = audio_embed[audio_mask].reshape(B, -1, num_audio_patches // 8, embed_dim)  # [B, 4, 64, 768]
        # print('audio feature', audio_embed.shape, audio_embed_vis.shape)

        # MARLIN encoding
        num_video_patches = Tv // 2 * H // P * W // P
        video_embed = self.video_backbone.encoder.extract_features(data_dict['video'], seq_mean_pool=False)  # [B, 1568, 768]
        video_embed = video_embed.reshape(B, 8, -1, embed_dim)
        video_embed_vis = video_embed[video_mask].reshape(B, -1, num_video_patches // 8, embed_dim)  # [B, 4, 196, 768]
        # print('video feature', video_embed.shape, video_embed_vis.shape)

        # feature fusion
        v2a_embed = self.v2a(video_embed_vis)  # [B, 8, 64, 768]
        audio_fused_embed = torch.zeros_like(audio_embed)
        audio_fused_embed[audio_mask] = audio_embed_vis.reshape(-1, num_audio_patches // 8, embed_dim)
        audio_fused_embed[~audio_mask] = v2a_embed.reshape(-1, num_audio_patches // 8, embed_dim)
        # print('audio fusion', audio_fused_embed.shape)

        a2v_embed = self.a2v(audio_embed_vis)  # [B, 8, 196, 768]
        video_fused_embed = torch.zeros_like(video_embed)
        video_fused_embed[video_mask] = video_embed_vis.reshape(-1, num_video_patches // 8, embed_dim)
        video_fused_embed[~video_mask] = a2v_embed.reshape(-1, num_video_patches // 8, embed_dim)
        # print('video fusion', video_fused_embed.shape)

        # AudioMAE decoding, AudioMAE forward_decoder() requires L + 1 token input.
        decoder_order = torch.arange(num_audio_patches, device=ids_restore.device).expand(B, -1)
        audio_pred, _, _ = self.audio_backbone.forward_decoder(
            torch.cat([audio_feat[:, 0:1, :], audio_fused_embed.reshape(B, num_audio_patches, embed_dim)], dim=1), decoder_order
        )  # [B, 512, 16 * 16 * 1]
        # print('audio pred', audio_pred.shape)

        # MARLIN decoding
        video_pred = self.video_backbone.enc_dec_proj(video_fused_embed.reshape(B, num_video_patches, embed_dim))
        video_pred = self.video_backbone.decoder(
            video_pred, torch.ones(B, num_video_patches, device=video_pred.device, dtype=torch.bool)
        )  # [B, 1568, 2 * 16 * 16 * 3]
        # print('video pred', video_pred.shape)

        # features for loss calculation
        audio_mean_embed = audio_embed.reshape(B, -1, embed_dim).mean(dim=1)
        video_mean_embed = video_embed.reshape(B, -1, embed_dim).mean(dim=1)

        audio_patch_dim = P * P
        audio_patches = audio.reshape(B, 1, Ta // P, P, F // P, P)
        audio_patches = torch.einsum('nchpwq->nhwpqc', audio_patches)  # [B，64, 8, 16, 16, 1]
        audio_patches = audio_patches.reshape(B, Ta // P * F // P, audio_patch_dim)
        audio_gt_masked = audio_patches.reshape(B, 8, -1, audio_patch_dim)[~audio_mask].reshape(B, -1, audio_patch_dim)
        audio_pred_masked = audio_pred.reshape(B, 8, -1, audio_patch_dim)[~audio_mask].reshape(B, -1, audio_patch_dim)
        audio_pred_unmasked = audio_pred.reshape(B, 8, -1, audio_patch_dim)[audio_mask].reshape(B, -1, audio_patch_dim)

        video_patch_dim = C * 2 * P * P
        video_patches = data_dict['video'].reshape(B, C, Tv // 2, 2, H // P, 16, W // P, P)
        video_patches = torch.einsum('nctkhpwq->nthwkpqc', video_patches)  # [B, 8, 14, 14, 2, 16, 16, 3]
        video_gt_masked = video_patches.reshape(B, 8, -1, video_patch_dim)[~video_mask].reshape(B, -1, video_patch_dim)
        video_pred_masked = video_pred.reshape(B, 8, -1, video_patch_dim)[~video_mask].reshape(B, -1, video_patch_dim)
        video_pred_unmasked = video_pred.reshape(B, 8, -1, video_patch_dim)[video_mask].reshape(B, -1, video_patch_dim)

        return {
            'audio_mean_embed': audio_mean_embed,
            'video_mean_embed': video_mean_embed,
            'audio_gt_masked': audio_gt_masked,
            'video_gt_masked': video_gt_masked,
            'audio_pred_masked': audio_pred_masked,
            'video_pred_masked': video_pred_masked,
            'audio_pred_unmasked': audio_pred_unmasked,
            'video_pred_unmasked': video_pred_unmasked,
        }

    def classifier(self, features):
        """Score masked and unmasked reconstructions with the modality discriminators."""
        # first half logit corresponds to prediction on masked patches, while last half for unmased patches
        audio_discriminator_input = torch.cat(
            [features['audio_pred_masked'].reshape(-1, 16 * 16), features['audio_pred_unmasked'].reshape(-1, 16 * 16)], dim=0
        )  # [2 * B * 256, 256]
        audio_cls_pred = self.audio_discriminator(audio_discriminator_input)

        video_discriminator_input = torch.cat(
            [
                features['video_pred_masked'].reshape(-1, 2 * 16 * 16 * 3),
                features['video_pred_unmasked'].reshape(-1, 2 * 16 * 16 * 3),
            ],
            dim=0,
        )  # [2 * B * 784, 1536]
        video_cls_pred = self.video_discriminator(video_discriminator_input)
        # print('discriminator output', audio_cls_pred.shape, video_cls_pred.shape)

        return {'audio_cls_pred': audio_cls_pred, 'video_cls_pred': video_cls_pred}

    def get_losses(self, data_dict, pred_dict, stage=None):
        """Return generator or discriminator losses selected by ``stage``.

        The call order is ``get_losses(data_dict, pred_dict, stage=...)``.
        Omitted stage uses the generator objective for validation.
        """
        stage = 'generator' if stage is None else stage
        if stage not in ('generator', 'discriminator'):
            raise ValueError(f'Unknown AVFF pretraining stage: {stage!r}.')
        num_masked_audio_patches = len(pred_dict['audio_cls_pred']) // 2
        num_masked_video_patches = len(pred_dict['video_cls_pred']) // 2
        if stage == 'generator':
            Lc = self.loss_func_dict['Lc'](pred_dict['audio_mean_embed'], pred_dict['video_mean_embed'])
            Lc_weight = self.loss_weight_dict.get('Lc', 1.0)
            Lrec_audio = self.loss_func_dict['Lrec'](pred_dict['audio_gt_masked'], pred_dict['audio_pred_masked'])
            Lrec_video = self.loss_func_dict['Lrec'](pred_dict['video_gt_masked'], pred_dict['video_pred_masked'])
            Lrec_weight = self.loss_weight_dict.get('Lrec', 1.0)
            LadvG_audio = self.loss_func_dict['LadvG'](pred_dict['audio_cls_pred'][:num_masked_audio_patches])
            LadvG_video = self.loss_func_dict['LadvG'](pred_dict['video_cls_pred'][:num_masked_video_patches])
            LadvG_weight = self.loss_weight_dict.get('LadvG', 1.0)
            return {
                'overall': Lc_weight * Lc + Lrec_weight * (Lrec_audio + Lrec_video) + LadvG_weight * (LadvG_audio + LadvG_video),
                'Lc': Lc,
                'Lrec': Lrec_audio + Lrec_video,
                'LadvG': LadvG_audio + LadvG_video,
            }
        elif stage == 'discriminator':
            LadvD_audio = self.loss_func_dict['LadvD'](
                fake_scores=pred_dict['audio_cls_pred'][:num_masked_audio_patches],
                real_scores=pred_dict['audio_cls_pred'][num_masked_audio_patches:],
            )
            LadvD_video = self.loss_func_dict['LadvD'](
                fake_scores=pred_dict['video_cls_pred'][:num_masked_video_patches],
                real_scores=pred_dict['video_cls_pred'][num_masked_video_patches:],
            )
            LadvD = LadvD_audio + LadvD_video
            return {'overall': self.loss_weight_dict.get('LadvD', 1.0) * LadvD, 'LadvD': LadvD}

    def forward(self, data_dict, stage_name=None, inference=False):
        """Compute the requested stage while isolating its gradient path.

        Discriminator updates use detached generator reconstructions. Generator
        updates differentiate through fixed discriminator weights. Omitted stage
        uses the generator path, including no-grad validation calls.
        """
        stage_name = 'generator' if stage_name is None else stage_name
        if stage_name not in ('generator', 'discriminator'):
            raise ValueError(f'Unknown AVFF pretraining stage: {stage_name!r}.')
        with torch.set_grad_enabled(torch.is_grad_enabled() and stage_name != 'discriminator'):
            features = self.features(data_dict)
        if stage_name == 'generator':
            parameters = list(self.audio_discriminator.parameters()) + list(self.video_discriminator.parameters())
            original_flags = [parameter.requires_grad for parameter in parameters]
            try:
                for parameter in parameters:
                    parameter.requires_grad_(False)
                pred = self.classifier(features)
            finally:
                for parameter, requires_grad in zip(parameters, original_flags):
                    parameter.requires_grad_(requires_grad)
        else:
            pred = self.classifier(features)
        pred_dict = {
            'audio_mean_embed': features['audio_mean_embed'],
            'video_mean_embed': features['video_mean_embed'],
            'audio_gt_masked': features['audio_gt_masked'],
            'video_gt_masked': features['video_gt_masked'],
            'audio_pred_masked': features['audio_pred_masked'],
            'video_pred_masked': features['video_pred_masked'],
            'audio_pred_unmasked': features['audio_pred_unmasked'],
            'video_pred_unmasked': features['video_pred_unmasked'],
            'audio_cls_pred': pred['audio_cls_pred'],
            'video_cls_pred': pred['video_cls_pred'],
        }
        return pred_dict


@DETECTOR.register_module(module_name='AVFF')
class AVFFDetector(AbstractDetector):
    """Supervised AVFF classification using AudioMAE and MARLIN token features."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.video_backbone, self.audio_backbone = self.build_backbone(config['backbone'])
        self.loss_func_dict, self.loss_weight_dict = self.build_loss(config['losses'])

        # In stage2, a2v and v2a has different dimension
        self.a2v_in_stage2 = A2VNetwork(audio_dim=64 * 8, visual_dim=196 * 8, embed_dim=768, num_heads=12)
        self.v2a_in_stage2 = V2ANetwork(audio_dim=64 * 8, visual_dim=196 * 8, embed_dim=768, num_heads=12)

        self.mlp_vision = torch.nn.Linear(1568, 1024)
        self.mlp_audio = torch.nn.Linear(512, 1024)
        self.mlp_head = nn.Sequential(
            nn.Linear(2048, 1024), nn.Dropout(0.5), nn.ReLU(), nn.Linear(1024, 1024), nn.Dropout(0.5), nn.ReLU(), nn.Linear(1024, 2)
        )

        if config.get('pretrained'):
            ckpt = torch.load(config['pretrained'], map_location='cpu')
            self.load_state_dict(ckpt, strict=False)

    def build_backbone(self, backbone_config):
        """Build the video encoder and decoder-free audio encoder from configuration."""
        video_backbone_config = backbone_config['video']
        name = video_backbone_config['name']
        version = video_backbone_config.get('version')
        params = {'as_feature_extractor': False}
        if version in BACKBONE[name].predefined_versions:
            params.update(BACKBONE[name].predefined_versions[version])
        params.update(video_backbone_config.get('params', {}))
        video_backbone = BACKBONE[name](**params)
        if video_backbone_config.get('weight'):
            ckpt = torch.load(video_backbone_config['weight'], map_location='cpu')
            video_backbone.load_state_dict(ckpt, strict=False)

        audio_backbone_config = backbone_config['audio']
        name = audio_backbone_config['name']
        version = audio_backbone_config.get('version')
        params = {'in_chans': 1, 'audio_exp': True, 'img_size': (1024, 128)}
        if version in BACKBONE[name].predefined_versions:
            params.update(BACKBONE[name].predefined_versions[version])
        params.update(audio_backbone_config.get('params', {}))
        audio_backbone = BACKBONE[name](**params)
        if audio_backbone_config.get('weight'):
            ckpt = torch.load(audio_backbone_config['weight'], map_location='cpu')
            audio_backbone.load_state_dict(ckpt, strict=False)

        audio_backbone = BACKBONE[name](**params)
        delattr(audio_backbone, 'decoder_embed')
        delattr(audio_backbone, 'mask_token')
        delattr(audio_backbone, 'decoder_pos_embed')
        delattr(audio_backbone, 'decoder_blocks')
        delattr(audio_backbone, 'decoder_norm')
        delattr(audio_backbone, 'decoder_pred')
        torch.cuda.empty_cache()

        return video_backbone, audio_backbone

    def build_loss(self, losses):
        """Construct named loss modules and return their modules and configured weights."""
        loss_func_dict, loss_weight_dict = {}, {}
        for loss in losses:
            loss_func_dict[loss['name']] = LOSSFUNC[loss['func']](**loss.get('params', {}))
            loss_weight_dict[loss['name']] = loss.get('weight', 1.0)

        return loss_func_dict, loss_weight_dict

    def features(self, data_dict):
        """Return audio, video and concatenated fusion features.

        Expects a 16-frame video clip and 128-bin audio spectrogram. Audio
        preprocessing pads/truncates and replaces the batch tensor in place.
        """
        embed_dim = 768
        target_Ta = 1024
        P = 16
        B, Ta, F = data_dict['audio'].shape
        _, C, Tv, H, W = data_dict['video'].shape

        # AudioMAE encoding
        if Ta > target_Ta:
            data_dict['audio'] = data_dict['audio'][:, :target_Ta, :]
        elif Ta < target_Ta:
            # Pad 0 here. But data_dict['audio'] is retrieved after normalization. To pad 0, we need pad (0 - mean) / std
            zero_pad = (
                torch.ones((B, target_Ta - Ta, 128), device=data_dict['audio'].device, dtype=data_dict['audio'].dtype)
                * 4.2677393
                / 9.1379948
            )
            data_dict['audio'] = torch.cat([data_dict['audio'], zero_pad], dim=1)
        Ta = target_Ta
        num_audio_patches = Ta // P * F // P
        data_dict['audio'] = data_dict['audio'].unsqueeze(1)
        audio_feat, _, ids_restore, _ = self.audio_backbone.forward_encoder(data_dict['audio'], mask_ratio=0)
        audio_embed = audio_feat[:, 1:, :]  # remove cls token
        audio_embed = audio_embed.reshape(B, 8, -1, embed_dim)
        # print('audio feature', audio_embed.shape, audio_embed_vis.shape)

        # MARLIN encoding
        num_video_patches = Tv // 2 * H // P * W // P
        video_embed = self.video_backbone.encoder.extract_features(data_dict['video'], seq_mean_pool=False)  # [B, 1568, 768]
        video_embed = video_embed.reshape(B, 8, -1, embed_dim)
        # print('video feature', video_embed.shape, video_embed_vis.shape)

        # feature fusion
        v2a_embed = self.v2a_in_stage2(video_embed)  # [B, 8, 64, 768]
        a2v_embed = self.a2v_in_stage2(audio_embed)  # [B, 8, 196, 768]
        # print('audio fusion', audio_fused_embed.shape)
        # print('video fusion', video_fused_embed.shape)

        # Concat along feature dimension
        video_fusion = torch.concat((a2v_embed, video_embed), dim=-1)
        audio_fusion = torch.concat((v2a_embed, audio_embed), dim=-1)
        video_fusion = video_fusion.mean(dim=-1)
        audio_fusion = audio_fusion.mean(dim=-1)

        video_feat = rearrange(video_fusion, 'b t c -> b (t c)')
        audio_feat = rearrange(audio_fusion, 'b t c -> b (t c)')
        video_feat = self.mlp_vision(video_feat)
        audio_feat = self.mlp_audio(audio_feat)
        fused_feature = torch.concat((video_feat, audio_feat), dim=-1)

        return {'video': video_feat, 'audio': audio_feat, 'fused': fused_feature}

    def classifier(self, features):
        """Return two-class logits from the concatenated fusion feature."""
        logit = self.mlp_head(features['fused'])
        return logit

    def get_losses(self, data_dict, pred_dict, stage=None):
        """Return overall-label cross-entropy as ``overall`` and ``CE``."""
        ce_loss = self.loss_func_dict['CE'](pred_dict['cls'], data_dict['label'])
        return {'overall': ce_loss, 'CE': ce_loss}

    def forward(self, data_dict, stage_name=None, inference=False):
        """Return ``cls``, fake-class ``prob`` and audio/video/fusion feature tensors."""
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {
            'video_feature': features['video'],
            'audio_feature': features['audio'],
            'fusion_feature': features['fused'],
            'cls': pred,
            'prob': prob,
        }
        return pred_dict


"""
The following code is from https://github.com/JoeLeelyf/OpenAVFF/blob/main/src/models/fusion_modules.py
"""


class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=False):
        super(SimpleTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.fc2(F.relu(self.fc1(x)))
        return self.norm2(x + ffn_output)


class A2VNetwork(nn.Module):
    def __init__(self, audio_dim, visual_dim, embed_dim, num_heads, batch_first=False):
        super(A2VNetwork, self).__init__()
        self.mlp = nn.Linear(audio_dim, visual_dim)
        self.transformer = SimpleTransformerBlock(embed_dim, num_heads, batch_first=batch_first)

    def forward(self, avis):
        b, t, c, d = avis.shape  # [B, 4, 64, 768]
        avis = einops.rearrange(avis, 'b t c d -> b d (t c)')  # [B, 768, 4 * 64]
        va = self.mlp(avis)  # MLP to match dimensions  # [B, 768, 4 * 196]
        va = va.transpose(1, 2)  # [B, 4 * 196, 768]
        va = self.transformer(va)
        va = einops.rearrange(va, 'b (t c) d -> b t c d', t=t)  # [B, 4, 196, 768]
        return va


class V2ANetwork(nn.Module):
    def __init__(self, visual_dim, audio_dim, embed_dim, num_heads, batch_first=False):
        super(V2ANetwork, self).__init__()
        self.mlp = nn.Linear(visual_dim, audio_dim)
        self.transformer = SimpleTransformerBlock(embed_dim, num_heads, batch_first=batch_first)

    def forward(self, vvis):
        b, t, c, d = vvis.shape
        vvis = einops.rearrange(vvis, 'b t c d -> b d (t c)')
        av = self.mlp(vvis)  # MLP to match dimensions
        av = av.transpose(1, 2)
        av = self.transformer(av)
        av = einops.rearrange(av, 'b (t c) d -> b t c d', t=t)
        return av
