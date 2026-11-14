import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

avhubert_dir = os.path.join(project_root, 'backbones/audio_video_backbones/av_hubert')
if avhubert_dir not in sys.path:
    sys.path.insert(0, avhubert_dir)

# Initialize Fairseq before importing AV-HuBERT task/model registration modules.
from fairseq import checkpoint_utils
from avhubert import hubert, hubert_asr, hubert_pretraining

from detectors.abstract_detector import AbstractDetector
from tools.registry import DETECTOR, LOSSFUNC


@DETECTOR.register_module(module_name='AVH')
class AVHDetector(AbstractDetector):
    """Separate AV-HuBERT stream extraction followed by audiovisual MLP classification."""

    def __init__(self, config):
        super().__init__()
        self.loss_func_dict, self.loss_weight_dict = self.build_loss(config['losses'])
        self.backbone = self.build_backbone(config)
        self.mlp = AVH_Sup_MLP()

    def build_backbone(self, config):
        """Restore the fine-tuned AV-HuBERT encoder and freeze its parameters."""
        models, _, task = checkpoint_utils.load_model_ensemble_and_task([config['pretrained']])
        backbone = models[0].encoder.w2v_model
        for param in backbone.parameters():
            param.requires_grad = False

        return backbone

    def build_loss(self, losses):
        """Construct named loss modules and return their modules and configured weights."""
        loss_func_dict, loss_weight_dict = {}, {}
        for loss in losses:
            loss_func_dict[loss['name']] = LOSSFUNC[loss['func']](**loss.get('params', {}))
            loss_weight_dict[loss['name']] = loss.get('weight', 1.0)

        return loss_func_dict, loss_weight_dict

    def features(self, data_dict):
        """Extract separate audio/video sequences from filterbanks and grayscale video.

        Stacks audio frames in groups of four, padding the batch audio in place.
        Video uses the fixed 96-pixel mouth crop within aligned 256-pixel frames.
        """
        # rearrange Log-Mel, 4 slice as 1 audio frame
        device = data_dict['audio'].device
        b, t, f = data_dict['audio'].shape
        if t % 4 != 0:
            pad = 4 - t % 4
            data_dict['audio'] = torch.cat([data_dict['audio'], torch.zeros((b, pad, f), device=device)], dim=1)
        audio_input = data_dict['audio'].reshape(b, -1, 4 * f)
        with torch.no_grad():
            audio_input = F.layer_norm(audio_input, audio_input.shape[-1:])
        audio_input = audio_input.transpose(1, 2)
        video_input = data_dict['video'][:, :, :, 158 - 48 : 158 + 48, 129 - 48 : 129 + 48]
        video_feature, _ = self.backbone.extract_finetune({'video': video_input, 'audio': None}, None, None)
        audio_feature, _ = self.backbone.extract_finetune({'video': None, 'audio': audio_input}, None, None)

        return {'video': video_feature, 'audio': audio_feature}

    def classifier(self, features):
        """Return the pooled MLP score and per-frame fused stream features."""
        output, fused_features = self.mlp(features)
        return output, fused_features

    def get_losses(self, data_dict, pred_dict):
        """Return overall-label cross-entropy as ``overall`` and ``CE``."""
        ce_loss = self.loss_func_dict['CE'](pred_dict['cls'], data_dict['label'])
        return {'overall': ce_loss, 'CE': ce_loss}

    def forward(self, data_dict, inference=False):
        """Return symmetric class logits, fake probability and modality/fusion features."""
        features = self.features(data_dict)
        output, fused_features = self.classifier(features)
        pred = torch.cat((-output.unsqueeze(1), output.unsqueeze(1)), dim=1)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {
            'cls': pred,
            'prob': prob,
            'audio_feature': features['audio'],
            'video_feature': features['video'],
            'fusion_feature': fused_features,
        }
        return pred_dict


'''
The following code is from https://github.com/bit-ml/AVH-Align/blob/main/avh_sup/mlp.py
'''


class AVH_Sup_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 1024
        self.visual_proj = nn.Linear(1024, hidden_dim // 2)
        self.audio_proj = nn.Linear(1024, hidden_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_feats):
        video_feats, audio_feats = input_feats['video'], input_feats['audio']
        visual_proj = self.visual_proj(video_feats)  # [B, T, 512]
        audio_proj = self.audio_proj(audio_feats)  # [B, T, 512]
        fused_features = torch.cat((visual_proj, audio_proj), dim=-1)  # [B, T, 1024]
        output = self.mlp(fused_features)[:, :, 0]  # [B, T]
        return torch.logsumexp(output, dim=-1), fused_features  # [B], [B, T, 1024]
