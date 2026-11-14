"""SpeechForensics evaluation adapter supplied by Xiaoyu.

Uses a frozen AV-HuBERT speech checkpoint and sliding-offset cosine matching.
Only the evaluation wrapper is integrated; benchmark-specific training is not
required. The supplied score and crop functions are retained unchanged.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_AVHUBERT_ROOT = os.path.join(project_root, 'backbones/audio_video_backbones/av_hubert')
if _AVHUBERT_ROOT not in sys.path:
    sys.path.insert(0, _AVHUBERT_ROOT)

from detectors.abstract_detector import AbstractDetector
from tools.registry import DETECTOR, LOSSFUNC


class _CheckpointBackbonePlaceholder(nn.Module):
    """Placeholder replaced by the Fairseq checkpoint loader in ``test.py``."""

    def extract_finetune(self, *args, **kwargs):
        raise RuntimeError(
            'SpeechForensics test backbone has not been injected. Load the official '
            'checkpoint with test.py before running inference.'
        )


def _crop_mouth_roi(video, landmarks, crop_size):
    """Reproduce the official landmark-centred mouth crop on batched tensors."""
    if video.ndim != 5:
        raise ValueError(f'Expected video shape (B, C, T, H, W), got {tuple(video.shape)}.')
    if landmarks.ndim != 4 or landmarks.shape[-2:] != (68, 2):
        raise ValueError(f'Expected landmarks shape (B, T, 68, 2), got {tuple(landmarks.shape)}.')

    b, c, t, h, w = video.shape
    if landmarks.shape[:2] != (b, t):
        raise ValueError(
            'Video and landmark batch/time dimensions do not match: ' f'video={(b, t)}, landmarks={tuple(landmarks.shape[:2])}.'
        )
    if crop_size % 2 != 0:
        raise ValueError(f'The official mouth crop size must be even, got {crop_size}.')
    if h < crop_size or w < crop_size:
        raise ValueError(f'Cannot extract a {crop_size}x{crop_size} mouth crop from video frames of size {h}x{w}.')

    half = crop_size // 2
    mouth_centres = landmarks[:, :, 48:68].mean(dim=2).round().long()
    centre_x = mouth_centres[..., 0].clamp(half, w - half).reshape(-1)
    centre_y = mouth_centres[..., 1].clamp(half, h - half).reshape(-1)
    offsets = torch.arange(-half, half, device=video.device)
    x_indices = centre_x[:, None] + offsets[None, :]
    y_indices = centre_y[:, None] + offsets[None, :]

    # Gather integer pixels so this has the same sampling semantics as the
    # official ``cut_patch`` implementation (no resize or interpolation).
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    rows = torch.gather(
        frames,
        2,
        y_indices[:, None, :, None].expand(-1, c, -1, w),
    )
    patches = torch.gather(
        rows,
        3,
        x_indices[:, None, None, :].expand(-1, c, crop_size, -1),
    )
    return patches.reshape(b, t, c, crop_size, crop_size).permute(0, 2, 1, 3, 4)


def _prepare_inputs(data_dict, image_crop_size=88, mouth_crop_size=None):
    audio = data_dict['audio']
    b, t, f = audio.shape
    if t % 4 != 0:
        pad = 4 - t % 4
        audio = torch.cat([audio, audio.new_zeros((b, pad, f))], dim=1)
    audio_input = audio.reshape(b, -1, 4 * f)
    audio_input = F.layer_norm(audio_input, audio_input.shape[-1:])
    audio_input = audio_input.transpose(1, 2)

    video = data_dict['video']
    if mouth_crop_size is not None:
        if 'landmarks' not in data_dict:
            raise RuntimeError(
                'SpeechForensics evaluation requires facial landmarks to reproduce '
                'the official 96x96 mouth ROI. Set data_config.with_landmark=true.'
            )
        video = _crop_mouth_roi(video, data_dict['landmarks'], mouth_crop_size)

    _, _, _, h, w = video.shape
    crop = int(image_crop_size)
    if h < crop or w < crop:
        raise ValueError(f'Cannot centre-crop {crop}x{crop} from video input of size {h}x{w}.')
    top = int(round(h - crop) / 2.0)
    left = int(round(w - crop) / 2.0)
    video_input = video[:, :, :, top : top + crop, left : left + crop]
    return video_input, audio_input


@DETECTOR.register_module(module_name='SpeechForensics')
class SpeechForensicsDetector(AbstractDetector):
    """Frozen stage-1 checkpoint wrapper for testing only.

    ``test.py`` calls load_checkpoint to restore the supplied speech checkpoint.
    This adapter performs no benchmark-specific fine-tuning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_func_dict, self.loss_weight_dict = self.build_loss(config.get('losses', []))
        self.backbone = self.build_backbone(config)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.offset_window = int(config.get('offset_window', 15))
        self.image_crop_size = int(config.get('image_crop_size', 88))
        self.mouth_crop_size = int(config.get('mouth_crop_size', 96))
        if self.mouth_crop_size < self.image_crop_size:
            raise ValueError('mouth_crop_size must be greater than or equal to image_crop_size.')

    def _encode(self, data_dict):
        """Prepare aligned mouth crops/audio stacks and extract frozen stream features."""
        video_input, audio_input = _prepare_inputs(
            data_dict,
            image_crop_size=self.image_crop_size,
            mouth_crop_size=self.mouth_crop_size,
        )
        with torch.no_grad():
            video_feature, _ = self.backbone.extract_finetune(
                {'video': video_input, 'audio': None}, padding_mask=None, output_layer=None
            )
            audio_feature, _ = self.backbone.extract_finetune(
                {'video': None, 'audio': audio_input}, padding_mask=None, output_layer=None
            )
        return video_feature, audio_feature

    @staticmethod
    def _sliding_window_similarity(video_feat, audio_feat, vshift):
        """Return the best mean cosine similarity across zero-padded temporal offsets."""
        video_n = F.normalize(video_feat, p=2, dim=-1)
        audio_n = F.normalize(audio_feat, p=2, dim=-1)

        # Match the official calc_cos_dist: when lengths differ, sample the
        # visual sequence to the audio feature length (not the reverse).
        if video_n.size(1) != audio_n.size(1):
            idx = torch.linspace(
                0,
                video_n.size(1) - 1,
                audio_n.size(1),
                device=video_n.device,
            ).long()
            video_n = video_n[:, idx]

        t = audio_n.size(1)
        audio_pad = F.pad(audio_n, (0, 0, vshift, vshift))
        window = 2 * vshift + 1
        audio_windows = torch.stack([audio_pad[:, i : i + t] for i in range(window)], dim=2)
        sims = (video_n.unsqueeze(2) * audio_windows).sum(dim=-1)
        mean_sims = sims.mean(dim=1)
        score, _ = mean_sims.max(dim=-1)
        return score

    def build_backbone(self, config):
        """Create the placeholder replaced by ``load_checkpoint`` before inference."""
        return _CheckpointBackbonePlaceholder()

    def load_checkpoint(self, checkpoint_path):
        """Restore architecture/task config with the supplied Fairseq checkpoint."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        # Import the vendored user module to register the checkpoint's task/model.
        import avhubert  # noqa: F401
        from fairseq import checkpoint_utils

        models, _saved_cfg, _task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        if len(models) != 1:
            raise RuntimeError('SpeechForensics expects exactly one checkpoint model.')
        backbone = models[0]
        if hasattr(backbone, 'decoder'):
            backbone = backbone.encoder.w2v_model
        if not callable(getattr(backbone, 'extract_finetune', None)):
            raise TypeError('Checkpoint does not expose AV-HuBERT extract_finetune.')
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def build_loss(self, losses):
        """Construct optional named losses; this adapter remains inference-only."""
        loss_func_dict, loss_weight_dict = {}, {}
        for loss in losses:
            loss_func_dict[loss['name']] = LOSSFUNC[loss['func']](**loss.get('params', {}))
            loss_weight_dict[loss['name']] = loss.get('weight', 1.0)
        return loss_func_dict, loss_weight_dict

    def train(self, mode=True):
        """Set module mode while keeping the backbone in evaluation mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, data_dict):
        """Return frozen audio/video sequences from a batch with aligned landmarks."""
        video_feature, audio_feature = self._encode(data_dict)
        return {'video': video_feature, 'audio': audio_feature}

    def classifier(self, features):
        """Return class scores, bounded fake score and raw matching similarity."""
        score = self._sliding_window_similarity(features['video'], features['audio'], self.offset_window)
        # The paper defines a real-video matching score (higher is more
        # authentic), while DeepfakeBench-MM expects probability of fake=1.
        fake_prob = ((1.0 - score) / 2.0).clamp(0.0, 1.0)
        class_scores = torch.stack([1.0 - fake_prob, fake_prob], dim=1)
        return class_scores, fake_prob, score

    def get_losses(self, data_dict, pred_dict):
        """Return optional CE when configured; raise when no training loss is defined."""
        ce = self.loss_func_dict['CE'](pred_dict['cls'], data_dict['label']) if 'CE' in self.loss_func_dict else None
        if ce is None:
            raise RuntimeError('SpeechForensics test wrapper does not define a training loss.')
        return {'overall': ce, 'CE': ce}

    def forward(self, data_dict, inference=False):
        """Run with ``inference=True`` and return class/fake scores and stream features.

        Requires grayscale ``video``, 26-bin ``audio`` filterbanks and aligned
        ``landmarks``. ``cls`` contains scores rather than unnormalized logits.
        """
        if not inference:
            raise RuntimeError('SpeechForensicsDetector is inference-only; call it with inference=True.')
        features = self.features(data_dict)
        class_scores, prob, score = self.classifier(features)
        return {
            'video_feature': features['video'],
            'audio_feature': features['audio'],
            'score': score,
            'cls': class_scores,
            'prob': prob,
        }
