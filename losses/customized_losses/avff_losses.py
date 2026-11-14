import os
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from losses.abstract_loss import AbstractLoss
from tools.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="avff_contrastive")
class AVFFContrastiveLoss(AbstractLoss):
    """Symmetric contrastive loss for paired audio and video features."""

    def __init__(self, tau=0.07, **params):
        """Initialize the similarity temperature.

        Args:
            tau: Temperature divisor for normalized feature similarities.
            **params: Additional configuration entries accepted for detector compatibility.
        """
        super().__init__()
        self.tau = tau

    def _all_gather(self, tensor):
        """Gather features across ranks while retaining local feature gradients."""
        tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        tensors_gather[dist.get_rank()] = tensor
        return torch.cat(tensors_gather, dim=0)

    def forward(self, audio_features, video_features):
        """Compare matching audio/video rows in both directions.

        Args:
            audio_features: Audio feature tensor with shape (batch, feature_dim).
            video_features: Matching video tensor with the same shape.

        Returns:
            Scalar mean of audio-to-video and video-to-audio cross-entropy losses.
            Initialized distributed runs compare the gathered features across ranks.
        """
        audio_features = F.normalize(audio_features, dim=-1)
        video_features = F.normalize(video_features, dim=-1)

        if dist.is_available() and dist.is_initialized():
            all_audio = self._all_gather(audio_features)
            all_video = self._all_gather(video_features)
        else:
            all_audio = audio_features
            all_video = video_features

        logits_a2v = torch.mm(all_audio, all_video.t()) / self.tau

        labels = torch.arange(logits_a2v.size(0), device=logits_a2v.device)

        loss_a2v = F.cross_entropy(logits_a2v, labels)
        loss_v2a = F.cross_entropy(logits_a2v.t(), labels)

        return (loss_a2v + loss_v2a) / 2
