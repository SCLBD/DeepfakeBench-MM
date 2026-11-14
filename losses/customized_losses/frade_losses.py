import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from losses.abstract_loss import AbstractLoss
from tools.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="frade_cc")
class FRADECenterClusterLoss(AbstractLoss):
    """Attract real fusion features to a learnable center and repel fake features."""

    def __init__(self, input_dim=768, gamma2=0.25, **params):
        super().__init__()
        self.center = nn.Parameter(torch.randn(input_dim, device='cuda' if torch.cuda.is_available() else 'cpu'))
        self.gamma2 = gamma2

    def forward(self, fusion_feature, labels):
        """Compute the center-cluster term for a batch.

        Args:
            fusion_feature: Fused features with shape (batch, input_dim).
            labels: Authenticity labels of shape (batch,), with 0 real and 1 fake.

        Returns:
            Scalar real half-MSE minus the fake half-MSE capped by gamma2.
        """
        real_feats = fusion_feature[labels == 0]
        fake_feats = fusion_feature[labels == 1]
        loss = torch.tensor(0.0, device=fusion_feature.device)
        if real_feats.shape[0] > 0:
            real_loss = 0.5 * F.mse_loss(real_feats, self.center.expand(real_feats.shape[0], -1), reduction='mean')
            loss += real_loss

        if fake_feats.shape[0] > 0:
            fake_loss = 0.5 * F.mse_loss(fake_feats, self.center, reduction='mean')
            loss -= torch.min(fake_loss, torch.tensor(self.gamma2, device=fusion_feature.device))

        return loss


@LOSSFUNC.register_module(module_name="frade_cmc")
class FRADEContrastiveLoss(AbstractLoss):
    """Apply label-dependent cosine penalties to paired video/audio features."""

    def __init__(self, gamma1=0.2, **params):
        super().__init__()
        self.gamma1 = gamma1
        if params.get('real_num') and params.get('fake_num'):
            if params['real_num'] < params['fake_num']:
                self.real_loss_weight, self.fake_loss_weight = params['fake_num'] / params['real_num'], 1.0
            else:
                self.real_loss_weight, self.fake_loss_weight = 1.0, params['real_num'] / params['fake_num']
        else:
            self.real_loss_weight, self.fake_loss_weight = 1.0, 1.0

    def forward(self, video_feature, audio_feature, labels):
        """Compute the cross-modal contrastive term.

        Args:
            video_feature: Video features with shape (batch, feature_dim).
            audio_feature: Paired audio features with the same shape.
            labels: Authenticity labels of shape (batch,), with 0 real and 1 fake.

        Returns:
            Mean weighted real-pair distance and fake-pair margin penalty.
        """
        cos_similarities = F.cosine_similarity(video_feature, audio_feature)
        loss = (
            self.real_loss_weight * (1 - labels.float()) * (1 - cos_similarities)
            + self.fake_loss_weight * labels.float() * torch.clamp(cos_similarities - self.gamma1, min=0.0)
        )

        return loss.mean()
