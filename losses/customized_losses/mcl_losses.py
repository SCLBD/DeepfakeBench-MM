import os
import sys
import torch
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from losses.abstract_loss import AbstractLoss
from tools.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name='mcl_jsd')
class MCLJSDLoss(AbstractLoss):
    """Sum both directed KL-style terms, with optional input softmax."""

    def __init__(self, weight=1.0, softmax_sign=False):
        super(MCLJSDLoss, self).__init__()
        self.weight = weight
        self.softmax_sign = softmax_sign

    def forward(self, p, q):
        """Compare paired class-score or probability matrices.

        Args:
            p: First tensor with shape (batch, num_classes).
            q: Second tensor with the same shape. Inputs are softmaxed unless
                softmax_sign is True, in which case callers supply probabilities.

        Returns:
            Batch mean of both directed KL-style terms, scaled by weight.
            The implementation does not construct a Jensen-Shannon mixture.
        """
        if self.softmax_sign is False:
            p = F.softmax(p, dim=1)
            q = F.softmax(q, dim=1)

        loss1 = p * torch.log(p / q + 1e-10)
        loss1 = loss1.sum(1)

        loss2 = q * torch.log(q / p + 1e-10)
        loss2 = loss2.sum(1)

        loss = loss1 + loss2
        loss = loss.mean() * self.weight
        return loss


@LOSSFUNC.register_module(module_name='mcl_nce')
class MCLNCELoss(AbstractLoss):
    """Contrast paired features using the MCL batch-softmax objective.

    Labels determine the similarity sign. Positive and negative pair losses are
    averaged separately before being summed.
    """

    def __init__(self, temperature=1.0):
        super(MCLNCELoss, self).__init__()
        self.temperature = float(temperature)

    def forward(self, f1, f2, targets1, targets2):
        """Compute label-dependent contrastive loss over the batch.

        Args:
            f1: First feature tensor with shape (batch, feature_dim).
            f2: Paired feature tensor with the same shape.
            targets1: Authenticity labels for f1, with shape (batch,).
            targets2: Authenticity labels for f2, with shape (batch,).

        Returns:
            Scalar sum of the separately averaged positive and negative pair terms.
        """
        # L2-normalise so that the squared Euclidean distance is in [0, 4]
        # and ``1 - 0.5 * dist`` is a proper cosine similarity in [-1, 1].
        f1 = F.normalize(f1, dim=1, p=2.0, eps=1e-10)
        f2 = F.normalize(f2, dim=1, p=2.0, eps=1e-10)

        dist_sq = ((f1 - f2).pow(2)).sum(dim=1)
        cos = 1.0 - 0.5 * dist_sq  # Paper Eq. (11).

        is_positive = (targets1 == targets2).float()
        is_negative = 1.0 - is_positive

        # paper Eq.(10): sign-flipped similarity
        sign = 2.0 * is_positive - 1.0
        H_s = sign * cos / self.temperature

        # paper Eq.(9): batch-level softmax (InfoNCE)
        p = F.softmax(H_s, dim=0)

        log_p = torch.log(p + 1e-10)
        log_one_minus_p = torch.log(1.0 - p + 1e-10)

        # Averaged over positive / negative pairs independently (1/O, 1/P).
        pos_count = torch.clamp(is_positive.sum(), min=1.0)
        neg_count = torch.clamp(is_negative.sum(), min=1.0)

        pos_loss = -(is_positive * log_p).sum() / pos_count
        neg_loss = -(is_negative * log_one_minus_p).sum() / neg_count

        return pos_loss + neg_loss
