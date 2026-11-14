import os
import sys
import torch

from einops import rearrange
from torch import nn

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from losses.abstract_loss import AbstractLoss
from tools.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name='wasserstein_generator')
class WassersteinGeneratorLoss(AbstractLoss):
    """Wasserstein generator objective over fake discriminator scores."""

    def __init__(self):
        super().__init__()

    def forward(self, fake_scores):
        """Compute the generator objective.

        Args:
            fake_scores: Discriminator scores for generated content.

        Returns:
            Negative mean fake score as a scalar tensor.
        """
        return -fake_scores.mean()


@LOSSFUNC.register_module(module_name='wasserstein_discriminator')
class WassersteinDiscriminatorLoss(AbstractLoss):
    """Wasserstein discriminator objective over fake and real scores."""

    def __init__(self):
        super().__init__()

    def forward(self, fake_scores, real_scores):
        """Compute the discriminator objective.

        Args:
            fake_scores: Discriminator scores for generated content.
            real_scores: Discriminator scores for real content.

        Returns:
            Mean fake score minus mean real score as a scalar tensor.
        """
        return fake_scores.mean() - real_scores.mean()


@LOSSFUNC.register_module(module_name='wasserstein_discriminator_gradient_penalty')
class WassersteinDiscriminatorLossGradientPenalty(nn.Module):
    """Wasserstein discriminator objective with an interpolation gradient penalty."""

    def __init__(self, gp_weight=10.0):
        """Initialize the gradient-penalty multiplier.

        Args:
            gp_weight: Weight applied to the gradient-penalty term.
        """
        super().__init__()
        self.gp_weight = gp_weight

    def _gradient_penalty(self, real_content, fake_content, discriminator):
        """Penalize channel-wise gradient norms on interpolated content.

        Args:
            real_content: Real inputs with shape (batch, tokens, channels).
            fake_content: Generated inputs with the same shape.
            discriminator: Module that scores the interpolated content.

        Returns:
            Mean squared deviation of token gradient norms from one.
        """
        batch_size = real_content.size(0)

        alpha = torch.rand(batch_size, 1, 1, device=real_content.device)
        alpha = alpha.expand_as(real_content)

        interpolates = alpha * real_content + (1 - alpha) * fake_content
        interpolates.requires_grad_(True)

        disc_interpolates = discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = rearrange(gradients, 'b n c -> (b n) c')
        grad_norm = gradients.norm(2, dim=1)

        gp = ((grad_norm - 1) ** 2).mean()

        return gp

    def forward(self, fake_scores, real_scores, fake_content, real_content, discriminator):
        """Compute the discriminator objective and its gradient penalty.

        Args:
            fake_scores: Discriminator scores for generated content.
            real_scores: Discriminator scores for real content.
            fake_content: Generated discriminator inputs with shape (batch, tokens, channels).
            real_content: Real discriminator inputs with the same shape.
            discriminator: Module that scores the interpolated content.

        Returns:
            Tuple of scalar tensors (total_loss, gradient_penalty).
        """
        wgan_loss = fake_scores.mean() - real_scores.mean()

        gp = self._gradient_penalty(real_content, fake_content, discriminator)

        total_loss = wgan_loss + self.gp_weight * gp

        return total_loss, gp
