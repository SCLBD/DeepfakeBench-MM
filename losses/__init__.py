import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from losses.customized_losses.avff_losses import AVFFContrastiveLoss
from losses.customized_losses.frade_losses import FRADECenterClusterLoss, FRADEContrastiveLoss
from losses.customized_losses.mcl_losses import MCLJSDLoss, MCLNCELoss
from losses.standard_losses.cross_entropy import CrossEntropyLoss
from losses.standard_losses.mean_squared_error import MeanSquaredErrorLoss
from losses.standard_losses.wasserstein_gan import (
    WassersteinDiscriminatorLoss,
    WassersteinDiscriminatorLossGradientPenalty,
    WassersteinGeneratorLoss,
)
