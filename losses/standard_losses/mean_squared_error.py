import os
import sys

from torch import nn

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from losses.abstract_loss import AbstractLoss
from tools.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="mse")
class MeanSquaredErrorLoss(AbstractLoss):
    """Mean-squared-error wrapper with a configurable PyTorch reduction."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)


    def forward(self, tensor1, tensor2):
        """Compute the configured MSE reduction.

        Args:
            tensor1: Predicted values.
            tensor2: Target values with a compatible shape.

        Returns:
            MSE tensor, reduced according to the constructor's reduction argument.
        """
        loss = self.loss_fn(tensor1, tensor2)

        return loss
