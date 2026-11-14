import os
import sys
import torch

from torch import nn

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from losses.abstract_loss import AbstractLoss
from tools.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="cross_entropy")
class CrossEntropyLoss(AbstractLoss):
    """Cross-entropy wrapper with optional inverse class-count weighting.

    Supplying both real_num and fake_num constructs the class weights on CUDA.
    """

    def __init__(self, **params):
        super().__init__()
        if params.get('real_num') and params.get('fake_num'):
            if params['real_num'] < params['fake_num']:
                loss_reweight = [params['fake_num'] / params['real_num'], 1.0]
            else:
                loss_reweight = [1.0, params['real_num'] / params['fake_num']]
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(loss_reweight).float().cuda())
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """Compute mean cross-entropy for class logits.

        Args:
            inputs: Predicted class scores with shape (batch, num_classes).
            targets: Integer ground-truth class indices with shape (batch,).

        Returns:
            Scalar cross-entropy loss using the configured class weights.
        """
        loss = self.loss_fn(inputs, targets)

        return loss
