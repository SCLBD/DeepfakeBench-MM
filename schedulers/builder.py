import os
import sys

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from schedulers.customized_schedulers.fauforensics_scheduler import fauforensics_scheduler


def choose_scheduler(optimizer, scheduler_config, total_iters):
    """Build the configured learning-rate scheduler.

    Args:
        optimizer: Optimizer whose learning rate is scheduled.
        scheduler_config: Dictionary with type, interval (step/epoch), and params, or None.
            The trainer handles interval; it is not passed to the scheduler constructor.
        total_iters: Total scheduler calls in the configured interval. For alternating
            step schedules, this is the number of updates of the corresponding optimizer.

    Returns:
        Configured scheduler, or None when no configuration is supplied.
    """
    if scheduler_config is None:
        return None
    if scheduler_config.get('interval') not in ('step', 'epoch'):
        raise ValueError('scheduler.interval must be explicitly set to "step" or "epoch".')
    if scheduler_config['type'] == 'step':
        return StepLR(optimizer, **scheduler_config['params'])
    elif scheduler_config['type'] == 'cosine':
        return CosineAnnealingLR(optimizer, **scheduler_config['params'])
    elif scheduler_config['type'] == 'FauForensics_Customized':
        return fauforensics_scheduler(optimizer, **scheduler_config['params'], total_iters=total_iters)
    else:
        raise NotImplementedError(f"Unknown scheduler type: {scheduler_config['type']}")
