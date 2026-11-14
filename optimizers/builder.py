from adamp import AdamP
from torch.optim import Adam, AdamW, RMSprop, SGD


def choose_optimizer(model_parameters, config):
    """Build an optimizer over parameters that require gradients.

    Args:
        model_parameters: Iterable of model or stage parameters.
        config: Dictionary with case-sensitive type and constructor params entries.

    Returns:
        Configured PyTorch-compatible optimizer.
    """
    trainable_params = filter(lambda p: p.requires_grad, model_parameters)

    if config['type'] == 'Adam':
        return Adam(params=trainable_params, **config['params'])
    elif config['type'] == 'AdamW':
        return AdamW(params=trainable_params, **config['params'])
    elif config['type'] == 'AdamP':
        return AdamP(params=trainable_params, **config['params'])
    elif config['type'] == 'SGD':
        return SGD(params=trainable_params, **config['params'])
    elif config['type'] == 'RMSprop':
        return RMSprop(params=trainable_params, **config['params'])
    else:
        raise NotImplementedError(f"Unknown optimizer type: {config['type']}")
