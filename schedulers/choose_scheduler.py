from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

def choose_scheduler(optimizer, config):
    """
    Args:
        optimizer: created optimizer instance
        config: [dict] config containing key 'scheduler'
    Returns:
        scheduler
    """
    if config.get('scheduler', None) is None:
        return None
    elif config['scheduler']['type'] == 'step':
        return StepLR(optimizer, **config['scheduler']['params'])
    elif config['scheduler']['type'] == 'cosine':
        return StepLR(optimizer, **config['scheduler']['params'])
    else:
        raise NotImplementedError(f"Unknown scheduler type: {config['scheduler']['type']}")