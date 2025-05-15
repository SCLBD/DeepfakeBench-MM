from torch.optim import Adam, AdamW, SGD, RMSprop

def choose_optimizer(model, config):
    """
    Choose metric function under ./optimizers and standard optimizers
    Args:
        model: [nn.Module] training model
        config: [dict] config containing key 'optimizer'
    Returns:
        optimizer
    """
    if config['optimizer']['type'] == 'adam':
        return Adam(params=model.parameters(), **config['optimizer']['params'])
    elif config['optimizer']['type'] == 'adamw':
        return AdamW(params=model.parameters(), **config['optimizer']['params'])
    elif config['optimizer']['type'] == 'sgd':
        return SGD(params=model.parameters(), **config['optimizer']['params'])
    elif config['optimizer']['type'] == 'rmsprop':
        return RMSprop(params=model.parameters(), **config['optimizer']['params'])
    else:
        raise NotImplementedError(f"Unknown optimizer type: {config['optimizer']['type']}")