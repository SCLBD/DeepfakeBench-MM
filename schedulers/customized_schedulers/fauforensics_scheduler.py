from torch.optim.lr_scheduler import LambdaLR


def fauforensics_scheduler(optimizer, power, lr_decay, total_iters):
    """Build the FauForensics polynomial decay schedule.

    Args:
        optimizer: Optimizer whose learning rate is scheduled.
        power: Exponent applied to the remaining iteration fraction.
        lr_decay: Residual multiplier at and beyond total_iters.
        total_iters: Positive number of total scheduler iterations.

    Returns:
        LambdaLR with multiplier max(0, 1 - step / total_iters) ** power
        scaled from one to lr_decay.
    """
    def poly_lambda(step):
        coeff = max(0.0, 1 - step / total_iters) ** power
        return coeff * (1 - lr_decay) + lr_decay

    assert total_iters is not None and total_iters > 0, 'total_iters must be provided for FauForensics scheduler.'
    return LambdaLR(optimizer, lr_lambda=poly_lambda)
