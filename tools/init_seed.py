import numpy as np
import os
import random
import torch


def set_seed(seed, deterministic=True):
    """Seed Python, NumPy, and PyTorch random generators.

    Args:
        seed: Integer seed, also recorded as PYTHONHASHSEED.
        deterministic: Enable deterministic cuDNN settings and the cuBLAS workspace
            configuration when True.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def seed_worker(worker_id):
    """Seed a DataLoader worker's NumPy and Python generators from PyTorch's seed.

    Args:
        worker_id: Worker identifier supplied by DataLoader.
    """
    import numpy as np
    import random
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed):
    """Create a seeded PyTorch generator for DataLoader sampling.

    Args:
        seed: Integer seed for the new generator.

    Returns:
        Seeded torch.Generator instance.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
