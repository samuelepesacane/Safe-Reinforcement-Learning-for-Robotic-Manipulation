import os
import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """
    Seed all relevant random number generators to improve experiment reproducibility.

    This function sets the seed for Python's built-in ``random`` module, NumPy,
    and PyTorch (CPU and all available CUDA devices). Optionally, it also
    configures PyTorch's CuDNN backend to run in deterministic mode, which
    reduces nondeterminism at the cost of some performance.

    :param seed: Global random seed to use for all libraries.
        :type seed: int
    :param deterministic_torch: If ``True``, enable deterministic CuDNN kernels
        by setting ``torch.backends.cudnn.deterministic = True`` and
        ``torch.backends.cudnn.benchmark = False``. If ``False``, CuDNN is left
        in its default (potentially faster but less reproducible) mode.
        :type deterministic_torch: bool

    :return: This function does not return anything; it configures global state.
        :rtype: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
