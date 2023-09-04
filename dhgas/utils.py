import logging
from numpy.lib.arraysetops import isin


def get_logger(*args, **kwargs):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    )
    return logging.getLogger(*args, **kwargs)


def setinfo(logger):
    logger.setLevel(logging.INFO)


def setdebug(logger):
    logger.setLevel(logging.DEBUG)


import torch
from typing import Union


def get_device(device: Union[str, torch.device]):
    """
    Get device of passed argument. Will return a torch.device based on passed arguments.
    Can parse auto, cpu, gpu, cpu:x, gpu:x, etc. If auto is given, will automatically find
    available devices.
    Parameters
    ----------
    device: ``str`` or ``torch.device``
        The device to parse. If ``auto`` if given, will determine automatically.
    Returns
    -------
    device: ``torch.device``
        The parsed device.
    """
    assert isinstance(
        device, (str, torch.device)
    ), "Only support device of str or torch.device, get {} instead".format(device)
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


import numpy as np


class EarlyStopping:
    """EarlyStopping class to keep NN from overfitting. copied from nni"""

    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        """EarlyStopping step on each epoch
        @params metrics: metric value
        @return : True if stop
        """

        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def reset(self):
        self.best = None

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


from prettytable import PrettyTable


def cnt2str(x):
    if x >= 1e6:
        return f"{x//1e6} M"
    if x >= 1e3:
        return f"{x//1e3} K"


def count_parameters(model, toprint=True):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if toprint:
        print(table)
    print(f"Total Trainable Params: {cnt2str(total_params)} ")

    return total_params


def move_to(obj, device):
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, tuple):
        return tuple(move_to(v, device) for v in obj)
    return obj.to(device)


import time

from functools import wraps
from time import time


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r took: %2.4f sec" % (f, te - ts))
        return result

    return wrap


def debug(f):
    @wraps(f)
    def wrap(*args, **kw):
        print(f"func:{f}")
        result = f(*args, **kw)
        return result

    return wrap


import random
import numpy as np
import torch
import os


def setup_seed(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
