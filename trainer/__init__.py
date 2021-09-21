import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import pdb
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

from __future__ import absolute_import


from .Ctrainer import CTrainer

__factory = {
             'cccseg': CTrainer,
}

def names():
    return sorted(__factory.keys())

def create_trainer(name, *args, **kwargs):
    """
    Create a dataset instance.
    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'pitts', 'tokyo'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown trainer:", name)
    return __factory[name]( *args, **kwargs)
