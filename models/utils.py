
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import pdb
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

