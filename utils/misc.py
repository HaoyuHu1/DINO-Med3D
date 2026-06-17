import os
import random
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def expand_to_25d_batch(x):
    """(B,1,D,H,W) -> (B*D, 3, H, W)"""
    d   = x.squeeze(1)
    xp  = F.pad(d, (0, 0, 0, 0, 1, 1), mode='constant', value=0)
    out = torch.stack([xp[:, :-2], xp[:, 1:-1], xp[:, 2:]], dim=1)  # (B,3,D,H,W)
    return out.permute(0, 2, 1, 3, 4).reshape(x.shape[0] * x.shape[2], 3,
                                               x.shape[3], x.shape[4])


def convert_to_onehot(tensor, num_classes):
    """Label tensor (...) -> one-hot (B, C, ...)."""
    tensor = tensor.long()
    res    = torch.nn.functional.one_hot(tensor, num_classes=num_classes)
    if tensor.dim() == 4:
        return res.permute(0, 4, 1, 2, 3).float()
    elif tensor.dim() == 3:
        return res.permute(0, 3, 1, 2).float()
    return res.float()
