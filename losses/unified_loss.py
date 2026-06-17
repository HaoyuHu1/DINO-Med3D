import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedLoss(nn.Module):
    """CE + Dice loss for multi-class segmentation."""

    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.ce          = nn.CrossEntropyLoss(ignore_index=255)

    def _dice(self, logits, targets):
        probs      = torch.softmax(logits, dim=1)
        valid      = (targets != 255)
        tgt_c      = targets.clone()
        tgt_c[~valid] = 0

        if probs.dim() == 5:
            oh = F.one_hot(tgt_c, self.num_classes).permute(0, 4, 1, 2, 3).float()
        else:
            oh = F.one_hot(tgt_c, self.num_classes).permute(0, 3, 1, 2).float()

        total = 0.0
        cnt   = 0
        for i in range(1, self.num_classes):
            p = probs[:, i, ...][valid]
            t = oh[:, i, ...][valid]
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            if t.sum() == 0:
                d = 1.0 if p.sum() < 10.0 else (2*inter + self.smooth) / (union + self.smooth)
            else:
                d = (2 * inter + self.smooth) / (union + self.smooth)
            total += d
            cnt   += 1

        return 1.0 - total / max(cnt, 1)

    def forward(self, logits, targets):
        l_ce   = self.ce(logits.float(), targets)
        l_dice = self._dice(logits.float(), targets)
        return 0.2 * l_ce + 0.8 * l_dice, l_ce, l_dice
