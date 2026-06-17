import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv3BackboneWrapper


class Stage1Model(nn.Module):
    """2.5D segmentation model: DINOv3 backbone + linear head."""

    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg      = cfg
        self.backbone = DINOv3BackboneWrapper(
            cfg,
            repo_dir=cfg.repo_dir,
            checkpoint_path=cfg.checkpoint_path,
            out_indices=cfg.out_indices,
            backbone_name=cfg.backbone_name,
            pretrained=pretrained,
        )
        self.head = nn.Conv2d(cfg.embed_dim, cfg.num_classes, kernel_size=1)
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        feats  = self.backbone(x)
        logits = self.head(feats[-1])
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
