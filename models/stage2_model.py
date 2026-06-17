import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv3BackboneWrapper

try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ───── 3-D building blocks ────────────────────────────────────

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch),
            )
        self.conv1 = nn.Conv3d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.InstanceNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.InstanceNorm3d(out_ch)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetHighResEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        b = base_channels
        self.stem          = nn.Sequential(nn.Conv3d(in_channels, b, 3, padding=1, bias=False),
                                           nn.InstanceNorm3d(b), nn.ReLU(inplace=True))
        self.layer1        = ResBlock3D(b,   b)
        self.layer2_down   = ResBlock3D(b,   b * 2, stride=(1, 2, 2))
        self.layer2_refine = ResBlock3D(b*2, b * 2)
        self.layer3_down   = ResBlock3D(b*2, b * 4, stride=(1, 2, 2))
        self.layer3_refine = ResBlock3D(b*4, b * 4)
        self.proj_c3       = nn.Conv3d(b*4, 256, 1)

    def forward(self, x):
        x   = self.stem(x)
        c1  = self.layer1(x)
        c2  = self.layer2_refine(self.layer2_down(c1))
        c3  = self.layer3_refine(self.layer3_down(c2))
        return c1, c2, self.proj_c3(c3)


class GatedFusion3D(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv3d(channels * 2, channels, 1), nn.Sigmoid())

    def forward(self, vit, cnn):
        if cnn.shape[2:] != vit.shape[2:]:
            cnn = F.interpolate(cnn, size=vit.shape[2:], mode='trilinear', align_corners=False)
        return vit + self.gate(torch.cat([vit, cnn], dim=1)) * cnn


class RefinementBlock(nn.Module):
    def __init__(self, in_low, in_high, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_low + in_high, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, low, high):
        up  = F.interpolate(low, size=high.shape[2:], mode='trilinear', align_corners=False)
        return self.conv(torch.cat([up, high], dim=1))


class HighResRefinementDecoder(nn.Module):
    def __init__(self, num_classes, base_channels_cnn=16):
        super().__init__()
        b = base_channels_cnn
        self.r12     = RefinementBlock(256, b*2, 64)
        self.r11     = RefinementBlock(64,  b,   32)
        self.cls_seg = nn.Conv3d(32, num_classes, 1)

    def forward(self, feat, c1, c2):
        return self.cls_seg(self.r11(self.r12(feat, c2), c1))


class Dino2DIntegratedEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg      = cfg
        self.backbone = DINOv3BackboneWrapper(
            cfg,
            repo_dir=cfg.repo_dir,
            checkpoint_path=cfg.checkpoint_path,
            out_indices=cfg.out_indices,
            backbone_name=cfg.backbone_name,
            pretrained=False,
        )

    def _expand_25d(self, x):
        d  = x.squeeze(1)
        xp = F.pad(d, (0, 0, 0, 0, 1, 1), mode='constant', value=0)
        return torch.stack([xp[:, :-2], xp[:, 1:-1], xp[:, 2:]], dim=1)  # (B,3,D,H,W)

    def forward(self, x):
        x3 = self._expand_25d(x)
        B, C, D, H, W = x3.shape
        x2d = x3.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        raw = self.backbone(x2d)
        outs = []
        for f in raw:
            _, cf, hf, wf = f.shape
            outs.append(f.view(B, D, cf, hf, wf).permute(0, 2, 1, 3, 4))
        return outs


class Adapter3D(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        ic, oc = in_channels, out_channels
        self.up4   = nn.Sequential(nn.Conv3d(ic, oc, 1, bias=False), nn.InstanceNorm3d(oc), nn.PReLU(),
                                   nn.ConvTranspose3d(oc, oc, 4, stride=(1, 4, 4), padding=0))
        self.up2   = nn.Sequential(nn.Conv3d(ic, oc, 1, bias=False), nn.InstanceNorm3d(oc), nn.PReLU(),
                                   nn.ConvTranspose3d(oc, oc, 2, stride=(1, 2, 2), padding=0))
        self.id1   = nn.Sequential(nn.Conv3d(ic, oc, 1, bias=False), nn.InstanceNorm3d(oc), nn.PReLU())
        self.down2 = nn.Sequential(nn.Conv3d(ic, oc, 1, bias=False), nn.InstanceNorm3d(oc), nn.PReLU(),
                                   nn.Conv3d(oc, oc, 3, stride=(1, 2, 2), padding=(1, 1, 1)))

    def forward(self, feats):
        return [self.up4(feats[0]), self.up2(feats[1]),
                self.id1(feats[2]), self.down2(feats[3])]


class UPerNet3DDecoder(nn.Module):
    def __init__(self, num_classes=2, in_channels=None, channels=256):
        super().__init__()
        if in_channels is None:
            in_channels = [256, 256, 256, 256]
        ch = channels
        self.ppm      = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool3d(s),
                                                      nn.Conv3d(in_channels[-1], ch // 4, 1),
                                                      nn.GroupNorm(32, ch // 4), nn.PReLU())
                                       for s in [1, 2, 3, 6]])
        self.ppm_last = nn.Sequential(nn.Conv3d(in_channels[-1] + ch, ch, 3, padding=1),
                                       nn.GroupNorm(32, ch), nn.PReLU())
        self.fpn_in   = nn.ModuleList([nn.Sequential(nn.Conv3d(ic, ch, 1),
                                                      nn.GroupNorm(32, ch), nn.PReLU())
                                        for ic in in_channels[:-1]])
        self.fpn_out  = nn.ModuleList([nn.Sequential(nn.Conv3d(ch, ch, 3, padding=1),
                                                       nn.GroupNorm(32, ch), nn.PReLU())
                                        for _ in in_channels[:-1]])
        self.fusion   = nn.Sequential(nn.Conv3d(ch * 4, ch, 3, padding=1),
                                       nn.GroupNorm(32, ch), nn.PReLU(), nn.Dropout(0.1))
        self.cls      = nn.Conv3d(ch, num_classes, 1)

    def forward(self, feats, return_feat=False):
        c4  = feats[-1]
        ppm = [F.interpolate(p(c4), size=c4.shape[2:], mode='trilinear', align_corners=False)
               for p in self.ppm]
        p4  = self.ppm_last(torch.cat([c4] + ppm, dim=1))

        fp   = [p4]
        prev = p4
        for i in range(len(feats) - 2, -1, -1):
            lat  = self.fpn_in[i](feats[i])
            prev = self.fpn_out[i](lat + F.interpolate(prev, size=lat.shape[2:],
                                                        mode='trilinear', align_corners=False))
            fp.append(prev)
        fp.reverse()
        sz  = fp[0].shape[2:]
        cat = torch.cat([fp[0]] + [F.interpolate(f, size=sz, mode='trilinear', align_corners=False)
                                    for f in fp[1:]], dim=1)
        out = self.fusion(cat)
        return out if return_feat else self.cls(out)


# ───── Full Stage 2 Model ─────────────────────────────────────

class Stage2Model(nn.Module):
    def __init__(self, cfg, stage1_weights_path):
        super().__init__()
        self.cfg          = cfg
        self.encoder      = Dino2DIntegratedEncoder(cfg)
        self.cnn_encoder  = ResNetHighResEncoder(in_channels=1, base_channels=16)
        self.adapter_3d   = Adapter3D(in_channels=cfg.embed_dim, out_channels=256)
        self.fusion_mid   = GatedFusion3D(channels=256)
        self.decoder      = UPerNet3DDecoder(num_classes=cfg.num_classes)
        self.decoder_final = HighResRefinementDecoder(num_classes=cfg.num_classes,
                                                       base_channels_cnn=16)
        self._load_stage1(stage1_weights_path)
        if cfg.use_lora and PEFT_AVAILABLE:
            self._apply_lora()
        else:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _load_stage1(self, path):
        if not (path and os.path.exists(path)):
            print('!!! Stage 1 weights not found, using random init !!!')
            return
        print(f'>>> Loading Stage 1 weights from {path}')
        ckpt = torch.load(path, map_location='cpu')
        s1   = ckpt.get('model_state_dict', ckpt)
        s2   = self.state_dict()
        cnt  = 0
        for k, v in s1.items():
            nk = 'encoder.' + k if k.startswith('backbone.') else None
            if nk and nk in s2 and s2[nk].shape == v.shape:
                s2[nk] = v
                cnt += 1
        self.load_state_dict(s2, strict=False)
        print(f'>>> Transferred {cnt} layers from Stage 1')

    def _apply_lora(self):
        print('>>> Applying LoRA to backbone...')
        cfg = self.cfg
        pc  = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                         lora_dropout=cfg.lora_dropout,
                         target_modules=cfg.lora_target, inference_mode=False)
        self.encoder.backbone.model = get_peft_model(self.encoder.backbone.model, pc)
        self.encoder.backbone.model.print_trainable_parameters()

    def forward(self, x):
        c1, c2, c3 = self.cnn_encoder(x)
        feats       = self.adapter_3d(self.encoder(x))
        feats[0]    = self.fusion_mid(feats[0], c3)
        feat_deep   = self.decoder(feats, return_feat=True)
        logits      = self.decoder_final(feat_deep, c1, c2)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='trilinear', align_corners=False)
        return logits
