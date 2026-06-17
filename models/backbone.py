import os
import torch
import torch.nn as nn


class DINOv3BackboneWrapper(nn.Module):
    """Wraps a DINOv3 ViT and exposes multi-scale patch features via hooks."""

    def __init__(self, cfg, repo_dir, checkpoint_path, out_indices, backbone_name,
                 pretrained=True):
        super().__init__()
        self.cfg         = cfg
        self.out_indices = out_indices
        self.embed_dim   = cfg.embed_dim
        self.patch_size  = cfg.s1_patch_size

        print(f'Initializing DINOv3 Backbone: {backbone_name}')
        self.model = torch.hub.load(repo_dir, backbone_name, source='local', pretrained=False)

        if pretrained and os.path.exists(checkpoint_path):
            print(f'Loading backbone weights from {checkpoint_path}')
            sd = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in sd:   sd = sd['model']
            elif 'teacher' in sd: sd = sd['teacher']
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            msg = self.model.load_state_dict(sd, strict=False)
            print(f'Weights loaded. Missing (first 5): {msg.missing_keys[:5]}')

        total = len(self.model.blocks)
        if max(out_indices) >= total:
            raise ValueError(f'out_indices {out_indices} exceeds depth {total}')

        self.hidden_states: dict = {}

        def _hook(name):
            def fn(m, inp, out): self.hidden_states[name] = out
            return fn

        for i in out_indices:
            self.model.blocks[i].register_forward_hook(_hook(f'block_{i}'))

    def forward(self, x):
        B, C, H, W = x.shape
        self.hidden_states = {}
        self.model.forward_features(x)

        h_f = H // self.patch_size
        w_f = W // self.patch_size
        n   = h_f * w_f
        outs = []
        for i in self.out_indices:
            raw = self.hidden_states[f'block_{i}']
            if isinstance(raw, (list, tuple)): raw = raw[0]
            tokens = raw[:, -n:, :]
            feat   = tokens.permute(0, 2, 1).reshape(B, self.embed_dim, h_f, w_f)
            outs.append(feat)
        return outs
