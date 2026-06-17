import os
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF

from .stage1_dataset import Stage1Dataset
from .transforms import Stage2MedicalTransform


class Stage2Dataset(Dataset):
    """3D volume dataset."""

    def __init__(self, cfg, root_dir, split='train', clip_depth=16, mean=0.5, std=0.5):
        self.cfg        = cfg
        self.split      = split
        self.clip_depth = clip_depth

        mode       = 'train' if split == 'train' else 'val'
        self.transform = Stage2MedicalTransform(cfg, mode=mode, size=cfg.img_size,
                                                mean=mean, std=std)

        s1_split_map = {'train': 'training', 'val': 'validation', 'test': 'test'}
        s1_split = s1_split_map.get(split, split)
        temp     = Stage1Dataset(cfg, root_dir, split=s1_split, mean=0.5, std=0.5)

        self.volumes = defaultdict(list)
        self.masks_map = defaultdict(list)
        for img, msk, pid in zip(temp.images, temp.masks, temp.patient_ids):
            self.volumes[pid].append(img)
            self.masks_map[pid].append(msk)

        self.patient_list = list(self.volumes.keys())
        self.samples      = []

        if self.split == 'train':
            stride = max(1, clip_depth)
            for pid in self.patient_list:
                n = len(self.volumes[pid])
                if n <= clip_depth:
                    self.samples.append((pid, 0))
                else:
                    for s in range(0, n, stride):
                        a = min(s, n - clip_depth)
                        if not self.samples or self.samples[-1] != (pid, a):
                            self.samples.append((pid, a))
            print(f'>>> Stage2Dataset ({split}): {len(self.samples)} clips, ' +
                  f'{len(self.patient_list)} patients')

    def __len__(self):
        return len(self.samples) if self.split == 'train' else len(self.patient_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            pid, start = self.samples[idx]
            img_paths  = self.volumes[pid]
            msk_paths  = self.masks_map[pid]
            n          = len(img_paths)
            if n < self.clip_depth:
                img_clip = img_paths
                msk_clip = msk_paths
                pad_d    = self.clip_depth - n
            else:
                img_clip = img_paths[start: start + self.clip_depth]
                msk_clip = msk_paths[start: start + self.clip_depth]
                pad_d    = 0
        else:
            pid      = self.patient_list[idx]
            img_clip = self.volumes[pid]
            msk_clip = self.masks_map[pid]
            pad_d    = 0

        first        = Image.open(img_clip[0])
        w_o, h_o     = first.size
        imgs_list    = []
        masks_list   = []

        for ip, mp in zip(img_clip, msk_clip):
            i_pil = Image.open(ip).convert('L')
            m_pil = Image.open(mp) if os.path.exists(mp) else Image.new('L', (w_o, h_o), 0)
            imgs_list.append(TF.to_tensor(i_pil))
            masks_list.append(torch.as_tensor(np.array(m_pil), dtype=torch.long))

        imgs_t  = torch.stack(imgs_list,  dim=1)   # (1, D, H, W)
        masks_t = torch.stack(masks_list, dim=0)   # (D, H, W)

        if self.split == 'train' and pad_d > 0:
            imgs_t  = F.pad(imgs_t,  (0, 0, 0, 0, 0, pad_d), value=0)
            masks_t = F.pad(masks_t, (0, 0, 0, 0, 0, pad_d), value=0)

        imgs_t, masks_t = self.transform(imgs_t, masks_t)
        return imgs_t, masks_t, pid


def collate_fn_3d(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch])
