import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

from .utils import natural_key
from .transforms import MedicalTransform


class Stage1Dataset(Dataset):
    """2.5D slice-level dataset."""

    def __init__(self, cfg, root_dir, split='training', mean=0.5, std=0.5):
        self.cfg   = cfg
        split_map  = {'training': ('train', 'train'),
                      'validation': ('val', 'val'),
                      'test': ('test', 'val')}
        split_folder, mode = split_map.get(split, (split, 'val'))

        self.transform = MedicalTransform(cfg, mode=mode, size=cfg.img_size, mean=mean, std=std)

        all_images = (glob.glob(os.path.join(root_dir, split_folder, '*', 'ct', '*.png')) +
                      glob.glob(os.path.join(root_dir, split_folder, '*', 'ct', '*.jpg')))

        self.images = sorted(
            all_images,
            key=lambda x: (os.path.basename(os.path.dirname(os.path.dirname(x))),
                           natural_key(os.path.basename(x)))
        )

        self.masks       = []
        self.patient_ids = []
        for img_path in self.images:
            ct_dir      = os.path.dirname(img_path)
            patient_dir = os.path.dirname(ct_dir)
            fname       = os.path.basename(img_path)
            self.masks.append(os.path.join(patient_dir, 'label', fname))
            self.patient_ids.append(os.path.basename(patient_dir))

    def __len__(self):
        return len(self.images)

    def _load_gray(self, path):
        try:
            return Image.open(path).convert('L')
        except Exception:
            return Image.new('L', (self.cfg.img_size, self.cfg.img_size), 0)

    def __getitem__(self, idx):
        pid       = self.patient_ids[idx]
        img_c     = self._load_gray(self.images[idx])
        w, h      = img_c.size

        img_p = (self._load_gray(self.images[idx - 1])
                 if idx > 0 and self.patient_ids[idx - 1] == pid
                 else Image.new('L', (w, h), 0))
        img_n = (self._load_gray(self.images[idx + 1])
                 if idx < len(self.images) - 1 and self.patient_ids[idx + 1] == pid
                 else Image.new('L', (w, h), 0))

        merged = Image.merge('RGB', (img_p, img_c, img_n))
        mask   = (Image.open(self.masks[idx])
                  if os.path.exists(self.masks[idx])
                  else Image.new('L', (w, h), 0))

        img_t, mask_t = self.transform(merged, mask)
        return img_t, mask_t, pid
