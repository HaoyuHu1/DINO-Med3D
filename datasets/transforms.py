import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class MedicalTransform:
    """2D (2.5D) image + mask transform."""

    def __init__(self, cfg, mode='train', size=512, mean=0.5, std=0.5):
        self.cfg  = cfg
        self.mode = mode
        self.size = size
        self.normalize = transforms.Normalize(mean=[mean] * 3, std=[std] * 3)

    def __call__(self, image, mask):
        image = TF.resize(image, (self.size, self.size),
                          interpolation=transforms.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (self.size, self.size),
                          interpolation=transforms.InterpolationMode.NEAREST)

        if self.mode == 'train':
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                scale = random.uniform(0.8, 1.2)
                image = TF.affine(image, angle=angle, translate=(0, 0), scale=scale, shear=0,
                                  interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
                mask  = TF.affine(mask,  angle=angle, translate=(0, 0), scale=scale, shear=0,
                                  interpolation=transforms.InterpolationMode.NEAREST, fill=0)

        image    = TF.to_tensor(image)
        mask_np  = np.array(mask)
        mapped   = np.zeros_like(mask_np)
        for orig, idx in self.cfg.label_mapping.items():
            mapped[mask_np == orig] = idx
        mask_t   = torch.as_tensor(mapped, dtype=torch.long)
        image    = self.normalize(image)
        return image, mask_t


class Stage2MedicalTransform:
    """3D tensor transform."""

    def __init__(self, cfg, mode='train', size=512, mean=0.5, std=0.5):
        self.cfg  = cfg
        self.mode = mode
        self.size = size
        self.mean = mean
        self.std  = std

    def __call__(self, img_t, mask_t):
        img_t  = TF.resize(img_t,  [self.size, self.size],
                           interpolation=transforms.InterpolationMode.BILINEAR,  antialias=True)
        mask_t = TF.resize(mask_t, [self.size, self.size],
                           interpolation=transforms.InterpolationMode.NEAREST, antialias=False)

        if self.mode == 'train':
            if random.random() > 0.5:
                img_t  = TF.hflip(img_t)
                mask_t = TF.hflip(mask_t)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                scale = random.uniform(0.8, 1.2)
                tx    = random.uniform(-0.05, 0.05) * self.size
                ty    = random.uniform(-0.05, 0.05) * self.size
                img_t  = TF.affine(img_t,  angle=angle, translate=(tx, ty), scale=scale, shear=0,
                                   interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
                mask_t = TF.affine(mask_t, angle=angle, translate=(tx, ty), scale=scale, shear=0,
                                   interpolation=transforms.InterpolationMode.NEAREST, fill=0)

        img_t = (img_t - self.mean) / self.std

        mask_clone = mask_t.clone()
        for orig, idx in self.cfg.label_mapping.items():
            mask_t[mask_clone == orig] = idx
        return img_t, mask_t
