import os
import glob
import re
import random
import numpy as np
from PIL import Image
from tqdm import tqdm


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def scan_dataset_classes(root_dir, split_folder='val'):
    """Scan masks in split_folder to discover all unique label values."""
    print(f"\n{'='*10} Scanning Masks for Labels {'='*10}")
    masks = (glob.glob(os.path.join(root_dir, split_folder, '*', 'label', '*.png')) +
             glob.glob(os.path.join(root_dir, split_folder, '*', 'label', '*.jpg')))

    unique = set()
    for p in tqdm(masks, desc='Scanning labels'):
        try:
            unique.update(np.unique(np.array(Image.open(p))).tolist())
        except Exception:
            continue

    sorted_labels = sorted(unique)
    print(f'Found labels: {sorted_labels}')
    label_mapping = {orig: idx for idx, orig in enumerate(sorted_labels)}
    return sorted_labels, label_mapping, len(sorted_labels)


def calculate_dataset_statistics(root_dir, split_folder='train', sample_rate=1.0):
    print(f"\n{'='*10} Calculating Dataset Statistics {'='*10}")
    imgs = (glob.glob(os.path.join(root_dir, split_folder, '*', 'ct', '*.png')) +
            glob.glob(os.path.join(root_dir, split_folder, '*', 'ct', '*.jpg')))

    if not imgs:
        return 0.5, 0.5

    if sample_rate < 1.0:
        random.shuffle(imgs)
        imgs = imgs[:int(len(imgs) * sample_rate)]

    ch_sum = ch_sq = px = 0.0
    for p in tqdm(imgs, desc='Computing Mean/Std'):
        try:
            a = np.array(Image.open(p).convert('L')) / 255.0
            ch_sum += np.sum(a)
            ch_sq  += np.sum(a ** 2)
            px     += a.size
        except Exception:
            continue

    mean = ch_sum / px
    std  = float(np.sqrt(ch_sq / px - mean ** 2))
    print(f'Stats -> Mean: {mean:.4f}, Std: {std:.4f}')
    return float(mean), std
