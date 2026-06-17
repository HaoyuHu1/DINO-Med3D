#!/usr/bin/env python3
"""
Training entry point for dinomed3d.

Usage:
    python train.py [--model {small,base,large}] [--dataset PATH]
                    [--stage1] [--stage2] [--debug]
"""
import sys
import os
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)
os.chdir(_PROJECT_DIR)

import gc
import argparse
import torch
import wandb

from configs.base import Config
from datasets import Stage1Dataset, Stage2Dataset, collate_fn_3d
from datasets.utils import scan_dataset_classes, calculate_dataset_statistics
from models import Stage1Model, Stage2Model
from engine import run_training_stage
from utils.misc import seed_everything
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   default='base',   choices=['small', 'base', 'large'])
    p.add_argument('--dataset', default=None,     help='override dataset_root')
    p.add_argument('--stage1',  action='store_true', help='run stage 1 training')
    p.add_argument('--stage2',  action='store_true', help='run stage 2 training')
    p.add_argument('--debug',   action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    cfg.update_model(args.model)
    if args.dataset:
        cfg.dataset_root = args.dataset
        import os as _os
        cfg.work_dir = f'work_dirs/{cfg.model_type}_{_os.path.basename(cfg.dataset_root.rstrip("/"))}'
    if args.debug:
        cfg.debug      = True
        cfg.s1_epochs  = 2
        cfg.s2_epochs  = 2
    if args.stage1 or args.stage2:
        cfg.run_stage1 = args.stage1
        cfg.run_stage2 = args.stage2

    os.makedirs(cfg.work_dir, exist_ok=True)
    seed_everything(cfg.seed)
    print(f'>>> Model: {cfg.model_type}, Dataset: {cfg.dataset_root}')
    print(f'>>> Work dir: {cfg.work_dir}')
    print(f'>>> Stage1={cfg.run_stage1}, Stage2={cfg.run_stage2}, Debug={cfg.debug}')

    # scan labels
    labels, mapping, n_cls = scan_dataset_classes(cfg.dataset_root, 'val')
    cfg.original_labels = labels
    cfg.label_mapping   = mapping
    cfg.num_classes     = n_cls
    print(f'>>> Classes: {n_cls}  Mapping: {mapping}')

    # Set WANDB_MODE=offline (or disabled) if you do not use Weights & Biases.
    wandb.init(project='dinomed3d', name=f'{cfg.model_type}_{os.path.basename(cfg.work_dir)}',
               config=cfg.__dict__, mode=os.environ.get('WANDB_MODE', 'online'))

    mean, std     = 0.5, 0.5
    s1_best_path  = os.path.join(cfg.work_dir, 'stage1_best.pth')

    # ── Stage 1 ──────────────────────────────────────────────
    if cfg.run_stage1:
        mean, std = calculate_dataset_statistics(cfg.dataset_root, 'train')

        ds_tr = Stage1Dataset(cfg, cfg.dataset_root, 'training',   mean=mean, std=std)
        ds_vl = Stage1Dataset(cfg, cfg.dataset_root, 'validation', mean=mean, std=std)
        ld_tr = DataLoader(ds_tr, batch_size=cfg.s1_batch_size, shuffle=True,
                           num_workers=8, pin_memory=True)
        ld_vl = DataLoader(ds_vl, batch_size=cfg.s1_batch_size, shuffle=False, num_workers=4)

        model_s1  = Stage1Model(cfg, pretrained=True).to(cfg.device)
        s1_best_path = run_training_stage(
            cfg, 1, model_s1, ld_tr, ld_vl,
            cfg.s1_epochs, cfg.s1_lr, 'stage1_best.pth',
            norm_stats={'mean': mean, 'std': std})

        del model_s1, ld_tr, ld_vl
        gc.collect(); torch.cuda.empty_cache()

    # ── Stage 2 ──────────────────────────────────────────────
    if cfg.run_stage2:
        if not cfg.run_stage1 and os.path.exists(s1_best_path):
            ck = torch.load(s1_best_path, map_location='cpu')
            if 'norm_stats' in ck and ck['norm_stats']:
                mean, std = ck['norm_stats']['mean'], ck['norm_stats']['std']
            if 'label_mapping' in ck:
                cfg.label_mapping = ck['label_mapping']; cfg.num_classes = ck['num_classes']

        ds_tr = Stage2Dataset(cfg, cfg.dataset_root, 'train', clip_depth=cfg.s2_clip_depth,
                              mean=mean, std=std)
        ds_vl = Stage2Dataset(cfg, cfg.dataset_root, 'val',   mean=mean, std=std)
        ld_tr = DataLoader(ds_tr, batch_size=cfg.s2_batch_size, shuffle=True,
                           num_workers=4, collate_fn=collate_fn_3d)
        ld_vl = DataLoader(ds_vl, batch_size=1, shuffle=False,
                           num_workers=2, collate_fn=collate_fn_3d)

        model_s2 = Stage2Model(cfg, stage1_weights_path=s1_best_path).to(cfg.device)
        run_training_stage(
            cfg, 2, model_s2, ld_tr, ld_vl,
            cfg.s2_epochs, cfg.s2_lr, 'stage2_best.pth',
            norm_stats={'mean': mean, 'std': std})

    wandb.finish()
    print('>>> Training complete.')


if __name__ == '__main__':
    main()
