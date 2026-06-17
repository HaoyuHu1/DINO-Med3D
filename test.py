#!/usr/bin/env python3
"""
Inference / test entry point for dinomed3d.

Usage:
    python test.py --weights PATH/stage1_best.pth [--stage 1] [--debug]
    python test.py --weights PATH/stage2_best.pth [--stage 2]
"""
import sys
import os
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)
os.chdir(_PROJECT_DIR)

import argparse
import torch
from torch.utils.data import DataLoader

from configs.base import Config
from datasets import Stage2Dataset, collate_fn_3d
from models import Stage1Model, Stage2Model
from engine import inference_and_save
from utils.misc import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='path to .pth checkpoint')
    p.add_argument('--stage',   type=int, default=1, choices=[1, 2])
    p.add_argument('--dataset', default=None)
    p.add_argument('--debug',   action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    if not os.path.exists(args.weights):
        print(f'!!! Weights not found: {args.weights}'); return

    print(f'>>> Loading config from {args.weights}')
    ck = torch.load(args.weights, map_location='cpu')
    if 'norm_stats' in ck and ck['norm_stats']:
        mean = ck['norm_stats']['mean']; std = ck['norm_stats']['std']
    else:
        mean, std = 0.5, 0.5
    if 'label_mapping' in ck:
        cfg.label_mapping   = ck['label_mapping']
        cfg.num_classes     = ck['num_classes']
    if 'model_type' in ck:
        cfg.update_model(ck['model_type'])

    if args.dataset:
        cfg.dataset_root = args.dataset
    if args.debug:
        cfg.debug = True

    # work_dir mirrors checkpoint location
    cfg.work_dir = os.path.dirname(args.weights)
    os.makedirs(cfg.work_dir, exist_ok=True)
    seed_everything(cfg.seed)

    print(f'>>> num_classes={cfg.num_classes}, label_mapping={cfg.label_mapping}')
    print(f'>>> Dataset: {cfg.dataset_root}/test')

    test_ds = Stage2Dataset(cfg, cfg.dataset_root, 'test', mean=mean, std=std)
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=2, collate_fn=collate_fn_3d)

    if args.stage == 2:
        model = Stage2Model(cfg, stage1_weights_path='').to(cfg.device)
    else:
        model = Stage1Model(cfg, pretrained=False).to(cfg.device)

    sd = ck.get('model_state_dict', ck)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    tag      = 'stage1' if args.stage == 1 else 'stage2'
    out_dir  = os.path.join(cfg.work_dir, f'predict_test_{tag}')
    inference_and_save(cfg, model, test_ld, out_dir, stage_num=args.stage)
    print('>>> Inference complete.')


if __name__ == '__main__':
    main()
