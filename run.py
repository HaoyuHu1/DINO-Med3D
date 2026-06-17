#!/usr/bin/env python3
# dinomed3d run config
# Edit flags below, then: python run.py

# ============================================================
#  Select stages (True / False)
# ============================================================
RUN_STAGE1 = False
RUN_STAGE2 = False
RUN_TEST   = True
# ============================================================

# Optional overrides
MODEL   = "base"   # "small" | "base" | "large"
DATASET = None      # None = use configs/base.py default
DEBUG   = False     # True = fast debug (2 epoch, 2 batch)

import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
os.chdir(_DIR)

import gc, torch, wandb
from configs.base import Config
from datasets import Stage1Dataset, Stage2Dataset, collate_fn_3d
from datasets.utils import scan_dataset_classes, calculate_dataset_statistics
from models import Stage1Model, Stage2Model
from engine import run_training_stage, inference_and_save
from utils.misc import seed_everything
from torch.utils.data import DataLoader


def main():
    cfg = Config()
    cfg.update_model(MODEL)
    if DATASET:
        cfg.dataset_root = DATASET
        cfg.work_dir = "work_dirs/{}_{}".format(
            cfg.model_type, os.path.basename(cfg.dataset_root.rstrip("/")))
    if DEBUG:
        cfg.debug     = True
        cfg.s1_epochs = 2
        cfg.s2_epochs = 2

    os.makedirs(cfg.work_dir, exist_ok=True)
    seed_everything(cfg.seed)
    print(">>> Model={}  Dataset={}".format(cfg.model_type, cfg.dataset_root))
    print(">>> WorkDir={}".format(cfg.work_dir))
    print(">>> Stage1={}  Stage2={}  Test={}  Debug={}".format(
        RUN_STAGE1, RUN_STAGE2, RUN_TEST, DEBUG))

    labels, mapping, n_cls = scan_dataset_classes(cfg.dataset_root, "val")
    cfg.original_labels = labels
    cfg.label_mapping   = mapping
    cfg.num_classes     = n_cls
    print(">>> Classes: {}  Mapping: {}".format(n_cls, mapping))

    # Set WANDB_MODE=offline (or disabled) if you do not use Weights & Biases.
    wandb.init(project="dinomed3d",
               name="{}_{}".format(cfg.model_type, os.path.basename(cfg.work_dir)),
               config=cfg.__dict__, mode=os.environ.get("WANDB_MODE", "online"))

    mean, std    = 0.5, 0.5
    s1_best_path = os.path.join(cfg.work_dir, "stage1_best.pth")

    # ── Stage 1 ──
    if RUN_STAGE1:
        mean, std = calculate_dataset_statistics(cfg.dataset_root, "train")
        ds_tr = Stage1Dataset(cfg, cfg.dataset_root, "training",   mean=mean, std=std)
        ds_vl = Stage1Dataset(cfg, cfg.dataset_root, "validation", mean=mean, std=std)
        ld_tr = DataLoader(ds_tr, batch_size=cfg.s1_batch_size,
                           shuffle=True, num_workers=8, pin_memory=True)
        ld_vl = DataLoader(ds_vl, batch_size=cfg.s1_batch_size,
                           shuffle=False, num_workers=4)
        model_s1 = Stage1Model(cfg, pretrained=True).to(cfg.device)
        s1_best_path = run_training_stage(
            cfg, 1, model_s1, ld_tr, ld_vl,
            cfg.s1_epochs, cfg.s1_lr, "stage1_best.pth",
            norm_stats={"mean": mean, "std": std})
        del model_s1, ld_tr, ld_vl
        gc.collect(); torch.cuda.empty_cache()

    # ── Stage 2 ──
    if RUN_STAGE2:
        if not RUN_STAGE1 and os.path.exists(s1_best_path):
            ck = torch.load(s1_best_path, map_location="cpu")
            if "norm_stats" in ck and ck["norm_stats"]:
                mean, std = ck["norm_stats"]["mean"], ck["norm_stats"]["std"]
            if "label_mapping" in ck:
                cfg.label_mapping = ck["label_mapping"]
                cfg.num_classes   = ck["num_classes"]
        ds_tr = Stage2Dataset(cfg, cfg.dataset_root, "train",
                              clip_depth=cfg.s2_clip_depth, mean=mean, std=std)
        ds_vl = Stage2Dataset(cfg, cfg.dataset_root, "val", mean=mean, std=std)
        ld_tr = DataLoader(ds_tr, batch_size=cfg.s2_batch_size, shuffle=True,
                           num_workers=4, collate_fn=collate_fn_3d)
        ld_vl = DataLoader(ds_vl, batch_size=1, shuffle=False,
                           num_workers=2, collate_fn=collate_fn_3d)
        model_s2 = Stage2Model(cfg, stage1_weights_path=s1_best_path).to(cfg.device)
        run_training_stage(
            cfg, 2, model_s2, ld_tr, ld_vl,
            cfg.s2_epochs, cfg.s2_lr, "stage2_best.pth",
            norm_stats={"mean": mean, "std": std})

    # ── Test ──
    if RUN_TEST:
        s2_path = os.path.join(cfg.work_dir, "stage2_best.pth")
        s1_path = os.path.join(cfg.work_dir, "stage1_best.pth")
        if os.path.exists(s2_path):
            w_path, stage = s2_path, 2
        elif os.path.exists(s1_path):
            w_path, stage = s1_path, 1
        else:
            print("!!! No weights found, skipping test.")
            wandb.finish(); return
        ck = torch.load(w_path, map_location="cpu")
        if "norm_stats" in ck and ck["norm_stats"]:
            mean, std = ck["norm_stats"]["mean"], ck["norm_stats"]["std"]
        if "label_mapping" in ck:
            cfg.label_mapping = ck["label_mapping"]
            cfg.num_classes   = ck["num_classes"]
        test_ds = Stage2Dataset(cfg, cfg.dataset_root, "test", mean=mean, std=std)
        test_ld = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=2, collate_fn=collate_fn_3d)
        if stage == 2:
            model = Stage2Model(cfg, stage1_weights_path="").to(cfg.device)
        else:
            model = Stage1Model(cfg, pretrained=False).to(cfg.device)
        sd = ck.get("model_state_dict", ck)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        out_dir = os.path.join(cfg.work_dir, "predict_test_stage{}".format(stage))
        inference_and_save(cfg, model, test_ld, out_dir, stage_num=stage)
        print(">>> Test complete.")

    wandb.finish()
    print(">>> All done.")


if __name__ == "__main__":
    main()
