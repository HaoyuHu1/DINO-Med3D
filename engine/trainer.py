import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import wandb
from tqdm import tqdm

from losses import UnifiedLoss
from utils.inference import sliding_window_inference


def _write_progress(cfg, stage_num, epoch, num_epochs, avg_loss, mean_dice,
                    best_dice, best_epoch, final_dices, cur_lr):
    """Overwrite progress.log with latest epoch only (keeps file tiny for monitoring)."""
    rev = {v: k for k, v in cfg.label_mapping.items()}
    class_line = "  ".join(
        "C{}(L{})={:.4f}".format(c, rev.get(c, c), d)
        for c, d in sorted(final_dices.items())
    )
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "[{}]  Stage {}  Ep {}/{}".format(ts, stage_num, epoch, num_epochs),
        "Loss  : {:.4f}".format(avg_loss),
        "Dice  : {:.4f}   (best {:.4f} @ Ep {})".format(mean_dice, best_dice, best_epoch),
        "LR    : {:.2e}".format(cur_lr),
        "Detail: {}".format(class_line),
    ]
    path = os.path.join(cfg.work_dir, "progress.log")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def run_training_stage(cfg, stage_num, model, train_loader, val_loader,
                       num_epochs, lr, save_name, norm_stats=None):
    print("\n" + "=" * 20 + " Start Stage {} ".format(stage_num) + "=" * 20)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    patience  = 5 if stage_num == 1 else 8
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience)
    criterion = UnifiedLoss(num_classes=cfg.num_classes).to(cfg.device)
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    best_dice  = -1.0
    best_epoch = 0
    save_path  = os.path.join(cfg.work_dir, save_name)

    for epoch in range(1, num_epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc="[S{}] Ep {}".format(stage_num, epoch))

        for bidx, (imgs, masks, _) in enumerate(pbar):
            if cfg.debug and bidx >= 2:
                break
            imgs, masks = imgs.to(cfg.device), masks.to(cfg.device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                logits = model(imgs)
                loss, l_ce, l_dice = criterion(logits, masks)

            if torch.isnan(loss):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            scaler.scale(loss).backward()
            if cfg.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"Loss": "{:.4f}".format(loss.item()),
                              "LR": "{:.2e}".format(cur_lr)})

        n_batches = len(train_loader) if not cfg.debug else 2
        avg_loss  = train_loss / n_batches

        model.eval()
        patient_metrics = defaultdict(
            lambda: defaultdict(lambda: {"inter": 0.0, "union": 0.0}))

        with torch.no_grad():
            for bidx, (imgs, masks, pids) in enumerate(val_loader):
                if cfg.debug and bidx >= 2:
                    break

                if stage_num == 1:
                    imgs, masks = imgs.to(cfg.device), masks.to(cfg.device)
                    with torch.amp.autocast("cuda", enabled=False):
                        logits = model(imgs)
                else:
                    masks = masks.to(cfg.device, non_blocking=True)
                    ws    = cfg.s2_infer_window
                    D_cur = imgs.shape[2]
                    with torch.amp.autocast("cuda", enabled=False):
                        if D_cur <= ws:
                            pad_d  = ws - D_cur
                            inp    = torch.nn.functional.pad(
                                imgs, (0, 0, 0, 0, 0, pad_d)) if pad_d else imgs
                            logits = model(inp.to(cfg.device))
                            if pad_d:
                                logits = logits[..., :D_cur, :, :]
                        else:
                            logits = sliding_window_inference(
                                model, imgs, cfg, window_size=ws).to(cfg.device)

                pred      = torch.argmax(torch.softmax(logits.float(), dim=1), dim=1)
                pred_cpu  = pred.cpu()
                masks_cpu = masks.long().cpu()

                for i, pid in enumerate(pids):
                    for c in range(1, cfg.num_classes):
                        valid = (masks_cpu[i] != 255)
                        p_c   = (pred_cpu[i]  == c).float() * valid
                        m_c   = (masks_cpu[i] == c).float() * valid
                        patient_metrics[pid][c]["inter"] += (p_c * m_c).sum().item()
                        patient_metrics[pid][c]["union"] += (
                            p_c.sum().item() + m_c.sum().item())

        sums = defaultdict(float)
        cnts = defaultdict(int)
        for pid, cls_data in patient_metrics.items():
            for c, m in cls_data.items():
                d = (2 * m["inter"]) / (m["union"] + 1e-6) if m["union"] > 0 else 1.0
                sums[c] += d
                cnts[c] += 1

        final_dices = {}
        for c in range(1, cfg.num_classes):
            final_dices[c] = sums[c] / cnts[c] if cnts[c] > 0 else 0.0

        mean_dice = float(np.mean(list(final_dices.values()))) if final_dices else 0.0
        print("Ep {} | Loss: {:.4f} | MeanDice: {:.4f} | LR: {:.2e}".format(
            epoch, avg_loss, mean_dice, cur_lr))

        rev = {v: k for k, v in cfg.label_mapping.items()}
        for c, d_val in final_dices.items():
            print("    Class {} (Label {}): {:.4f}".format(c, rev.get(c, c), d_val))

        wandb.log(
            {"S{}/Loss".format(stage_num): avg_loss,
             "S{}/MeanDice".format(stage_num): mean_dice,
             "S{}/LR".format(stage_num): cur_lr},
            step=epoch + (0 if stage_num == 1 else cfg.s1_epochs))
        scheduler.step(mean_dice)

        if mean_dice > best_dice:
            best_dice  = mean_dice
            best_epoch = epoch
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "norm_stats": norm_stats,
                 "num_classes": cfg.num_classes,
                 "label_mapping": cfg.label_mapping}, save_path)
            print(">>> Saved best: {:.4f}".format(best_dice))

        _write_progress(cfg, stage_num, epoch, num_epochs, avg_loss,
                        mean_dice, best_dice, best_epoch, final_dices, cur_lr)

        if stage_num == 1 and cur_lr < cfg.min_lr:
            print("!!! LR below threshold, stopping Stage 1 early.")
            break

    if cfg.debug:
        torch.save(
            {"model_state_dict": model.state_dict(),
             "norm_stats": norm_stats,
             "num_classes": cfg.num_classes,
             "label_mapping": cfg.label_mapping}, save_path)
        print("[Debug] Saved debug checkpoint to {}".format(save_path))

    return save_path
