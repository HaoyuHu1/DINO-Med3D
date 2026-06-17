import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utils.misc import expand_to_25d_batch
from utils.metrics import calculate_dice_numpy, calculate_hd95_medpy
from utils.inference import sliding_window_inference


def load_prediction_from_disk(patient_dir, spatial_shape, reverse_mapping):
    """Load PNG slices from disk and rebuild 3D label volume."""
    try:
        files = sorted([f for f in os.listdir(patient_dir) if f.endswith('.png')])
        if len(files) != spatial_shape[0]:
            return None
        pixel_to_label = {v: k for k, v in reverse_mapping.items()}
        slices = []
        for f in files:
            arr   = np.array(Image.open(os.path.join(patient_dir, f)))
            lbl   = np.zeros_like(arr, dtype=np.uint8)
            for pv, li in pixel_to_label.items():
                lbl[arr == pv] = li
            slices.append(lbl)
        return np.stack(slices, axis=0)
    except Exception as e:
        print(f'Error loading prediction: {e}')
        return None


def inference_and_save(cfg, model, test_loader, save_dir, stage_num=2):
    print(f"\n{'='*10} Inference (stage {stage_num}) {'='*10}")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    reverse_mapping = {v: k for k, v in cfg.label_mapping.items()}
    class_names     = [k for k, v in sorted(cfg.label_mapping.items(), key=lambda x: x[1])]
    metrics_list    = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Inference')

        for batch_idx, (imgs, masks, pids) in enumerate(pbar):
            if cfg.debug and batch_idx >= 2:
                break

            # --- check cached predictions ---
            need_inference    = False
            batch_preds_np    = []

            for i, pid in enumerate(pids):
                pdir     = os.path.join(save_dir, str(pid))
                sp_shape = masks[i].shape[-3:]
                loaded   = load_prediction_from_disk(pdir, sp_shape, reverse_mapping) if os.path.exists(pdir) else None
                if loaded is None:
                    need_inference = True
                    break
                batch_preds_np.append(loaded)

            # --- run model if needed ---
            if need_inference:
                batch_preds_np = []
                if stage_num == 1:
                    imgs_vol   = imgs.to(cfg.device)
                    imgs_25d   = expand_to_25d_batch(imgs_vol)
                    bs         = cfg.s1_batch_size
                    logits_lst = []
                    with torch.amp.autocast('cuda', enabled=False):
                        for i in range(0, imgs_25d.shape[0], bs):
                            logits_lst.append(model(imgs_25d[i: i + bs]))
                    logits = torch.cat(logits_lst, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
                else:
                    ws     = cfg.s2_infer_window
                    D_cur  = imgs.shape[2]
                    with torch.amp.autocast('cuda', enabled=False):
                        if D_cur <= ws:
                            pd_ = ws - D_cur
                            inp = F.pad(imgs, (0,0,0,0,0,pd_)) if pd_ else imgs
                            logits = model(inp.to(cfg.device))
                            if pd_: logits = logits[..., :D_cur, :, :]
                        else:
                            logits = sliding_window_inference(
                                model, imgs, cfg, window_size=ws).to(cfg.device)

                pred_t     = torch.argmax(torch.softmax(logits.float(), dim=1), dim=1)
                batch_preds_np = pred_t.cpu().numpy()

            # --- metrics + save ---
            gt_np = masks.cpu().numpy()

            for i, pid in enumerate(pids):
                pred_np_i = (batch_preds_np[i] if isinstance(batch_preds_np, list)
                             else batch_preds_np[i])
                gt_np_i   = gt_np[i]
                spacing   = cfg.voxel_spacing
                if pred_np_i.ndim == 2 and len(spacing) == 3:
                    spacing = spacing[1:]

                row = {'PID': str(pid)}
                for cls_idx, cls_name in enumerate(class_names):
                    pm = (pred_np_i == cls_idx).astype(bool)
                    gm = (gt_np_i   == cls_idx).astype(bool)
                    row[f'Dice_{cls_name}'] = calculate_dice_numpy(pm, gm)
                    row[f'HD95_{cls_name}'] = calculate_hd95_medpy(pm, gm, spacing)
                metrics_list.append(row)

                # save slices
                pdir = os.path.join(save_dir, str(pid))
                if not os.path.exists(pdir) or not need_inference:
                    os.makedirs(pdir, exist_ok=True)
                    mapped = np.vectorize(reverse_mapping.get)(pred_np_i).astype(np.uint8)
                    if mapped.ndim == 3:
                        for d in range(mapped.shape[0]):
                            Image.fromarray(mapped[d]).save(os.path.join(pdir, f'pred_{d:03d}.png'))
                    else:
                        Image.fromarray(mapped).save(os.path.join(pdir, 'pred.png'))

    # --- summary ---
    print(f"\n{'='*10} Test Report (Stage {stage_num}) {'='*10}")
    if not metrics_list:
        print('No metrics.')
        return

    df      = pd.DataFrame(metrics_list)
    summary = []
    fg_dice = []
    fg_hd95 = []

    for ci, cn in enumerate(class_names):
        dk = f'Dice_{cn}'
        hk = f'HD95_{cn}'
        dm = df[dk].mean() if dk in df else float('nan')
        ds = df[dk].std()  if dk in df else float('nan')
        hm = df[hk].mean() if hk in df else float('nan')
        hs = df[hk].std()  if hk in df else float('nan')
        summary.append({'Class': cn,
                        'Dice (Mean±Std)': f'{dm:.4f}±{ds:.4f}' if np.isfinite(dm) else 'N/A',
                        'HD95 (Mean±Std)': f'{hm:.4f}±{hs:.4f}' if np.isfinite(hm) else 'N/A'})
        if ci != 0:
            if np.isfinite(dm): fg_dice.append(dm)
            if np.isfinite(hm): fg_hd95.append(hm)

    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nFG Avg Dice: {np.mean(fg_dice):.4f}" if fg_dice else "\nFG Avg Dice: N/A")
    print(f"FG Avg HD95: {np.mean(fg_hd95):.4f}" if fg_hd95 else "FG Avg HD95: N/A")
    df.to_csv(os.path.join(cfg.work_dir, 'test_metrics.csv'), index=False)
    print(f'Saved metrics -> {cfg.work_dir}/test_metrics.csv')
