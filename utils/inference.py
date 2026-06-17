import torch
import torch.nn.functional as F


def sliding_window_inference(model, image, cfg, window_size=16, overlap=0.5):
    """Sliding-window 3D inference. image: (B,C,D,H,W). Returns (B,C,D,H,W)."""
    B, C, D, H, W = image.shape
    stride = max(1, int(window_size * (1 - overlap)))

    pred   = torch.zeros((cfg.num_classes, D, H, W), device='cpu')
    count  = torch.zeros((D, H, W), device='cpu')

    start = 0
    while True:
        end    = start + window_size
        if end > D:
            start = max(0, D - window_size)
            end   = D

        crop   = image[:, :, start:end, :, :]
        actual = crop.shape[2]
        pad_d  = window_size - actual
        if pad_d > 0:
            crop = F.pad(crop, (0, 0, 0, 0, 0, pad_d), value=0)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                out = model(crop.to(cfg.device))

        out = out.cpu()
        if pad_d > 0:
            out = out[:, :, :actual, :, :]

        pred[:, start:start + actual, :, :] += out[0]
        count[start:start + actual, :, :]   += 1.0

        if end >= D:
            break
        start += stride

    pred /= count.unsqueeze(0)
    return pred.unsqueeze(0)
