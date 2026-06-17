import numpy as np

try:
    from medpy.metric.binary import hd95
    MEDPY_AVAILABLE = True
except ImportError:
    MEDPY_AVAILABLE = False


def calculate_dice_numpy(pred, gt):
    p = pred > 0
    g = gt   > 0
    s = np.sum(p) + np.sum(g)
    if s == 0:
        return 1.0
    return 2.0 * np.sum(p & g) / s


def calculate_hd95_medpy(pred, gt, spacing=(1.0, 1.0, 1.0)):
    p = pred > 0
    g = gt   > 0
    if np.sum(p) == 0 and np.sum(g) == 0:
        return 0.0
    if np.sum(p) == 0 or np.sum(g) == 0:
        return np.nan
    if not MEDPY_AVAILABLE:
        return np.nan
    try:
        return hd95(p, g, voxelspacing=spacing)
    except Exception:
        return np.nan
