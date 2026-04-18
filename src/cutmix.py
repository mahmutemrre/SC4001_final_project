"""
CutMix augmentation (Yun et al., 2019).

Instead of linearly blending whole images (like MixUp), CutMix cuts a
rectangular region from one image and pastes it onto another.  The label
is mixed proportionally to the area of the pasted region.
"""
import torch
import numpy as np


def rand_bbox(size, lam):
    """Return a random bounding box whose area is (1 - lam) * image_area."""
    _, _, H, W = size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    # centre of the box (uniform random)
    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    return y1, x1, y2, x2


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix to a batch.

    Returns (mixed_x, y_a, y_b, lam) with the same interface as mixup_data
    so the same mixup_criterion can be reused.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    y1, x1, y2, x2 = rand_bbox(x.size(), lam)
    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to the actual area ratio (box may be clipped at borders)
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (x.size(-2) * x.size(-1)))

    return x_cut, y, y[index], lam
