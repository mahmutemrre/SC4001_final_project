"""
generate_gradcam.py — produce Grad-CAM / attention-rollout visualisations.

Loads best checkpoints for each model and generates heatmap grids for
sample images, highlighting which regions drive the classification.

Usage:
    python generate_gradcam.py                       # default dirs
    python generate_gradcam.py --models_dir ./models  # custom model dir
"""
import argparse
import os
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders, CLASSES
from src.models import BaselineCNN, DilatedCNN, SE_DilatedCNN, SimplViT
from src.gradcam import GradCAM, attention_rollout, get_target_layer

# Unnormalize for display (Fashion-MNIST stats)
MEAN, STD = 0.2860, 0.3530


MODEL_MAP = {
    "baseline":   BaselineCNN,
    "dilated":    DilatedCNN,
    "se_dilated": SE_DilatedCNN,
    "vit":        SimplViT,
}


def load_model(model_name, checkpoint_path, dilation=2):
    """Instantiate and load a model from its checkpoint."""
    if model_name in ("dilated", "se_dilated"):
        model = MODEL_MAP[model_name](dilation=dilation)
    else:
        model = MODEL_MAP[model_name]()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def unnormalize(img_tensor):
    """Convert a normalised (1, 28, 28) tensor back to [0, 1] for display."""
    img = img_tensor.squeeze().cpu().numpy() * STD + MEAN
    return np.clip(img, 0, 1)


def overlay_cam(img, cam, alpha=0.4):
    """Overlay a heatmap on a grayscale image. Returns an RGB array."""
    # Resize cam to image size using torch interpolate (avoids cv2 dependency)
    cam_t = torch.tensor(cam).float().unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
    cam_resized = F.interpolate(cam_t, size=img.shape, mode="bilinear",
                                align_corners=False).squeeze().numpy()
    heatmap = plt.cm.jet(cam_resized)[:, :, :3]  # (H, W, 3)
    img_rgb = np.stack([img] * 3, axis=-1)        # (H, W, 3)
    return np.clip(img_rgb * (1 - alpha) + heatmap * alpha, 0, 1)


def find_checkpoints(models_dir):
    """Find all *_best.pth files and parse model info from filenames.

    Returns dict mapping model_name -> (checkpoint_path, dilation).
    Prefers nomixup + d=2 checkpoints (the best performing configs).
    """
    checkpoints = {}
    for path in sorted(glob.glob(os.path.join(models_dir, "*_best.pth"))):
        basename = os.path.basename(path).replace("_best.pth", "")
        parts = basename.split("_")
        if parts[0] == "se" and len(parts) > 1 and parts[1] == "dilated":
            model_name = "se_dilated"
            rest = parts[2:]
        else:
            model_name = parts[0]
            rest = parts[1:]

        # Parse dilation from tag (e.g., "d2" -> 2)
        dilation = 2  # default
        for p in rest:
            if p.startswith("d") and p[1:].isdigit():
                dilation = int(p[1:])

        is_nomixup = "nomixup" in rest

        if model_name in MODEL_MAP:
            if model_name not in checkpoints:
                checkpoints[model_name] = (path, dilation, is_nomixup)
            else:
                _, prev_dil, prev_nomixup = checkpoints[model_name]
                # Prefer nomixup over augmented, and d=2 for dilated models
                better_aug = is_nomixup and not prev_nomixup
                better_dil = (model_name in ("dilated", "se_dilated")
                              and dilation == 2 and prev_dil != 2)
                if better_aug or (is_nomixup == prev_nomixup and better_dil):
                    checkpoints[model_name] = (path, dilation, is_nomixup)

    # Strip the is_nomixup flag from return values
    return {k: (v[0], v[1]) for k, v in checkpoints.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--out_dir",    default="./results/figures")
    parser.add_argument("--n_samples",  type=int, default=8,
                        help="Number of sample images to visualise")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Find available checkpoints
    checkpoints = find_checkpoints(args.models_dir)
    if not checkpoints:
        print(f"No checkpoints found in {args.models_dir}. Run experiments first.")
        return

    print(f"Found checkpoints for: {list(checkpoints.keys())}")

    # Load test data
    _, test_loader = get_dataloaders(args.data_dir, batch_size=1, num_workers=0, augment=False)

    # Pick diverse sample images (one per class if possible)
    samples = []
    class_seen = set()
    for x, y in test_loader:
        label = y.item()
        if label not in class_seen and len(samples) < args.n_samples:
            samples.append((x, label))
            class_seen.add(label)
        if len(samples) >= args.n_samples:
            break

    n_models = len(checkpoints)
    n_samples = len(samples)

    # Pre-load all models once (avoid reloading per sample)
    loaded_models = {}
    for model_name, (ckpt_path, dilation) in checkpoints.items():
        loaded_models[model_name] = load_model(model_name, ckpt_path, dilation=dilation)

    # Create figure: rows = samples, cols = original + one per model
    fig, axes = plt.subplots(n_samples, n_models + 1,
                              figsize=(3 * (n_models + 1), 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    # Column headers
    axes[0, 0].set_title("Original", fontsize=10, fontweight="bold")
    for j, model_name in enumerate(checkpoints.keys()):
        axes[0, j + 1].set_title(model_name, fontsize=10, fontweight="bold")

    for i, (x, label) in enumerate(samples):
        # Show original image
        img = unnormalize(x)
        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(CLASSES[label], fontsize=9, rotation=0,
                               labelpad=60, va="center")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        for j, model_name in enumerate(checkpoints.keys()):
            model = loaded_models[model_name]

            if model_name == "vit":
                cam = attention_rollout(model, x)
                pred_idx = model(x).argmax(dim=1).item()
            else:
                target_layer = get_target_layer(model)
                gradcam = GradCAM(model, target_layer)
                cam, pred_idx = gradcam(x)

            overlaid = overlay_cam(img, cam)
            axes[i, j + 1].imshow(overlaid)
            pred_label = CLASSES[pred_idx]
            color = "green" if pred_idx == label else "red"
            axes[i, j + 1].set_xlabel(f"pred: {pred_label}", fontsize=8, color=color)
            axes[i, j + 1].set_xticks([]); axes[i, j + 1].set_yticks([])

    fig.suptitle("Grad-CAM / Attention Visualisation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "gradcam_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM grid → {out_path}")


if __name__ == "__main__":
    main()
