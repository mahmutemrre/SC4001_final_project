# Changelog: Erdem's Contributions

Changes made on top of Mahmut Emre's last commit (`2032a5c` — "ablation study is done, f1 scores and readme eklendi").

---

## Bugs Found & Fixed

### Bug 1: `is_ablation` logic excludes dilated models from main plots — `plot_results.py`

**Problem:** The `is_ablation()` function checked if a tag has any dilation value. Since ALL dilated experiments include dilation in their tag (e.g., `dilated_d2_nomixup`), `main_experiments()` filtered out ALL dilated model results from the main comparison plots. The accuracy curves, bar chart, per-class accuracy, and F1 plots only showed 4 of 6 experiments (baseline + vit), completely missing the dilated model.

```python
# Mahmut's version — broken
def is_ablation(tag):
    _, dilation, _ = parse_tag(tag)
    return dilation is not None   # True for ALL dilated experiments!

def main_experiments(data):
    return {t: v for t, v in data.items() if not is_ablation(t)}
```

**Fix:** Replaced the heuristic with an explicit `MAIN_TAGS` set. Ablation function now explicitly requires `model == "dilated"` and `aug == "nomixup"`.

```python
# Fixed version
MAIN_TAGS = {
    "baseline_nomixup", "baseline_mixup",
    "dilated_d2_nomixup", "dilated_d2_mixup",
    "vit_nomixup", "vit_mixup",
    "se_dilated_d2_nomixup", "se_dilated_d2_mixup",
    "baseline_cutmix", "dilated_d2_cutmix", "se_dilated_d2_cutmix",
}

def main_experiments(data):
    return {t: v for t, v in data.items() if t in MAIN_TAGS}

def ablation_experiments(data):
    abl = {}
    for t, v in data.items():
        model, dilation, aug = parse_tag(t)
        if model == "dilated" and dilation is not None and aug == "nomixup":
            abl[t] = v
    return abl
```

### Bug 2: `parse_tag` cannot handle two-word model names — `plot_results.py`

**Problem:** `parse_tag()` assumed the model name is always the first underscore-separated token (`parts[0]`). This meant any model with a two-word name (e.g., `se_dilated`) would be incorrectly parsed as model=`se`.

```python
# Mahmut's version
def parse_tag(tag):
    parts = tag.split("_")
    model = parts[0]   # "se_dilated_d2_nomixup" → model="se" (WRONG)
```

**Fix:** Added detection for `se_dilated` prefix and `cutmix` augmentation keyword.

```python
# Fixed version
def parse_tag(tag):
    parts = tag.split("_")
    if parts[0] == "se" and len(parts) > 1 and parts[1] == "dilated":
        model = "se_dilated"
        rest = parts[2:]
    else:
        model = parts[0]
        rest = parts[1:]
    # ...
```

### Bug 3: Missing `weights_only=True` in `torch.load()` — `main.py`

**Problem:** `torch.load(...)` called without `weights_only=True`, producing a deprecation warning in PyTorch 2.6+.

**Fix:** Added `weights_only=True` parameter.

---

## New Features Added

### 1. SE-Net Attention Model (Novelty #1) — `src/models.py`

Added Squeeze-and-Excitation (SE) channel attention mechanism (Hu et al., 2018):

- **`SEBlock`** class: Global average pooling -> FC(C, C/r) -> ReLU -> FC(C/r, C) -> Sigmoid -> channel-wise multiplication. Default reduction ratio r=16.
- **`SE_DilatedCNN`** class: DilatedCNN architecture with SE blocks inserted after each convolutional block. Adds only 768 extra parameters (871,786 total vs 871,018 for DilatedCNN).
- Registered as `"se_dilated"` in `MODEL_MAP` in both `main.py` and `model_summary.py`.

**Result:** SE_DilatedCNN achieved the highest test accuracy of **92.82%**, surpassing all other models.

### 2. CutMix Augmentation (Novelty #2) — `src/cutmix.py` (new file)

Implemented CutMix (Yun et al., 2019):

- **`rand_bbox()`**: Generates a random bounding box whose area ratio matches the mixing parameter.
- **`cutmix_data()`**: Cuts a rectangular region from one image and pastes it onto another. Returns the same interface as `mixup_data()` for compatibility.
- Lambda is adjusted to the actual pasted area ratio after boundary clipping.

Added `--cutmix` and `--cutmix_alpha` arguments to `main.py`. Made `--mixup` and `--cutmix` mutually exclusive.

### 3. CutMix Support in Training Loop — `src/train.py`

Extended `train_one_epoch()` with CutMix path:
- Added `use_cutmix` and `cutmix_alpha` parameters
- CutMix uses the same `mixup_criterion` (weighted cross-entropy) for loss computation
- Three training paths: standard, MixUp, CutMix

### 4. Label Smoothing (Novelty #3) — `main.py`

Added `--label_smoothing` argument (default 0.0). Passed directly to `nn.CrossEntropyLoss(label_smoothing=...)`. One-line feature that enables systematic regularization experiments.

Label smoothing suffix is appended to the experiment tag (e.g., `se_dilated_d2_mixup_ls010`).

### 5. Grad-CAM Visualization — `src/gradcam.py` + `generate_gradcam.py` (new files)

Model interpretability through visual explanations:

- **`GradCAM` class** (`src/gradcam.py`): Implements Selvaraju et al. (2017). Hooks into the last convolutional layer, computes gradient-weighted class activation maps.
- **`attention_rollout()`** (`src/gradcam.py`): For ViT models — multiplies attention matrices across all transformer layers and extracts the CLS token's attention over image patches.
- **`get_target_layer()`**: Automatically finds the last Conv2d layer in any CNN model.
- **`generate_gradcam.py`**: Loads best checkpoints for all 4 models, picks one sample per class, generates a grid visualization (rows=samples, columns=original + one heatmap per model).
- Uses `torch.nn.functional.interpolate` instead of cv2 to avoid OpenCV dependency.

**ViT attention fix:** PyTorch's `TransformerEncoderLayer` calls attention with `need_weights=False` internally. Fixed by monkey-patching the forward method to force `need_weights=True` during rollout computation.

---

## Updated Files

| File | Changes |
|------|---------|
| `main.py` | Added SE model to MODEL_MAP, CutMix + label smoothing args, mutual exclusion check, `weights_only=True` |
| `src/models.py` | Added `SEBlock` (~20 lines) and `SE_DilatedCNN` (~50 lines) |
| `src/cutmix.py` | **New file** — CutMix augmentation (52 lines) |
| `src/train.py` | Added CutMix path in `train_one_epoch()` (+12 lines) |
| `src/gradcam.py` | **New file** — GradCAM class + attention rollout (134 lines) |
| `generate_gradcam.py` | **New file** — Grad-CAM visualization grid generator (166 lines) |
| `plot_results.py` | Fixed `is_ablation` bug, fixed `parse_tag`, added SE/CutMix to palette, updated ablation filter |
| `model_summary.py` | Added SE_DilatedCNN to model list |
| `run_all.sh` | Expanded from 6 to 11 experiments (SE model + CutMix runs) |
| `run_ablation.sh` | Minor cleanup |
| `CLAUDE.md` | **New file** — Project tracking and documentation |

---

## Experiment Results

15 experiments run on Google Colab (Tesla T4 GPU), 30 epochs each:

| # | Experiment | Test Acc | Macro F1 |
|---|---|---|---|
| 1 | **SE_DilatedCNN (no aug)** | **92.82%** | **0.9280** |
| 2 | DilatedCNN d=1 (no aug) | 92.62% | 0.9260 |
| 3 | DilatedCNN d=3 (no aug) | 92.50% | 0.9248 |
| 4 | BaselineCNN (no aug) | 92.49% | 0.9247 |
| 5 | DilatedCNN d=2 (no aug) | 92.45% | 0.9244 |
| 6 | SE_Dilated + MixUp + LS=0.1 | 91.86% | 0.9178 |
| 7 | Baseline + MixUp + LS=0.1 | 91.78% | 0.9172 |
| 8 | DilatedCNN + MixUp | 91.62% | 0.9152 |
| 9 | SE_Dilated + MixUp | 91.56% | 0.9151 |
| 10 | BaselineCNN + MixUp | 91.31% | 0.9116 |
| 11 | SimplViT + MixUp | 91.28% | 0.9124 |
| 12 | SimplViT (no aug) | 91.12% | 0.9106 |
| 13 | DilatedCNN + CutMix | 90.50% | 0.9037 |
| 14 | BaselineCNN + CutMix | 90.34% | 0.9023 |
| 15 | SE_Dilated + CutMix | 90.30% | 0.9021 |

**22 figures generated:** accuracy/loss curves, best accuracy bar chart, per-class accuracy, per-class F1 scores, 15 confusion matrices, dilation ablation, Grad-CAM grid.
