# Erdem Ugurlu - Contributions & Bug Fix Report

SC4001 Neural Networks and Deep Learning - Group Project  
Project E: Clothing Classification on Fashion-MNIST  
Date: 19 April 2026

---

## 1. Bugs Found & Fixed

### Bug 1 (Critical): Report numbers did not match actual experiment results — `report.tex`

**Severity:** Critical — academic integrity issue  
**Found in:** Mahmut's report commit (`86dd9b1`)

**Problem:** The LaTeX report contained experiment results that did not match the actual JSON output files. Out of 11 experiments in the original report, only 2 had correct numbers. Examples:

| Experiment | Report claimed | Actual JSON | Difference |
|---|---|---|---|
| SE-DilatedCNN (no aug) | 93.01% | 92.82% | -0.19 pp |
| BaselineCNN (no aug) | 92.73% | 92.49% | -0.24 pp |
| DilatedCNN d=2 (no aug) | 92.72% | 92.45% | -0.27 pp |
| DilatedCNN + CutMix | 90.10% | 90.50% | +0.40 pp |

**Fix:** Replaced all 11 experiment results in the report with the actual values from the JSON files. Added verification script to confirm all 13 experiments match. Every number in the final report is now traceable to its corresponding `*_results.json` file.

---

### Bug 2 (Critical): `is_ablation` logic excluded dilated models from main plots — `plot_results.py`

**Severity:** Critical — main comparison plots were incomplete  
**Found in:** Mahmut's commit `2032a5c`

**Problem:** The `is_ablation()` function checked whether a tag contained any dilation value. Since ALL dilated experiments include dilation in their tag (e.g., `dilated_d2_nomixup`), the `main_experiments()` function filtered out ALL dilated model results. The accuracy curves, bar chart, per-class accuracy, and F1 plots only showed 4 of 6 experiments (baseline + vit), completely omitting the dilated model — one of the project's key contributions.

```python
# Broken code (Mahmut's version)
def is_ablation(tag):
    _, dilation, _ = parse_tag(tag)
    return dilation is not None   # True for ALL dilated experiments!
```

**Fix:** Replaced the heuristic with an explicit `MAIN_TAGS` set that enumerates which experiments belong to the main comparison.

---

### Bug 3 (Medium): `ablation_experiments()` matched wrong experiments — `plot_results.py`

**Severity:** Medium — ablation plot showed incorrect d=2 value  
**Found in:** Our initial fix (still relied on exclusion from MAIN_TAGS)

**Problem:** The ablation filter picked up `se_dilated_d2_mixup_ls010` as the d=2 entry (because it wasn't in MAIN_TAGS and had a dilation value). The ablation bar chart showed d=2 = 91.86% instead of the correct 92.45%.

**Fix:** Made the filter explicit: requires `model == "dilated"` AND `aug == "nomixup"`.

---

### Bug 4 (Medium): `parse_tag` could not handle two-word model names — `plot_results.py`

**Severity:** Medium — blocked SE model integration into plots  
**Found in:** Mahmut's commit `2032a5c`

**Problem:** `parse_tag()` assumed the model name is always the first underscore-separated token (`parts[0]`). The tag `se_dilated_d2_nomixup` was parsed as model=`se` instead of `se_dilated`.

**Fix:** Added detection for the `se_dilated` prefix before parsing the remaining tokens.

---

### Bug 5 (Low): ViT attention rollout crashed — `src/gradcam.py`

**Severity:** Low — Grad-CAM generation failed for ViT model  

**Problem:** PyTorch's `TransformerEncoderLayer` internally calls `self.self_attn(..., need_weights=False)`, so the forward hook received `None` for attention weights, causing `AttributeError: 'NoneType' object has no attribute 'detach'`.

**Fix:** Monkey-patched the `MultiheadAttention.forward` method during rollout to force `need_weights=True`.

---

### Bug 6 (Low): Hardware claim incorrect — `report.tex`

**Problem:** Report stated "Experiments were conducted on Apple MPS hardware" but all experiments were actually run on Google Colab with an NVIDIA Tesla T4 GPU.

**Fix:** Updated to "Google Colab with an NVIDIA Tesla T4 GPU".

---

### Bug 7 (Low): Missing `weights_only=True` — `main.py`

**Problem:** `torch.load()` called without `weights_only=True`, producing deprecation warnings in PyTorch 2.6+.

**Fix:** Added `weights_only=True` parameter.

---

## 2. New Features Added

### Feature 1: SE-Net Channel Attention (Novelty) — `src/models.py`

Added Squeeze-and-Excitation (Hu et al., 2018) channel attention mechanism:
- `SEBlock` class: Global average pooling -> FC bottleneck -> Sigmoid -> channel-wise scaling
- `SE_DilatedCNN` class: DilatedCNN with SE blocks after each conv block
- Only 768 extra parameters (871,786 total vs 871,018 for DilatedCNN)
- **Result: Achieved highest test accuracy of 92.82%**

### Feature 2: CutMix Augmentation — `src/cutmix.py` (new file)

Implemented CutMix (Yun et al., 2019):
- `rand_bbox()`: Random bounding box generation
- `cutmix_data()`: Region-based image mixing with adjusted lambda
- Added `--cutmix` and `--cutmix_alpha` CLI arguments
- Made `--mixup` and `--cutmix` mutually exclusive

### Feature 3: Label Smoothing — `main.py`

- Added `--label_smoothing` argument (default 0.0)
- Passed to `nn.CrossEntropyLoss(label_smoothing=...)`
- Finding: LS=0.1 with MixUp recovers +0.30-0.47 pp over plain MixUp

### Feature 4: Grad-CAM Visualization — `src/gradcam.py` + `generate_gradcam.py` (new files)

Model interpretability through visual explanations:
- `GradCAM` class for CNN models (Selvaraju et al., 2017)
- `attention_rollout()` for ViT models (Abnar & Zuidema, 2020)
- Grid visualization: rows = sample classes, columns = models
- Uses `torch.nn.functional.interpolate` (no OpenCV dependency)

### Feature 5: CutMix Training Loop Support — `src/train.py`

Extended `train_one_epoch()` with CutMix path alongside existing MixUp.

---

## 3. Report Corrections

### Numbers corrected (all 13 experiments now match JSON files):

| Experiment | Old (wrong) | New (correct) |
|---|---|---|
| SE-DilatedCNN (no aug) | 93.01% / 0.9298 | 92.82% / 0.9280 |
| BaselineCNN (no aug) | 92.73% / 0.9271 | 92.49% / 0.9247 |
| DilatedCNN d=2 (no aug) | 92.72% / 0.9270 | 92.45% / 0.9244 |
| DilatedCNN d=2 + MixUp | 91.80% / 0.9174 | 91.62% / 0.9152 |
| SE-Dilated + MixUp | 91.78% / 0.9174 | 91.56% / 0.9151 |
| BaselineCNN + MixUp | 91.74% / 0.9166 | 91.31% / 0.9116 |
| SimplViT (no aug) | 91.06% / 0.9103 | 91.12% / 0.9106 |
| SimplViT + MixUp | 91.02% / 0.9098 | 91.28% / 0.9124 |
| SE-Dilated + CutMix | 90.63% / 0.9055 | 90.30% / 0.9021 |
| BaselineCNN + CutMix | 90.32% / 0.9016 | 90.34% / 0.9023 |
| DilatedCNN + CutMix | 90.10% / 0.8986 | 90.50% / 0.9037 |

### Label smoothing results added to report:

| Experiment | Test Acc | Macro F1 |
|---|---|---|
| SE-DilatedCNN + MixUp + LS 0.1 | 91.86% | 0.9178 |
| BaselineCNN + MixUp + LS 0.1 | 91.78% | 0.9172 |

### Ablation analysis corrected:

Old (wrong): "d=2 is optimal (inverted-U curve)"  
New (correct): "All three rates within 0.17 pp; d=1 marginally best (92.62%); SE attention (+0.37 pp) is the dominant improvement"

---

## 4. Code Quality Improvements

- Added missing docstrings to `models.py` (SEBlock, PatchEmbedding)
- Added missing docstrings to `main.py` (parse_args, main)
- Added missing docstrings to `gradcam.py` (hook methods)
- Updated module-level docstring in `models.py` to include all 4 models
- Added `PALETTE` entry for `se_dilated` and `LINESTYLE` for `cutmix` in `plot_results.py`

---

## 5. Files Modified/Created

| File | Action | Lines changed |
|---|---|---|
| `src/models.py` | Modified | +75 (SEBlock, SE_DilatedCNN, docstrings) |
| `src/cutmix.py` | **Created** | +52 (CutMix augmentation) |
| `src/gradcam.py` | **Created** | +134 (Grad-CAM + attention rollout) |
| `src/train.py` | Modified | +12 (CutMix training path) |
| `generate_gradcam.py` | **Created** | +166 (visualization grid) |
| `main.py` | Modified | +20 (SE model, CutMix, LS args, docstrings) |
| `plot_results.py` | Modified | +30 (bug fixes, SE/CutMix palette) |
| `model_summary.py` | Modified | +5 (added SE model) |
| `run_all.sh` | Modified | +10 (expanded to 11 experiments) |
| `report.tex` | Modified | +40 (corrected numbers, LS, ablation) |
| `report_overleaf/report.tex` | Modified | same as above |
| `CLAUDE.md` | **Created** | +90 (project tracking) |
| `CHANGELOG.md` | **Created** | +120 (change documentation) |

---

## 6. Experiment Results Summary (15 total)

All experiments run on Google Colab (Tesla T4 GPU), 30 epochs, Adam optimizer, cosine annealing LR.

| # | Experiment | Test Acc | Macro F1 |
|---|---|---|---|
| 1 | **SE_DilatedCNN d=2 (no aug)** | **92.82%** | **0.9280** |
| 2 | BaselineCNN (no aug) | 92.49% | 0.9247 |
| 3 | DilatedCNN d=2 (no aug) | 92.45% | 0.9244 |
| 4 | SE_Dilated + MixUp + LS=0.1 | 91.86% | 0.9178 |
| 5 | Baseline + MixUp + LS=0.1 | 91.78% | 0.9172 |
| 6 | DilatedCNN d=2 + MixUp | 91.62% | 0.9152 |
| 7 | SE_Dilated + MixUp | 91.56% | 0.9151 |
| 8 | BaselineCNN + MixUp | 91.31% | 0.9116 |
| 9 | SimplViT + MixUp | 91.28% | 0.9124 |
| 10 | SimplViT (no aug) | 91.12% | 0.9106 |
| 11 | DilatedCNN d=2 + CutMix | 90.50% | 0.9037 |
| 12 | BaselineCNN + CutMix | 90.34% | 0.9023 |
| 13 | SE_Dilated + CutMix | 90.30% | 0.9021 |

**Dilation ablation (DilatedCNN, no augmentation):**

| Dilation | Test Acc | Macro F1 |
|---|---|---|
| d=1 | 92.62% | 0.9260 |
| d=3 | 92.50% | 0.9248 |
| d=2 | 92.45% | 0.9244 |
