# SC4001 Fashion-MNIST Clothing Classification Project

## Overview
- **Course**: SC4001 Neural Networks and Deep Learning (NTU)
- **Topic**: E — Clothing Classification (Fashion-MNIST)
- **Deadline**: 19 April 2026, 11:59 PM
- **Grade weight**: 30% of course grade
- **Team**: Erdem Ugurlu, Mahmut Emre Boga

## Grading Criteria
- Project execution: 30%
- Experiments & results: 30%
- Report presentation: 15%
- Novelty: 15%
- Peer review: 10% (via Eureka)

## Deliverables
- 10-page PDF report (Arial 10, excluding refs/cover/content page/appendix)
- Code in .zip file, well-commented, easy to test

## Project Structure
```
src/models.py      — Model architectures (BaselineCNN, DilatedCNN, SE_DilatedCNN, SimplViT)
src/dataset.py     — Fashion-MNIST data loading & transforms
src/train.py       — Training & evaluation loops
src/mixup.py       — MixUp augmentation
src/cutmix.py      — CutMix augmentation
src/metrics.py     — Confusion matrix & per-class accuracy
src/gradcam.py     — Grad-CAM visualization utilities
main.py            — CLI entry point for training experiments
plot_results.py    — Generate all result figures from JSON
model_summary.py   — Print model parameters & inference speed
run_all.sh         — Run all main experiments
run_ablation.sh    — Dilation rate ablation study
```

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run a single experiment
python main.py --model baseline --epochs 30
python main.py --model dilated --dilation 2 --epochs 30 --mixup
python main.py --model se_dilated --epochs 30 --cutmix
python main.py --model vit --epochs 30

# Run all experiments
bash run_all.sh --epochs 30

# Run dilation ablation
bash run_ablation.sh --epochs 30

# Generate plots (after experiments)
python plot_results.py

# Model summary
python model_summary.py
```

## Models
1. **BaselineCNN** — 2-block CNN baseline (Conv+BN+ReLU+MaxPool+Dropout)
2. **DilatedCNN** — Same structure but Block 2 uses dilated convolutions (parameterizable d=1,2,3)
3. **SE_DilatedCNN** — DilatedCNN + Squeeze-and-Excitation channel attention (novelty)
4. **SimplViT** — Lightweight Vision Transformer (patch_size=4, depth=6, embed_dim=128)

## Key Techniques
- **Dilated Convolutions**: Widen receptive field without losing spatial resolution
- **SE Attention**: Channel attention to learn which feature channels are important
- **MixUp**: Soft label augmentation (Zhang et al., 2018)
- **CutMix**: Region-based augmentation (Yun et al., 2019)
- **Label Smoothing**: Regularization via soft targets
- **Grad-CAM**: Model interpretability visualization

## Existing Results (Mahmut Emre's runs, 30 epochs)
| Model | Test Acc | Macro F1 |
|-------|----------|----------|
| DilatedCNN (no MixUp) | 92.76% | 0.9275 |
| BaselineCNN (no MixUp) | 92.73% | 0.9271 |
| DilatedCNN + MixUp | 92.05% | 0.9199 |
| BaselineCNN + MixUp | 91.74% | 0.9166 |
| SimplViT (no MixUp) | 91.06% | 0.9103 |
| SimplViT + MixUp | 91.02% | 0.9098 |

Ablation: d=1 (92.61%), d=2 (92.72%), d=3 (92.34%)

## Progress Tracker
- [x] Project scaffold (models, training loop, data loading)
- [x] MixUp implementation
- [x] Plotting and metrics code
- [x] F1 score metrics (Mahmut Emre)
- [x] Main experiments run — 6 runs complete (Mahmut Emre)
- [x] Dilation ablation run — d=1,2,3 (Mahmut Emre)
- [x] Bug fixes (tag ordering in main.py, is_ablation logic in plot_results.py)
- [x] SE-Net attention model
- [x] CutMix augmentation
- [x] Label smoothing support
- [x] Grad-CAM visualization code
- [ ] Run SE-Net + CutMix + label smoothing experiments
- [ ] Generate all plots (with fixed plotting code)
- [ ] Generate Grad-CAM visualizations
- [ ] Write 10-page report

## Important Notes
- Professor says: open-source libs OK if cited, but must train from scratch and add own innovation
- 10-page limit excludes references, cover page, content page, and appendix
- Appendix is optional supplementary material
- Fashion-MNIST normalization: mean=0.2860, std=0.3530
- Device priority: CUDA > MPS > CPU
