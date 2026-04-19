# SC4001 — Clothing Classification on Fashion-MNIST

Neural Networks and Deep Learning Group Project (NTU, AY2025/26)

---

## Overview

This project investigates deep learning architectures for clothing classification on the Fashion-MNIST dataset (60,000 train / 10,000 test, 10 classes, 28x28 grayscale images).

Four architectures are compared:
- **BaselineCNN** — standard two-block CNN
- **DilatedCNN** — CNN with dilated convolutions (dilation > 1) in the second block
- **SE-DilatedCNN** — DilatedCNN + Squeeze-and-Excitation channel attention *(novelty)*
- **SimplViT** — lightweight Vision Transformer (patch-based)

Three augmentation strategies are evaluated: MixUp, CutMix, and label smoothing. An ablation study on dilation rates (d = 1, 2, 3) validates the optimal configuration. Grad-CAM and attention rollout visualisations provide model interpretability.

---

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.0+. Runs on CPU, MPS (Apple Silicon), or CUDA — device is selected automatically.

---

## Reproducing All Results

### Option 1 — Run everything at once

```bash
bash run_all.sh          # trains all main experiments
bash run_ablation.sh     # trains dilation ablation d=1,2,3
```

Both scripts automatically regenerate all figures on completion.

### Option 2 — Run a single experiment

```bash
python main.py --model baseline --epochs 30
python main.py --model dilated  --epochs 30 --mixup
python main.py --model se_dilated --epochs 30
python main.py --model se_dilated --epochs 30 --cutmix
python main.py --model vit      --epochs 30
python main.py --model baseline --epochs 30 --mixup --label_smoothing 0.1
```

Full argument reference:

| Argument | Default | Description |
|---|---|---|
| `--model` | `baseline` | `baseline`, `dilated`, `se_dilated`, or `vit` |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `64` | Mini-batch size |
| `--lr` | `1e-3` | Initial learning rate (cosine annealed) |
| `--mixup` | off | Enable MixUp augmentation |
| `--mixup_alpha` | `0.4` | MixUp Beta distribution parameter |
| `--cutmix` | off | Enable CutMix augmentation |
| `--cutmix_alpha` | `1.0` | CutMix Beta distribution parameter |
| `--label_smoothing` | `0.0` | Label smoothing factor |
| `--dilation` | `2` | Dilation rate for DilatedCNN / SE-DilatedCNN (1/2/3) |

### Option 3 — Regenerate figures from saved results

```bash
python plot_results.py
```

### Grad-CAM visualisations

```bash
python generate_gradcam.py
```

### Model complexity summary

```bash
python model_summary.py
```

---

## Project Structure

```
SC4001_final_project/
├── main.py                  # training entry point
├── plot_results.py          # figure generation
├── generate_gradcam.py      # Grad-CAM / attention rollout visualisation
├── model_summary.py         # parameter count & inference speed
├── run_all.sh               # runs all main experiments
├── run_ablation.sh          # runs dilation ablation (d=1,2,3)
├── requirements.txt
├── data/                    # Fashion-MNIST (auto-downloaded)
├── models/                  # saved best checkpoints (.pth)
├── results/
│   ├── *_results.json       # per-experiment metrics
│   └── figures/             # all generated plots
└── src/
    ├── dataset.py           # data loading & augmentation
    ├── models.py            # BaselineCNN, DilatedCNN, SE_DilatedCNN, SimplViT
    ├── train.py             # training & evaluation loops
    ├── mixup.py             # MixUp augmentation
    ├── cutmix.py            # CutMix augmentation
    ├── metrics.py           # confusion matrix, accuracy, F1
    └── gradcam.py           # Grad-CAM and attention rollout utilities
```

---

## Results Summary

| Model | Augmentation | Test Acc | Macro F1 |
|---|---|---|---|
| **SE-DilatedCNN d=2** | **None** | **92.82%** | **0.9280** |
| BaselineCNN | None | 92.49% | 0.9247 |
| DilatedCNN d=2 | None | 92.45% | 0.9244 |
| SimplViT | None | 91.12% | 0.9106 |
| SE-DilatedCNN d=2 | MixUp + LS 0.1 | 91.86% | 0.9178 |
| BaselineCNN | MixUp + LS 0.1 | 91.78% | 0.9172 |
| DilatedCNN d=2 | MixUp | 91.62% | 0.9152 |
| SE-DilatedCNN d=2 | MixUp | 91.56% | 0.9151 |
| BaselineCNN | MixUp | 91.31% | 0.9116 |
| SimplViT | MixUp | 91.28% | 0.9124 |
| DilatedCNN d=2 | CutMix | 90.50% | 0.9037 |
| BaselineCNN | CutMix | 90.34% | 0.9023 |
| SE-DilatedCNN d=2 | CutMix | 90.30% | 0.9021 |

**Dilation ablation (DilatedCNN, no augmentation):**

| Dilation | Test Acc | Macro F1 |
|---|---|---|
| d=1 | **92.62%** | **0.9260** |
| d=2 | 92.45% | 0.9244 |
| d=3 | 92.50% | 0.9248 |

---

## Dataset

Fashion-MNIST is downloaded automatically via TorchVision on first run. No manual setup required.

Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
