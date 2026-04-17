# SC4001 — Clothing Classification on Fashion-MNIST

Neural Networks and Deep Learning Group Project (NTU, AY2025/26)

---

## Overview

This project investigates deep learning architectures for clothing classification on the Fashion-MNIST dataset (60,000 train / 10,000 test, 10 classes, 28×28 grayscale images).

Three architectures are compared:
- **BaselineCNN** — standard two-block CNN
- **DilatedCNN** — CNN with dilated convolutions (dilation > 1) in the second block *(novelty)*
- **SimplViT** — lightweight Vision Transformer (patch-based)

MixUp augmentation (Zhang et al., 2018) is evaluated on all three models. An ablation study on dilation rates (d = 1, 2, 3) validates the optimal configuration.

---

## Requirements

```bash
pip install torch torchvision numpy matplotlib
```

Tested with Python 3.10+, PyTorch 2.0+. Runs on CPU, MPS (Apple Silicon), or CUDA — device is selected automatically.

---

## Reproducing All Results

### Option 1 — Run everything at once

```bash
bash run_all.sh          # trains all 6 main experiments (≈ 1.5 hrs on MPS)
bash run_ablation.sh     # trains dilation ablation d=1,2,3 (≈ 20 min on MPS)
```

Both scripts automatically regenerate all figures on completion.

### Option 2 — Run a single experiment

```bash
python main.py --model baseline --epochs 30
python main.py --model dilated  --epochs 30 --mixup
python main.py --model vit      --epochs 30
python main.py --model dilated  --epochs 30 --dilation 3   # ablation
```

Full argument reference:

| Argument | Default | Description |
|---|---|---|
| `--model` | `baseline` | `baseline`, `dilated`, or `vit` |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `64` | Mini-batch size |
| `--lr` | `1e-3` | Initial learning rate (cosine annealed) |
| `--mixup` | off | Enable MixUp augmentation |
| `--mixup_alpha` | `0.4` | MixUp Beta distribution parameter |
| `--dilation` | `2` | Dilation rate for DilatedCNN (1/2/3) |

### Option 3 — Regenerate figures from saved results

```bash
python plot_results.py
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
├── model_summary.py         # parameter count & inference speed
├── run_all.sh               # runs all 6 main experiments
├── run_ablation.sh          # runs dilation ablation (d=1,2,3)
├── requirements.txt
├── data/                    # Fashion-MNIST (auto-downloaded)
├── models/                  # saved best checkpoints (.pth)
├── results/
│   ├── *_results.json       # per-experiment metrics
│   └── figures/             # all generated plots
└── src/
    ├── dataset.py           # data loading & augmentation
    ├── models.py            # BaselineCNN, DilatedCNN, SimplViT
    ├── train.py             # training & evaluation loops
    ├── mixup.py             # MixUp augmentation
    └── metrics.py           # confusion matrix, accuracy, F1
```

---

## Results Summary

| Model | Best Test Acc | Macro F1 |
|---|---|---|
| DilatedCNN (no MixUp) | **92.76%** | **0.9275** |
| BaselineCNN (no MixUp) | 92.73% | 0.9271 |
| DilatedCNN + MixUp | 92.05% | 0.9199 |
| BaselineCNN + MixUp | 91.74% | 0.9166 |
| SimplViT (no MixUp) | 91.06% | 0.9103 |
| SimplViT + MixUp | 91.02% | 0.9098 |

**Dilation ablation (DilatedCNN, no MixUp):**

| Dilation | Test Acc | Macro F1 |
|---|---|---|
| d=1 | 92.61% | 0.9258 |
| d=2 | **92.72%** | **0.9270** |
| d=3 | 92.34% | 0.9234 |

---

## Dataset

Fashion-MNIST is downloaded automatically via TorchVision on first run. No manual setup required.

Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
