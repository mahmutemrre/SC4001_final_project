#!/usr/bin/env bash
# run_all.sh — train all experiment configurations sequentially.
# Usage:  bash run_all.sh [--epochs 30]

set -e

EPOCHS=${2:-30}   # default 30 epochs; override with: bash run_all.sh --epochs 10

echo "======================================================"
echo " SC4001 Fashion-MNIST — running all experiments"
echo " Epochs: $EPOCHS"
echo "======================================================"

run() {
    echo ""
    echo ">>> $@"
    python main.py "$@" --epochs "$EPOCHS"
}

# ── 1. Baseline CNN ──────────────────────────────────────────
run --model baseline
run --model baseline --mixup
run --model baseline --cutmix

# ── 2. Dilated CNN ───────────────────────────────────────────
run --model dilated
run --model dilated --mixup
run --model dilated --cutmix

# ── 3. SE-Dilated CNN (novelty: channel attention) ───────────
run --model se_dilated
run --model se_dilated --mixup
run --model se_dilated --cutmix

# ── 4. Simple ViT ────────────────────────────────────────────
run --model vit
run --model vit --mixup

echo ""
echo "======================================================"
echo " All main experiments complete. Generating plots..."
echo "======================================================"
python plot_results.py

echo ""
echo "Done! Results in ./results/  |  Figures in ./results/figures/"
