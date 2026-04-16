#!/usr/bin/env bash
# run_all.sh — train all 6 experiment configurations sequentially.
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

# ── Baseline CNN ──────────────────────────────────────────
run --model baseline
run --model baseline --mixup

# ── Dilated CNN ───────────────────────────────────────────
run --model dilated
run --model dilated --mixup

# ── Simple ViT ────────────────────────────────────────────
run --model vit
run --model vit --mixup

echo ""
echo "======================================================"
echo " All experiments complete. Generating plots..."
echo "======================================================"
python plot_results.py

echo ""
echo "Done! Results in ./results/  |  Figures in ./results/figures/"
