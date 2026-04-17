#!/usr/bin/env bash
# run_ablation.sh — ablation study on dilation rate (d=1, 2, 3), no MixUp.
# Isolates the effect of dilation by keeping everything else fixed.
# Usage:  bash run_ablation.sh [--epochs 30]

set -e

EPOCHS=${2:-30}

echo "======================================================"
echo " Dilation Rate Ablation Study"
echo " Model: DilatedCNN | MixUp: OFF | Epochs: $EPOCHS"
echo "======================================================"

for D in 1 3; do
    echo ""
    echo ">>> DilatedCNN  dilation=$D"
    python main.py --model dilated --dilation $D --epochs "$EPOCHS"
done
# d=2 result reused from main experiments (dilated_nomixup_results.json → dilated_d2_nomixup_results.json)

echo ""
echo "======================================================"
echo " Ablation complete. Generating plots..."
echo "======================================================"
python plot_results.py

echo ""
echo "Done! Ablation results in ./results/  |  Figures in ./results/figures/"
