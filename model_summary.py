"""
model_summary.py — print parameter counts and inference speed for all models.

Usage:
    python model_summary.py
"""
import time
import torch
from src.models import BaselineCNN, DilatedCNN, SimplViT

MODELS = [
    ("BaselineCNN",    BaselineCNN()),
    ("DilatedCNN d=1", DilatedCNN(dilation=1)),
    ("DilatedCNN d=2", DilatedCNN(dilation=2)),
    ("DilatedCNN d=3", DilatedCNN(dilation=3)),
    ("SimplViT",       SimplViT()),
]

BATCH = torch.randn(64, 1, 28, 28)
WARMUP = 5
RUNS   = 20

print(f"\n{'Model':<18} {'Params':>10}  {'Inference (ms/batch)':>22}")
print("-" * 54)

for name, model in MODELS:
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    # warm-up
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(BATCH)

    # timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(RUNS):
            _ = model(BATCH)
    elapsed_ms = (time.perf_counter() - start) / RUNS * 1000

    print(f"{name:<18} {params:>10,}  {elapsed_ms:>19.2f} ms")

print()
