"""
Evaluation metrics: confusion matrix and per-class accuracy.
"""
import torch
import numpy as np


@torch.no_grad()
def compute_confusion_matrix(model, loader, num_classes, device):
    """Return (num_classes x num_classes) confusion matrix as a numpy array."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
            cm[t, p] += 1

    return cm


def per_class_accuracy(cm):
    """Return per-class accuracy array from a confusion matrix."""
    row_sums = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(row_sums > 0, cm.diagonal() / row_sums, 0.0)
    return acc
