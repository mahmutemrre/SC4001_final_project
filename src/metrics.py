"""
Evaluation metrics: confusion matrix, per-class accuracy, and per-class F1.
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


def per_class_f1(cm):
    """Return per-class F1 score and macro-F1 from a confusion matrix.

    Precision_i = TP_i / (TP_i + FP_i) = cm[i,i] / cm[:,i].sum()
    Recall_i    = TP_i / (TP_i + FN_i) = cm[i,i] / cm[i,:].sum()
    F1_i        = 2 * P_i * R_i / (P_i + R_i)
    """
    tp = cm.diagonal().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(cm.sum(axis=0) > 0, tp / cm.sum(axis=0), 0.0)
        recall    = np.where(cm.sum(axis=1) > 0, tp / cm.sum(axis=1), 0.0)
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    macro_f1 = f1.mean()
    return f1, macro_f1
