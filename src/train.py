"""
Training and evaluation loop.
"""
import torch
import torch.nn as nn
from src.mixup import mixup_data, mixup_criterion


def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=False, mixup_alpha=0.4):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha)
            logits = model(x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(x)
            loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += pred.eq(y if not use_mixup else y_a).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total
