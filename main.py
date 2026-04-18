"""
main.py — run experiments for SC4001 Fashion-MNIST project.

Usage:
    python main.py --model baseline --epochs 30
    python main.py --model dilated  --epochs 30 --mixup
    python main.py --model se_dilated --epochs 30 --cutmix
    python main.py --model vit      --epochs 30 --mixup
"""
import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn

from src.dataset import get_dataloaders, CLASSES
from src.models import BaselineCNN, DilatedCNN, SE_DilatedCNN, SimplViT
from src.train import train_one_epoch, evaluate
from src.metrics import compute_confusion_matrix, per_class_accuracy, per_class_f1


MODEL_MAP = {
    "baseline":   BaselineCNN,
    "dilated":    DilatedCNN,
    "se_dilated": SE_DilatedCNN,
    "vit":        SimplViT,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="baseline", choices=MODEL_MAP.keys())
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    # Augmentation flags (mutually exclusive: mixup OR cutmix, not both)
    p.add_argument("--mixup",       action="store_true", help="Enable MixUp augmentation")
    p.add_argument("--mixup_alpha", type=float, default=0.4)
    p.add_argument("--cutmix",      action="store_true", help="Enable CutMix augmentation")
    p.add_argument("--cutmix_alpha", type=float, default=1.0)
    # Regularisation
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing factor (0 = hard labels)")
    # Architecture
    p.add_argument("--dilation",    type=int,   default=2,
                   help="Dilation rate for DilatedCNN / SE_DilatedCNN block-2")
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--save_dir",    default="./models")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mixup and args.cutmix:
        raise ValueError("Cannot use both --mixup and --cutmix at the same time.")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)

    # Instantiate model
    if args.model in ("dilated", "se_dilated"):
        model = MODEL_MAP[args.model](dilation=args.dilation).to(device)
    else:
        model = MODEL_MAP[args.model]().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # Build experiment tag (used for checkpoint & results filenames)
    dilation_suffix = f"_d{args.dilation}" if args.model in ("dilated", "se_dilated") else ""
    if args.mixup:
        aug_tag = "mixup"
    elif args.cutmix:
        aug_tag = "cutmix"
    else:
        aug_tag = "nomixup"
    ls_suffix = f"_ls{args.label_smoothing:.2f}".replace(".", "") if args.label_smoothing > 0 else ""
    tag = f"{args.model}{dilation_suffix}_{aug_tag}{ls_suffix}"

    print(f"Experiment: {tag}")
    print(f"Model: {args.model} | Epochs: {args.epochs} | LR: {args.lr}")

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=args.mixup, mixup_alpha=args.mixup_alpha,
            use_cutmix=args.cutmix, cutmix_alpha=args.cutmix_alpha,
        )
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"Test loss {te_loss:.4f} acc {te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), f"{args.save_dir}/{tag}_best.pth")

    # -------------------------------------------------------------------------
    # Post-training: confusion matrix & per-class accuracy on the best checkpoint
    # -------------------------------------------------------------------------
    model.load_state_dict(torch.load(f"{args.save_dir}/{tag}_best.pth",
                                     map_location=device, weights_only=True))
    cm  = compute_confusion_matrix(model, test_loader, num_classes=10, device=device)
    pca = per_class_accuracy(cm)
    f1, macro_f1 = per_class_f1(cm)

    results = {
        "tag": tag,
        "model": args.model,
        "augmentation": aug_tag,
        "label_smoothing": args.label_smoothing,
        "history": history,
        "best_test_acc": best_acc,
        "macro_f1": float(f"{macro_f1:.4f}"),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": {cls: float(f"{acc:.4f}")
                                for cls, acc in zip(CLASSES, pca)},
        "per_class_f1": {cls: float(f"{s:.4f}")
                         for cls, s in zip(CLASSES, f1)},
    }

    out_path = f"./results/{tag}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest test accuracy : {best_acc:.4f}  |  Macro F1: {macro_f1:.4f}")
    print(f"{'Class':<15s} {'Accuracy':>8}  {'F1':>8}")
    print("-" * 35)
    for cls, acc, score in zip(CLASSES, pca, f1):
        print(f"  {cls:<13s} {acc:>8.4f}  {score:>8.4f}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
