"""
plot_results.py — generate all figures from saved experiment JSON files.

Usage:
    python plot_results.py                  # reads ./results/*.json
    python plot_results.py --out_dir figs   # save to ./figs/
"""
import argparse
import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.dataset import CLASSES

# ── colour palette (one per model) ──────────────────────────────────────────
PALETTE = {
    "baseline":   "#4C72B0",
    "dilated":    "#DD8452",
    "se_dilated": "#C44E52",
    "vit":        "#55A868",
}
ABLATION_PALETTE = {1: "#f4a261", 2: "#e76f51", 3: "#a8201a"}
LINESTYLES = {"nomixup": "-", "mixup": "--", "cutmix": ":"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir="./results"):
    data = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_results.json"))):
        tag = os.path.basename(path).replace("_results.json", "")
        with open(path) as f:
            data[tag] = json.load(f)
    return data


def parse_tag(tag):
    """Parse experiment tag into (model, dilation, aug).

    Examples:
      'baseline_nomixup'        → ('baseline', None, 'nomixup')
      'dilated_d2_nomixup'      → ('dilated', 2, 'nomixup')
      'se_dilated_d2_mixup'     → ('se_dilated', 2, 'mixup')
      'baseline_cutmix'         → ('baseline', None, 'cutmix')
    """
    parts = tag.split("_")

    # Detect model name (may be two parts like "se_dilated")
    if parts[0] == "se" and len(parts) > 1 and parts[1] == "dilated":
        model = "se_dilated"
        rest = parts[2:]
    else:
        model = parts[0]
        rest = parts[1:]

    dilation_str = next((p for p in rest if p.startswith("d") and p[1:].isdigit()), None)
    dilation = int(dilation_str[1:]) if dilation_str else None

    # Augmentation is the last known keyword (strip any label-smoothing suffix)
    aug = "nomixup"
    for p in rest:
        if p in ("mixup", "nomixup", "cutmix"):
            aug = p
    return model, dilation, aug


# Tags produced by run_all.sh (the main model comparison experiments)
MAIN_TAGS = {
    "baseline_nomixup", "baseline_mixup",
    "dilated_d2_nomixup", "dilated_d2_mixup",
    "vit_nomixup", "vit_mixup",
    "se_dilated_d2_nomixup", "se_dilated_d2_mixup",
    "baseline_cutmix", "dilated_d2_cutmix", "se_dilated_d2_cutmix",
}


def main_experiments(data):
    """Return only the main comparison experiments."""
    return {t: v for t, v in data.items() if t in MAIN_TAGS}


def ablation_experiments(data):
    """Return only the dilation ablation experiments (d=1, d=2, d=3).
    Must be DilatedCNN (not SE), no augmentation."""
    abl = {}
    for t, v in data.items():
        model, dilation, aug = parse_tag(t)
        if model == "dilated" and dilation is not None and aug == "nomixup":
            abl[t] = v
    return abl


def readable_label(tag):
    model, dilation, aug = parse_tag(tag)
    d_str = f" d={dilation}" if dilation else ""
    return f"{model}{d_str}\n({aug})"


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Training & test accuracy curves (main experiments only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_curves(data, out_dir):
    data = main_experiments(data)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Accuracy curves — all experiments", fontsize=13, fontweight="bold")

    for tag, res in data.items():
        model, _, aug = parse_tag(tag)
        h = res["history"]
        epochs = range(1, len(h["train_acc"]) + 1)
        label = f"{model} ({aug})"
        color = PALETTE.get(model, "grey")
        ls = LINESTYLES.get(aug, "-")

        axes[0].plot(epochs, h["train_acc"], color=color, linestyle=ls, label=label)
        axes[1].plot(epochs, h["test_acc"],  color=color, linestyle=ls, label=label)

    for ax, title in zip(axes, ["Train accuracy", "Test accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    save(fig, out_dir, "accuracy_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Training & test loss curves (main experiments only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(data, out_dir):
    data = main_experiments(data)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Loss curves — all experiments", fontsize=13, fontweight="bold")

    for tag, res in data.items():
        model, _, aug = parse_tag(tag)
        h = res["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        label = f"{model} ({aug})"
        color = PALETTE.get(model, "grey")
        ls = LINESTYLES.get(aug, "-")

        axes[0].plot(epochs, h["train_loss"], color=color, linestyle=ls, label=label)
        axes[1].plot(epochs, h["test_loss"],  color=color, linestyle=ls, label=label)

    for ax, title in zip(axes, ["Train loss", "Test loss"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    save(fig, out_dir, "loss_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Best test-accuracy bar chart (main experiments only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_acc_bar(data, out_dir):
    data = main_experiments(data)
    tags   = list(data.keys())
    values = [data[t]["best_test_acc"] for t in tags]
    colors = [PALETTE.get(parse_tag(t)[0], "grey") for t in tags]
    labels = [readable_label(t) for t in tags]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(min(values) - 0.02, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Best test accuracy by model & augmentation", fontweight="bold")
    ax.set_ylabel("Test accuracy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, out_dir, "best_accuracy_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Per-class accuracy grouped bar chart (main experiments only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class(data, out_dir):
    data = main_experiments(data)
    tags   = list(data.keys())
    n_tags = len(tags)
    n_cls  = len(CLASSES)
    x = np.arange(n_cls)
    width = 0.8 / n_tags

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, tag in enumerate(tags):
        model, _, aug = parse_tag(tag)
        pca = list(data[tag]["per_class_accuracy"].values())
        offset = (i - n_tags / 2 + 0.5) * width
        ax.bar(x + offset, pca, width,
               label=f"{model} ({aug})",
               color=PALETTE.get(model, "grey"),
               alpha=0.85 if aug == "nomixup" else 0.55,
               edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Per-class accuracy")
    ax.set_title("Per-class accuracy across all models", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, out_dir, "per_class_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Confusion matrices (one per experiment)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(data, out_dir):
    for tag, res in data.items():
        if "confusion_matrix" not in res:
            continue
        cm = np.array(res["confusion_matrix"])
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        model, dilation, aug = parse_tag(tag)
        d_str = f" (d={dilation})" if dilation else ""
        ax.set_title(f"Confusion matrix — {model}{d_str} ({aug})", fontweight="bold")

        for i in range(10):
            for j in range(10):
                val = cm_norm[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        fig.tight_layout()
        save(fig, out_dir, f"cm_{tag}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Per-class F1 score grouped bar chart (main experiments only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_f1_scores(data, out_dir):
    data = main_experiments(data)
    if not any("per_class_f1" in v for v in data.values()):
        return  # old results without F1

    tags   = list(data.keys())
    n_tags = len(tags)
    n_cls  = len(CLASSES)
    x = np.arange(n_cls)
    width = 0.8 / n_tags

    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    fig.suptitle("Per-class F1 scores & Macro F1 comparison", fontsize=13, fontweight="bold")

    # Top: per-class F1 grouped bars
    for i, tag in enumerate(tags):
        if "per_class_f1" not in data[tag]:
            continue
        model, _, aug = parse_tag(tag)
        f1_vals = list(data[tag]["per_class_f1"].values())
        offset = (i - n_tags / 2 + 0.5) * width
        axes[0].bar(x + offset, f1_vals, width,
                    label=f"{model} ({aug})",
                    color=PALETTE.get(model, "grey"),
                    alpha=0.85 if aug == "nomixup" else 0.55,
                    edgecolor="white")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("Per-class F1 score")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Bottom: macro F1 bar chart
    macro_tags   = [t for t in tags if "macro_f1" in data[t]]
    macro_values = [data[t]["macro_f1"] for t in macro_tags]
    macro_colors = [PALETTE.get(parse_tag(t)[0], "grey") for t in macro_tags]
    macro_labels = [readable_label(t) for t in macro_tags]

    bars = axes[1].bar(macro_labels, macro_values, color=macro_colors,
                       edgecolor="white", width=0.5)
    for bar, val in zip(bars, macro_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(min(macro_values) - 0.02, 1.0)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[1].set_title("Macro F1 score by model & augmentation")
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, out_dir, "f1_scores.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Dilation ablation — accuracy vs dilation rate
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation(data, out_dir):
    abl = ablation_experiments(data)
    if not abl:
        return  # no ablation data yet

    # sort by dilation rate
    sorted_tags = sorted(abl.keys(), key=lambda t: parse_tag(t)[1])
    dilation_rates = [parse_tag(t)[1] for t in sorted_tags]
    acc_values     = [abl[t]["best_test_acc"] for t in sorted_tags]
    colors         = [ABLATION_PALETTE.get(d, "grey") for d in dilation_rates]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Ablation Study: Effect of Dilation Rate on DilatedCNN", fontsize=13, fontweight="bold")

    # Left: bar chart of best accuracy per dilation rate
    bars = axes[0].bar([f"d={d}" for d in dilation_rates], acc_values,
                       color=colors, edgecolor="white", width=0.4)
    for bar, val in zip(bars, acc_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylim(min(acc_values) - 0.01, max(acc_values) + 0.015)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].set_title("Best test accuracy vs dilation rate")
    axes[0].set_xlabel("Dilation rate")
    axes[0].set_ylabel("Test accuracy")
    axes[0].grid(axis="y", alpha=0.3)

    # Right: test accuracy curves per dilation rate
    for tag, d in zip(sorted_tags, dilation_rates):
        h = abl[tag]["history"]
        epochs = range(1, len(h["test_acc"]) + 1)
        axes[1].plot(epochs, h["test_acc"],
                     color=ABLATION_PALETTE.get(d, "grey"),
                     label=f"d={d}", linewidth=2)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[1].set_title("Test accuracy curves by dilation rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test accuracy")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save(fig, out_dir, "ablation_dilation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--out_dir",     default="./results/figures")
    args = parser.parse_args()

    data = load_results(args.results_dir)
    if not data:
        print(f"No *_results.json files found in {args.results_dir}. "
              "Run main.py first.")
        return

    print(f"Loaded {len(data)} experiment(s): {list(data.keys())}")

    plot_accuracy_curves(data, args.out_dir)
    plot_loss_curves(data, args.out_dir)
    plot_best_acc_bar(data, args.out_dir)
    plot_per_class(data, args.out_dir)
    plot_f1_scores(data, args.out_dir)
    plot_confusion_matrices(data, args.out_dir)
    plot_ablation(data, args.out_dir)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
