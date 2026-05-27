"""Test-set evaluation: metrics, confusion matrix and ROC curve."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from .config import CLASS_NAMES


@torch.no_grad()
def _collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    for images, labels in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)[:, 1]  # P(PNEUMONIA)
        preds = (probs >= 0.5).long()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        y_score.extend(probs.cpu().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_score)


def _plot_confusion(cm, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_roc(y_true, y_score, path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return roc_auc


def evaluate_model(model, loader, device, model_name: str, out_dir: Path) -> dict:
    """Compute metrics and save confusion-matrix + ROC figures. Returns metrics."""
    y_true, y_pred, y_score = _collect_predictions(model, loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion(cm, out_dir / f"{model_name}_confusion.png",
                    f"{model_name} - confusion matrix")
    roc_auc = _plot_roc(y_true, y_score, out_dir / f"{model_name}_roc.png",
                        f"{model_name} - ROC")

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_true)),
    }
    with open(out_dir / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def write_summary_table(all_metrics: list[dict], out_dir: Path) -> None:
    """Write a comparative summary table (Markdown + JSON) across models."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    header = "| Model | Accuracy | Precision | Recall | Specificity | F1 | AUC |"
    sep = "|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for m in all_metrics:
        rows.append(
            f"| {m['model']} | {m['accuracy']*100:.1f}% | {m['precision']*100:.1f}% | "
            f"{m['recall']*100:.1f}% | {m['specificity']*100:.1f}% | "
            f"{m['f1']*100:.1f}% | {m['auc']:.3f} |"
        )
    table = "\n".join(rows) + "\n"
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Test-set performance summary\n\n" + table)
    print("\n" + table)
