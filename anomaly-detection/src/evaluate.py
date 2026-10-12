"""
evaluate.py — Threshold selection, metrics computation, and result reporting.

Metrics: precision, recall, F1, point-adjusted F1, PR-AUC.
"""

import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)

from src.utils import ensure_dir

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Threshold
# ────────────────────────────────────────────────────────────────────────
def select_threshold(
    val_scores: np.ndarray, percentile: float = 99.0
) -> float:
    """Pick threshold as the *percentile*-th value of validation scores."""
    thr = float(np.percentile(val_scores, percentile))
    logger.info(f"Threshold ({percentile}th percentile of val scores): {thr:.6f}")
    return thr


# ────────────────────────────────────────────────────────────────────────
# Point-adjusted F1
# ────────────────────────────────────────────────────────────────────────
def _point_adjust_predictions(
    preds: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """
    Point-adjusted evaluation (Xu et al., 2018):
    If *any* prediction within a contiguous anomaly segment is correct,
    mark the entire segment as correctly detected.
    """
    adjusted = preds.copy()
    in_segment = False
    seg_start = 0

    for i in range(len(labels)):
        if labels[i] == 1 and not in_segment:
            in_segment = True
            seg_start = i
        elif labels[i] == 0 and in_segment:
            in_segment = False
            # Check if any pred in [seg_start, i) is 1
            if adjusted[seg_start:i].any():
                adjusted[seg_start:i] = 1
        # Handle segment that extends to the end
        if in_segment and i == len(labels) - 1:
            if adjusted[seg_start : i + 1].any():
                adjusted[seg_start : i + 1] = 1

    return adjusted


# ────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────
def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    model_name: str = "model",
) -> dict:
    """
    Compute anomaly-detection metrics for a single model.

    Returns dict with precision, recall, f1, pa_f1, pr_auc.
    """
    preds = (scores > threshold).astype(int)

    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    # Point-adjusted F1
    pa_preds = _point_adjust_predictions(preds, labels)
    pa_prec = precision_score(labels, pa_preds, zero_division=0)
    pa_rec = recall_score(labels, pa_preds, zero_division=0)
    pa_f1 = f1_score(labels, pa_preds, zero_division=0)

    # PR-AUC (using raw scores, not thresholded)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall_curve, precision_curve)

    metrics = {
        "model": model_name,
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pa_precision": float(pa_prec),
        "pa_recall": float(pa_rec),
        "pa_f1": float(pa_f1),
        "pr_auc": float(pr_auc),
    }
    return metrics


# ────────────────────────────────────────────────────────────────────────
# Reporting
# ────────────────────────────────────────────────────────────────────────
def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table of model metrics."""
    header = f"{'Model':<18} {'Prec':>6} {'Rec':>6} {'F1':>6} {'PA-F1':>6} {'PR-AUC':>7}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(
            f"{r['model']:<18} "
            f"{r['precision']:>6.4f} "
            f"{r['recall']:>6.4f} "
            f"{r['f1']:>6.4f} "
            f"{r['pa_f1']:>6.4f} "
            f"{r['pr_auc']:>7.4f}"
        )
    print(sep + "\n")


def save_results(
    results: list[dict],
    results_dir: str | Path,
) -> Path:
    """Write metrics list to results/metrics.json."""
    results_dir = ensure_dir(results_dir)
    out = results_dir / "metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {out}")
    return out


# ────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────
def plot_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    model_name: str,
    results_dir: str | Path,
) -> Path:
    """Save a scores-vs-labels plot with threshold line."""
    results_dir = ensure_dir(results_dir)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

    ax1.plot(scores, linewidth=0.5, color="#2563eb", label="Anomaly score")
    ax1.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1, label="Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title(f"{model_name} — Anomaly Scores vs Ground Truth")
    ax1.legend(loc="upper right")

    ax2.fill_between(
        range(len(labels)), labels, alpha=0.4, color="#f97316", label="Ground truth"
    )
    preds = (scores > threshold).astype(int)
    ax2.fill_between(
        range(len(preds)), preds, alpha=0.3, color="#7c3aed", label="Predictions"
    )
    ax2.set_ylabel("Anomaly")
    ax2.set_xlabel("Window index")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out = results_dir / f"{model_name.lower().replace(' ', '_')}_scores.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Score plot saved to {out}")
    return out
