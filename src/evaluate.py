"""
Evaluation module.
Implements Phase 4 & Phase 8 evaluation metrics from Classification_Plan.md:
ROC AUC (primary), PR AUC, Brier Score, Lift/Recall@Top-K, and fold metrics summaries.
"""

from typing import Dict, Any, List
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.config import METRICS_DIR


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        f"precision@{threshold:.2f}": float(precision),
        f"recall@{threshold:.2f}": float(recall),
        f"f1@{threshold:.2f}": float(f1),
    }

    # Lift and Recall at Top 5%, 10%, 20%
    for top_pct in [0.05, 0.10, 0.20]:
        k = int(np.ceil(len(y_true) * top_pct))
        top_k_indices = np.argsort(y_prob)[::-1][:k]
        actual_positives_in_top_k = y_true[top_k_indices].sum()
        total_positives = y_true.sum()
        base_rate = total_positives / len(y_true) if len(y_true) > 0 else 0

        recall_top_k = float(actual_positives_in_top_k / total_positives) if total_positives > 0 else 0.0
        precision_top_k = float(actual_positives_in_top_k / k) if k > 0 else 0.0
        lift_top_k = float(precision_top_k / base_rate) if base_rate > 0 else 0.0

        metrics[f"recall@top_{int(top_pct*100)}pct"] = recall_top_k
        metrics[f"lift@top_{int(top_pct*100)}pct"] = lift_top_k

    return metrics


def summarize_cv_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Summarize fold-level metrics: calculate mean and std across folds.
    """
    keys = fold_metrics[0].keys()
    summary = {}
    for key in keys:
        values = [fm[key] for fm in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    summary["n_folds"] = len(fold_metrics)
    summary["fold_details"] = fold_metrics
    return summary


def save_metrics_report(
    model_name: str,
    summary: Dict[str, Any],
    output_dir: Path = METRICS_DIR
) -> Path:
    """
    Save validation metrics summary as a JSON file in reports/metrics/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{model_name}_metrics.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return file_path
