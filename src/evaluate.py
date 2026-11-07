"""
Model Evaluation and Diagnostic Reporting Module.
Computes evaluation metrics across multiple thresholds, plots ROC & PR curves,
confusion matrices, and saves evaluation reports.
"""

import os
import json
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

# Style configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Calculate all standard metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, pd.DataFrame]:
    """
    Search grid of thresholds between 0.05 and 0.95 to find threshold maximizing F1-score
    and high-recall business operating points.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    records = []

    for th in thresholds:
        metrics = evaluate_predictions(y_true, y_prob, threshold=th)
        records.append({
            "threshold": th,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "tp": metrics["true_positives"],
            "fp": metrics["false_positives"],
            "fn": metrics["false_negatives"],
        })

    df_thresh = pd.DataFrame(records)
    best_idx = df_thresh["f1_score"].idxmax()
    best_threshold = float(df_thresh.loc[best_idx, "threshold"])
    return best_threshold, df_thresh


def plot_threshold_tradeoff(
    df_thresh: pd.DataFrame,
    best_threshold: float,
    output_path: str = "reports/figures/16_threshold_analysis.png",
) -> None:
    """Plot precision, recall, and F1-score as a function of decision threshold."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(df_thresh["threshold"], df_thresh["precision"], label="Precision", color="#1D3557", lw=2)
    plt.plot(df_thresh["threshold"], df_thresh["recall"], label="Recall", color="#E63946", lw=2)
    plt.plot(df_thresh["threshold"], df_thresh["f1_score"], label="F1-Score", color="#2A9D8F", lw=2.5)

    plt.axvline(
        best_threshold,
        color="black",
        linestyle="--",
        label=f"Optimal F1 Threshold ({best_threshold:.2f})",
    )
    plt.axvline(
        0.50,
        color="gray",
        linestyle=":",
        label="Default Threshold (0.50)",
    )

    plt.title("Decision Threshold Optimization Curve", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Classification Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.ylim(-0.02, 1.05)
    plt.legend(frameon=True, loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EVALUATION] Threshold tradeoff plot saved to '{output_path}'.")


def plot_confusion_matrix_heatmap(
    cm: list,
    model_name: str,
    output_path: str,
    threshold: float = 0.5,
) -> None:
    """Plot confusion matrix with counts and percentages."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(cm)

    group_names = ["True Neg\n(Legit Safe)", "False Pos\n(False Alarm)", "False Neg\n(Missed Fraud)", "True Pos\n(Detected Fraud)"]
    group_counts = [f"{v:,}" for v in cm_arr.flatten()]
    group_percentages = [f"{v:.2%}" for v in cm_arr.flatten() / np.sum(cm_arr)]
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm_arr,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=1.5,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )
    plt.title(f"Confusion Matrix: {model_name} (Thresh={threshold:.2f})", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.xticks([0.5, 1.5], ["Legitimate (0)", "Fraudulent (1)"])
    plt.yticks([0.5, 1.5], ["Legitimate (0)", "Fraudulent (1)"], rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_combined_roc_and_pr_curves(
    model_probs: Dict[str, np.ndarray],
    y_test: np.ndarray,
    figures_dir: str = "reports/figures",
) -> None:
    """Plot multi-model ROC Curves and Precision-Recall Curves."""
    os.makedirs(figures_dir, exist_ok=True)
    colors = ["#1D3557", "#457B9D", "#2A9D8F", "#E63946", "#F4A261"]

    # 1. Combined ROC Curve
    plt.figure(figsize=(8, 6))
    for (name, probs), col in zip(model_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, probs)
        score = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {score:.4f})", color=col, lw=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)")
    plt.title("Comparative ROC Curves (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "17_roc_curves.png"), dpi=300)
    plt.close()

    # 2. Combined Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for (name, probs), col in zip(model_probs.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, probs)
        score = average_precision_score(y_test, probs)
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {score:.4f})", color=col, lw=2)

    no_skill = np.sum(y_test) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], "k--", label=f"Baseline Prevalence ({no_skill:.2%})")
    plt.title("Comparative Precision-Recall Curves (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "18_precision_recall_curves.png"), dpi=300)
    plt.close()
    print(f"[EVALUATION] Combined ROC and PR curves saved to '{figures_dir}'.")


def plot_model_comparison_bars(
    df_results: pd.DataFrame,
    output_path: str = "reports/figures/19_model_comparison_bar.png",
) -> None:
    """Plot multi-metric bar chart comparing all evaluated models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_to_plot = ["Precision", "Recall", "F1-Score", "PR-AUC", "ROC-AUC"]
    df_plot = df_results.set_index("Model")[metrics_to_plot]

    ax = df_plot.plot(
        kind="bar",
        figsize=(11, 5),
        colormap="viridis",
        width=0.75,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.title("Comparative Model Performance Across Key Evaluation Metrics", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Model Architecture", fontsize=11)
    plt.ylabel("Score (0.0 – 1.0)", fontsize=11)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15, ha="right")
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EVALUATION] Model comparison bar chart saved to '{output_path}'.")

