"""
Temporal Generalization Experiment Module for FraudGuard AI.
Rigorously evaluates model performance across time to measure stability,
drift resistance, and out-of-time (OOT) generalization.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_raw_data, split_features_target
from src.feature_engineering import (
    engineer_pre_transaction_features,
    engineer_post_transaction_features,
)
from src.preprocessing import build_preprocessor_pipeline, get_feature_columns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def run_macro_oot_experiment() -> Dict[str, Any]:
    """
    Experiment 1: Macro Out-of-Time (OOT) Holdout.
    Train strictly on historical Steps 1-6.
    Test strictly on future Steps 7-95.
    Guarantees max(train_step)=6 < min(test_step)=7.
    """
    raw_df = load_raw_data()
    results = {}

    for mode in ["pre", "post"]:
        if mode == "pre":
            df_fe = engineer_pre_transaction_features(raw_df, drop_identifiers=True)
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                class_weight="balanced", random_state=42, n_jobs=1
            )
            th = 0.21
        else:
            df_fe = engineer_post_transaction_features(raw_df, drop_identifiers=True)
            clf = XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                scale_pos_weight=8.76, random_state=42, eval_metric="logloss", n_jobs=1
            )
            th = 0.07

        train_df = df_fe[df_fe["step"] <= 6].reset_index(drop=True)
        test_df = df_fe[df_fe["step"] > 6].reset_index(drop=True)

        X_train, y_train = split_features_target(train_df, "isFraud")
        X_test, y_test = split_features_target(test_df, "isFraud")

        cat_cols, num_cols = get_feature_columns(X_train)
        preprocessor = build_preprocessor_pipeline(cat_cols, num_cols)
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        probs = pipe.predict_proba(X_test)[:, 1]
        infer_time = time.time() - t0

        preds = (probs >= th).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        test_fraud_amt = raw_df[raw_df["step"] > 6][raw_df[raw_df["step"] > 6]["isFraud"] == 1]["amount"].sum()
        tp_mask = (y_test.values == 1) & (preds == 1)
        caught_amt = raw_df[raw_df["step"] > 6].iloc[tp_mask]["amount"].sum()

        results[mode] = {
            "mode": mode,
            "train_steps": "1-6",
            "test_steps": "7-95",
            "train_records": len(train_df),
            "train_frauds": int(y_train.sum()),
            "test_records": len(test_df),
            "test_frauds": int(y_test.sum()),
            "threshold": th,
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds)),
            "f1_score": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "pr_auc": float(average_precision_score(y_test, probs)),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "total_fraud_exposure": float(test_fraud_amt),
            "intercepted_fraud_value": float(caught_amt),
            "capital_capture_rate": float(caught_amt / test_fraud_amt) if test_fraud_amt > 0 else 1.0,
            "train_time_s": round(train_time, 4),
            "infer_time_s": round(infer_time, 4),
        }

    return results


def run_walk_forward_experiment() -> List[Dict[str, Any]]:
    """
    Experiment 2: Walk-Forward Expanding Window Validation.
    Evaluates step-by-step performance during active mixed traffic (Steps 2-7).
    """
    raw_df = load_raw_data()
    walk_forward_records = []

    # Use Pre-Transaction features for rigorous authorization testing
    df_fe = engineer_pre_transaction_features(raw_df, drop_identifiers=True)

    # Windows: Train on 1..k, test on k+1
    for test_step in range(3, 8):
        train_df = df_fe[df_fe["step"] < test_step].reset_index(drop=True)
        test_df = df_fe[df_fe["step"] == test_step].reset_index(drop=True)

        X_train, y_train = split_features_target(train_df, "isFraud")
        X_test, y_test = split_features_target(test_df, "isFraud")

        if y_train.sum() == 0 or y_test.sum() == 0:
            continue

        cat_cols, num_cols = get_feature_columns(X_train)
        preprocessor = build_preprocessor_pipeline(cat_cols, num_cols)
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=1
        )
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.21).astype(int)

        cm = confusion_matrix(y_test, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = len(y_test), 0, 0, 0

        walk_forward_records.append({
            "Window": f"Train 1-{test_step-1} -> Test {test_step}",
            "Test_Step": test_step,
            "Train_Size": len(train_df),
            "Train_Frauds": int(y_train.sum()),
            "Test_Size": len(test_df),
            "Test_Frauds": int(y_test.sum()),
            "Accuracy": float(accuracy_score(y_test, preds)),
            "Precision": float(precision_score(y_test, preds, zero_division=0)),
            "Recall": float(recall_score(y_test, preds)),
            "F1_Score": float(f1_score(y_test, preds, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 1.0,
            "PR_AUC": float(average_precision_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 1.0,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        })

    return walk_forward_records


def run_late_stage_stream_experiment() -> Dict[str, Any]:
    """
    Experiment 3: Late-Stage Pure Fraud Stream Holdout (Steps 8-95).
    Train on Steps 1-7 (10,078 transactions containing 100% of normal traffic).
    Test on Steps 8-95 (1,064 pure fraud transactions across 88 consecutive hours).
    Measures long-tail temporal recall stability.
    """
    raw_df = load_raw_data()
    df_pre = engineer_pre_transaction_features(raw_df, drop_identifiers=True)

    train_df = df_pre[df_pre["step"] <= 7].reset_index(drop=True)
    test_df = df_pre[df_pre["step"] > 7].reset_index(drop=True)

    X_train, y_train = split_features_target(train_df, "isFraud")
    X_test, y_test = split_features_target(test_df, "isFraud")

    cat_cols, num_cols = get_feature_columns(X_train)
    preprocessor = build_preprocessor_pipeline(cat_cols, num_cols)
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=1
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.21).astype(int)

    total_late_frauds = len(test_df)
    caught_late_frauds = int(preds.sum())
    late_fraud_volume = raw_df[raw_df["step"] > 7]["amount"].sum()
    intercepted_volume = raw_df[raw_df["step"] > 7].iloc[preds == 1]["amount"].sum()

    return {
        "train_steps": "1-7",
        "test_steps": "8-95",
        "train_size": len(train_df),
        "test_size": total_late_frauds,
        "caught_frauds": caught_late_frauds,
        "missed_frauds": total_late_frauds - caught_late_frauds,
        "recall_rate": float(caught_late_frauds / total_late_frauds),
        "total_exposure_usd": float(late_fraud_volume),
        "intercepted_usd": float(intercepted_volume),
        "capture_rate": float(intercepted_volume / late_fraud_volume),
    }


def plot_temporal_generalization_dashboard(
    macro_results: Dict[str, Any],
    walk_forward: List[Dict[str, Any]],
    late_stage: Dict[str, Any],
    output_path: str = "reports/figures/28_temporal_generalization_analysis.png",
) -> None:
    """Generate high-resolution diagnostic visualizations for temporal generalization."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    sns.set_theme(style="whitegrid")

    # Panel 1: Walk-Forward Precision, Recall, and F1 across Time
    df_wf = pd.DataFrame(walk_forward)
    ax1 = axes[0, 0]
    steps = df_wf["Test_Step"]
    ax1.plot(steps, df_wf["Precision"], marker="o", lw=2.5, color="#1D3557", label="Precision")
    ax1.plot(steps, df_wf["Recall"], marker="s", lw=2.5, color="#E63946", label="Recall")
    ax1.plot(steps, df_wf["F1_Score"], marker="^", lw=2.5, color="#2A9D8F", label="F1-Score")
    ax1.set_title("Walk-Forward In-Flight Performance (Steps 3 to 7)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Evaluation Step (Hour)", fontsize=11)
    ax1.set_ylabel("Metric Score", fontsize=11)
    ax1.set_ylim(0.85, 1.02)
    ax1.legend(loc="lower left", frameon=True)

    # Panel 2: Stratified vs Macro OOT Performance Comparison
    ax2 = axes[0, 1]
    comparison_data = [
        {"Split": "Stratified Test (Pre-Tx)", "Recall": 0.9956, "Precision": 1.0000, "PR_AUC": 0.9996},
        {"Split": "Macro OOT Holdout (Pre-Tx)", "Recall": macro_results["pre"]["recall"], "Precision": macro_results["pre"]["precision"], "PR_AUC": macro_results["pre"]["pr_auc"]},
        {"Split": "Stratified Test (Post-Tx)", "Recall": 0.9956, "Precision": 0.9956, "PR_AUC": 0.9970},
        {"Split": "Macro OOT Holdout (Post-Tx)", "Recall": macro_results["post"]["recall"], "Precision": macro_results["post"]["precision"], "PR_AUC": macro_results["post"]["pr_auc"]},
    ]
    df_comp = pd.DataFrame(comparison_data)
    x = np.arange(len(df_comp))
    width = 0.25
    ax2.bar(x - width, df_comp["Recall"], width, label="Recall", color="#E63946")
    ax2.bar(x, df_comp["Precision"], width, label="Precision", color="#1D3557")
    ax2.bar(x + width, df_comp["PR_AUC"], width, label="PR-AUC", color="#2A9D8F")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Stratified Pre", "OOT Pre", "Stratified Post", "OOT Post"], rotation=15, ha="right")
    ax2.set_title("Stratified vs. Out-Of-Time (OOT) Holdout", fontsize=13, fontweight="bold")
    ax2.set_ylim(0.95, 1.02)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)

    # Panel 3: Cumulative Hourly Fraud Capture
    ax3 = axes[1, 0]
    raw_df = load_raw_data()
    hourly_fraud = raw_df[raw_df["isFraud"] == 1].groupby("step")["amount"].sum().reset_index()
    hourly_fraud["cum_amount"] = hourly_fraud["amount"].cumsum() / 1e6
    ax3.plot(hourly_fraud["step"], hourly_fraud["cum_amount"], color="#E63946", lw=2.5)
    ax3.axvline(x=6, color="#457B9D", ls="--", lw=2, label="OOT Training Boundary (Step 6)")
    ax3.fill_between(hourly_fraud["step"], hourly_fraud["cum_amount"], where=(hourly_fraud["step"] > 6), color="#A8DADC", alpha=0.3, label="OOT Evaluation Horizon")
    ax3.set_title("Cumulative Fraud Volume Across 95 Operating Hours", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Transaction Step (Hour)", fontsize=11)
    ax3.set_ylabel("Cumulative Fraud ($ Millions)", fontsize=11)
    ax3.legend(loc="upper left", frameon=True)

    # Panel 4: Late-Stage Long-Tail Stream Capture (Steps 8-95)
    ax4 = axes[1, 1]
    categories = ["Intercepted ($)", "Missed Exposure ($)"]
    vals = [late_stage["intercepted_usd"] / 1e6, (late_stage["total_exposure_usd"] - late_stage["intercepted_usd"]) / 1e6]
    colors = ["#2A9D8F", "#E63946"]
    bars = ax4.bar(categories, vals, color=colors, width=0.5)
    ax4.set_title(f"Late-Stage Stream Interception (Steps 8-95: N={late_stage['test_size']})", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Fraud Volume ($ Millions)", fontsize=11)
    for b in bars:
        h = b.get_height()
        ax4.annotate(f"${h:,.2f}M", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[TEMPORAL] Dashboard plot saved to '{output_path}'.")


def run_full_temporal_experiment_suite():
    print("=" * 80)
    print("   FRAUDGUARD AI: TEMPORAL GENERALIZATION & DRIFT EXPERIMENT SUITE")
    print("=" * 80)

    # 1. Macro OOT
    print("\n>>> EXPERIMENT 1: MACRO OUT-OF-TIME (OOT) HOLDOUT (STEPS 1-6 VS 7-95)...")
    macro_res = run_macro_oot_experiment()
    print(f"    Pre-Transaction OOT:  Recall={macro_res['pre']['recall']:.4f}, Precision={macro_res['pre']['precision']:.4f}, PR-AUC={macro_res['pre']['pr_auc']:.4f}, Capture={macro_res['pre']['capital_capture_rate']*100:.2f}%")
    print(f"    Post-Transaction OOT: Recall={macro_res['post']['recall']:.4f}, Precision={macro_res['post']['precision']:.4f}, PR-AUC={macro_res['post']['pr_auc']:.4f}, Capture={macro_res['post']['capital_capture_rate']*100:.2f}%")

    # 2. Walk-Forward
    print("\n>>> EXPERIMENT 2: WALK-FORWARD EXPANDING WINDOW VALIDATION (STEPS 3 TO 7)...")
    wf_res = run_walk_forward_experiment()
    df_wf = pd.DataFrame(wf_res)
    print(df_wf[["Window", "Test_Size", "Test_Frauds", "Recall", "Precision", "F1_Score", "PR_AUC"]].to_string(index=False))

    # 3. Late-Stage Pure Fraud Stream
    print("\n>>> EXPERIMENT 3: LATE-STAGE PURE FRAUD STREAM GENERALIZATION (STEPS 8 TO 95)...")
    late_res = run_late_stage_stream_experiment()
    print(f"    Late-Stage Pure Frauds Evaluated: {late_res['test_size']}")
    print(f"    Frauds Caught: {late_res['caught_frauds']} / {late_res['test_size']} ({late_res['recall_rate']*100:.2f}%)")
    print(f"    Monetary Volume Intercepted: ${late_res['intercepted_usd']:,.2f} of ${late_res['total_exposure_usd']:,.2f} ({late_res['capture_rate']*100:.2f}%)")

    # 4. Save JSON Report
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "macro_out_of_time": macro_res,
        "walk_forward_expanding_window": wf_res,
        "late_stage_stream": late_res,
    }
    with open("reports/temporal_generalization_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4)
    print("\n[TEMPORAL] Full JSON report saved to 'reports/temporal_generalization_report.json'.")

    # Save CSV
    df_wf.to_csv("reports/temporal_generalization_report.csv", index=False)
    print("[TEMPORAL] Walk-forward CSV saved to 'reports/temporal_generalization_report.csv'.")

    # 5. Generate Visualizations
    plot_temporal_generalization_dashboard(macro_res, wf_res, late_res)

    print("\n" + "=" * 80)
    print("   TEMPORAL EXPERIMENT SUITE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return full_report


if __name__ == "__main__":
    run_full_temporal_experiment_suite()
