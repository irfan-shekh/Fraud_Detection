"""
Temporal Feature Ablation Experiment for FraudGuard AI.
Evaluates the impact of removing temporal features (step, hour, day)
and balance-ratio signals on Out-Of-Time (OOT) generalization.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_raw_data, split_features_target
from src.feature_engineering import engineer_pre_transaction_features
from src.preprocessing import build_preprocessor_pipeline, get_feature_columns
from src.evaluate import find_optimal_threshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def run_ablation_experiments() -> Dict[str, Any]:
    print("=" * 80)
    print("   FRAUDGUARD AI: TEMPORAL FEATURE ABLATION EXPERIMENT")
    print("=" * 80)

    raw_df = load_raw_data()
    total_test_fraud_value = raw_df[raw_df["step"] > 6][raw_df[raw_df["step"] > 6]["isFraud"] == 1]["amount"].sum()
    total_test_frauds = int((raw_df[raw_df["step"] > 6]["isFraud"] == 1).sum())

    print(f"\n[DATA] OOT Horizon: Steps 1-6 Train ({len(raw_df[raw_df['step'] <= 6])} rows, {raw_df[raw_df['step'] <= 6]['isFraud'].sum()} frauds)")
    print(f"[DATA] OOT Horizon: Steps 7-95 Test  ({len(raw_df[raw_df['step'] > 6])} rows, {total_test_frauds} frauds, ${total_test_fraud_value:,.2f} exposure)")

    # Define the 3 experimental configurations
    configurations = {
        "Exp_A_Full_Pre_Transaction": {
            "name": "Experiment A: Full Pre-Transaction Features",
            "description": "Baseline pre-transaction feature set (including step, hour, day, and balance ratio).",
            "drop_cols": [],
        },
        "Exp_B_Remove_Temporal": {
            "name": "Experiment B: Remove Temporal Features",
            "description": "Excludes temporal attributes: step, transaction_hour, transaction_day.",
            "drop_cols": ["step", "transaction_hour", "transaction_day"],
        },
        "Exp_C_Remove_Temporal_and_Ratio": {
            "name": "Experiment C: Remove Temporal & Balance Ratio",
            "description": "Excludes temporal attributes (step, hour, day) AND amount_to_oldbalance_orig_ratio.",
            "drop_cols": ["step", "transaction_hour", "transaction_day", "amount_to_oldbalance_orig_ratio"],
        },
    }

    results = {}
    summary_table = []

    # Chronological Split: Train Steps 1-6, Test Steps 7-95
    train_raw = raw_df[raw_df["step"] <= 6].reset_index(drop=True)
    test_raw = raw_df[raw_df["step"] > 6].reset_index(drop=True)

    test_amounts = test_raw["amount"].values

    with joblib.parallel_backend("threading"):
        for exp_key, config in configurations.items():
            print(f"\n>>> EXECUTING: {config['name']}...")

            # 1. Engineer pre-transaction features
            train_fe = engineer_pre_transaction_features(train_raw, drop_identifiers=True)
            test_fe = engineer_pre_transaction_features(test_raw, drop_identifiers=True)

            # Drop specified ablation columns
            if config["drop_cols"]:
                train_fe = train_fe.drop(columns=[c for c in config["drop_cols"] if c in train_fe.columns])
                test_fe = test_fe.drop(columns=[c for c in config["drop_cols"] if c in test_fe.columns])

            X_train, y_train = split_features_target(train_fe, "isFraud")
            X_test, y_test = split_features_target(test_fe, "isFraud")

            cat_cols, num_cols = get_feature_columns(X_train)
            preprocessor = build_preprocessor_pipeline(cat_cols, num_cols)

            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            )

            pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

            # 2. Threshold Calibration strictly on Training Out-of-Fold Predictions
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            print("    Calculating training out-of-fold probabilities via 5-fold CV...")
            oof_probs = cross_val_predict(
                pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
            )[:, 1]

            opt_th, df_thresh = find_optimal_threshold(y_train.values, oof_probs)
            print(f"    Selected optimal validation threshold: {opt_th:.4f}")

            # 3. Fit pipeline on full training set
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - t0

            # 4. Predict on untouched OOT test partition
            t0 = time.time()
            test_probs = pipeline.predict_proba(X_test)[:, 1]
            infer_time = time.time() - t0

            test_preds = (test_probs >= opt_th).astype(int)

            # 5. Calculate evaluation metrics
            cm = confusion_matrix(y_test, test_preds)
            tn, fp, fn, tp = cm.ravel()

            acc = accuracy_score(y_test, test_preds)
            prec = precision_score(y_test, test_preds, zero_division=0)
            rec = recall_score(y_test, test_preds)
            f1 = f1_score(y_test, test_preds)
            roc_auc = roc_auc_score(y_test, test_probs)
            pr_auc = average_precision_score(y_test, test_probs)

            # Monetary metrics
            tp_mask = (y_test.values == 1) & (test_preds == 1)
            flagged_value = float(test_amounts[tp_mask].sum())
            capture_rate = (flagged_value / total_test_fraud_value) if total_test_fraud_value > 0 else 1.0

            res_obj = {
                "experiment_id": exp_key,
                "name": config["name"],
                "description": config["description"],
                "features_excluded": config["drop_cols"],
                "features_used_count": X_train.shape[1],
                "feature_names": list(X_train.columns),
                "optimal_threshold": float(opt_th),
                "metrics": {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1_score": float(f1),
                    "roc_auc": float(roc_auc),
                    "pr_auc": float(pr_auc),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                    "fraud_cases_detected": int(tp),
                    "total_fraud_cases": total_test_frauds,
                    "fraud_value_flagged_usd": flagged_value,
                    "total_fraud_exposure_usd": float(total_test_fraud_value),
                    "capital_capture_rate": float(capture_rate),
                    "training_time_seconds": round(train_time, 4),
                    "inference_time_seconds": round(infer_time, 4),
                },
                "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            }

            results[exp_key] = res_obj

            summary_table.append({
                "Experiment": config["name"],
                "Features": X_train.shape[1],
                "Threshold": round(opt_th, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "PR_AUC": round(pr_auc, 4),
                "FP": int(fp),
                "FN": int(fn),
                "Frauds_Caught": f"{tp}/{total_test_frauds}",
                "Capture_Rate": f"{capture_rate * 100:.2f}%",
            })

    # Output comparison table
    df_summary = pd.DataFrame(summary_table)
    print("\n" + "=" * 80)
    print("   TEMPORAL FEATURE ABLATION EXPERIMENT SUMMARY TABLE")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    # Persist JSON report
    with open("reports/temporal_feature_ablation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("\n[ABLATION] JSON report saved to 'reports/temporal_feature_ablation.json'.")

    # Persist CSV report
    df_summary.to_csv("reports/temporal_feature_ablation.csv", index=False)
    print("[ABLATION] CSV report saved to 'reports/temporal_feature_ablation.csv'.")

    return results


if __name__ == "__main__":
    run_ablation_experiments()

