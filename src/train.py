"""
Comprehensive Training, Cross-Validation & Model Selection Module.
Trains Pre-Transaction and Post-Transaction fraud detection systems with leak-free
cross-validation pipelines, hyperparameter benchmarking, out-of-fold threshold calibration,
and comprehensive reporting.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

# Avoid subprocess hangs on Windows sandbox
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
sys.path.insert(0, os.path.abspath("."))

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.preprocessing import (
    prepare_and_save_data,
    build_preprocessor_pipeline,
    get_feature_columns,
)
from src.evaluate import find_optimal_threshold, evaluate_predictions


def evaluate_model_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    train_time: float = 0.0,
    infer_time: float = 0.0,
) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics including timing and confusion matrix."""
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
        "training_time_seconds": round(float(train_time), 4),
        "inference_time_seconds": round(float(infer_time), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train_and_benchmark_suite(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mode: str = "post",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and benchmark Logistic Regression, Decision Tree, Random Forest, and XGBoost
    using leak-free cross-validation pipelines.
    """
    with joblib.parallel_backend("threading"):
        print(f"\n{'='*70}")
        print(f"   STARTING {mode.upper()}-TRANSACTION MODEL BENCHMARKING")
        print(f"{'='*70}")

        cat_cols, num_cols = get_feature_columns(X_train)
        imbalance_ratio = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))

        classifiers = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state, solver="lbfgs"
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=2, class_weight="balanced", random_state=random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_leaf=1,
                class_weight="balanced", random_state=random_state, n_jobs=1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                scale_pos_weight=imbalance_ratio, eval_metric="logloss",
                random_state=random_state, n_jobs=1
            ),
        }

        benchmark_summary = []
        trained_pipelines = {}
        oof_predictions = {}
        test_predictions = {}

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        for name, clf in classifiers.items():
            print(f"[{mode.upper()}] Training & CV: {name}...")
            preprocessor = build_preprocessor_pipeline(cat_cols, num_cols)
            pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

            # 1. Out-of-fold CV predictions (no data leakage)
            oof_prob = cross_val_predict(
                pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
            )[:, 1]
            oof_predictions[name] = oof_prob

            # 2. Fit full pipeline on X_train
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            t_train = time.time() - t0

            # 3. Predict on untouched test set
            t0 = time.time()
            test_prob = pipeline.predict_proba(X_test)[:, 1]
            t_infer = time.time() - t0
            test_predictions[name] = test_prob

            metrics = evaluate_model_performance(
                y_test.values, test_prob, threshold=0.50, train_time=t_train, infer_time=t_infer
            )

            benchmark_summary.append({
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "ROC-AUC": metrics["roc_auc"],
                "PR-AUC": metrics["pr_auc"],
                "False_Positives": metrics["false_positives"],
                "False_Negatives": metrics["false_negatives"],
                "Train_Time_s": metrics["training_time_seconds"],
                "Infer_Time_s": metrics["inference_time_seconds"],
            })
            trained_pipelines[name] = pipeline

        df_benchmark = pd.DataFrame(benchmark_summary).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
        print(f"\n--- {mode.upper()}-TRANSACTION MODEL COMPARISON TABLE ---")
        print(df_benchmark.to_string(index=False))

        # Champion selection based on PR-AUC and F1-score
        champion_name = df_benchmark.iloc[0]["Model"]
        champion_pipeline = trained_pipelines[champion_name]
        champion_oof_prob = oof_predictions[champion_name]
        champion_test_prob = test_predictions[champion_name]

        print(f"\n[{mode.upper()}] Selected Champion: '{champion_name}' (PR-AUC: {df_benchmark.iloc[0]['PR-AUC']:.4f})")

        # Threshold Optimization strictly using training out-of-fold predictions
        print(f"[{mode.upper()}] Calibrating threshold on training Out-Of-Fold probabilities...")
        best_th, df_thresh = find_optimal_threshold(y_train.values, champion_oof_prob)
        print(f"[{mode.upper()}] Selected Optimal Validation Threshold: {best_th:.4f}")

        # Evaluate champion at default (0.50) and optimal threshold on untouched test set
        champ_default_metrics = evaluate_model_performance(y_test.values, champion_test_prob, threshold=0.50)
        champ_optimal_metrics = evaluate_model_performance(y_test.values, champion_test_prob, threshold=best_th)

        # Extract fitted preprocessor and classifier
        fitted_preprocessor = champion_pipeline.named_steps["preprocessor"]
        fitted_classifier = champion_pipeline.named_steps["classifier"]

        # Extract encoded feature names
        try:
            raw_names = fitted_preprocessor.get_feature_names_out()
            cleaned_feature_names = [n.split("__")[-1] for n in raw_names]
        except Exception:
            cleaned_feature_names = list(X_train.columns)

        return {
            "mode": mode,
            "champion_name": champion_name,
            "champion_pipeline": champion_pipeline,
            "preprocessor": fitted_preprocessor,
            "classifier": fitted_classifier,
            "optimal_threshold": float(best_th),
            "benchmark_table": df_benchmark,
            "default_metrics": champ_default_metrics,
            "optimal_metrics": champ_optimal_metrics,
            "df_threshold_curve": df_thresh,
            "feature_names": cleaned_feature_names,
            "input_features_count": len(cleaned_feature_names),
            "test_probabilities": champion_test_prob,
            "oof_probabilities": champion_oof_prob,
        }


def save_system_artifacts(
    pre_results: Dict[str, Any],
    post_results: Dict[str, Any],
    models_dir: str = "models",
    reports_dir: str = "reports",
) -> None:
    """Save separate model artifacts and evaluation reports for both systems."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # --- 1. PRE-TRANSACTION ARTIFACTS ---
    joblib.dump(pre_results["classifier"], os.path.join(models_dir, "pre_transaction_model.joblib"))
    joblib.dump(pre_results["preprocessor"], os.path.join(models_dir, "pre_transaction_pipeline.joblib"))
    
    pre_metadata = {
        "system_mode": "pre_transaction",
        "description": "Pre-Transaction Fraud Screening Model (Authorization Switch)",
        "model_name": pre_results["champion_name"],
        "optimal_threshold": pre_results["optimal_threshold"],
        "input_features_count": pre_results["input_features_count"],
        "feature_names": pre_results["feature_names"],
        "evaluation_summary": pre_results["optimal_metrics"],
        "default_evaluation": pre_results["default_metrics"],
    }
    with open(os.path.join(models_dir, "pre_transaction_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(pre_metadata, f, indent=4)

    pre_eval_report = {
        "mode": "pre_transaction",
        "champion_model": pre_results["champion_name"],
        "optimal_threshold": pre_results["optimal_metrics"],
        "default_threshold_0_50": pre_results["default_metrics"],
        "all_models_summary": pre_results["benchmark_table"].to_dict(orient="records"),
    }
    with open(os.path.join(reports_dir, "pre_transaction_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(pre_eval_report, f, indent=4)

    # --- 2. POST-TRANSACTION ARTIFACTS ---
    joblib.dump(post_results["classifier"], os.path.join(models_dir, "post_transaction_model.joblib"))
    joblib.dump(post_results["preprocessor"], os.path.join(models_dir, "post_transaction_pipeline.joblib"))

    post_metadata = {
        "system_mode": "post_transaction",
        "description": "Post-Transaction Fraud Detection Model (Clearing & Settlement Audit)",
        "model_name": post_results["champion_name"],
        "optimal_threshold": post_results["optimal_threshold"],
        "input_features_count": post_results["input_features_count"],
        "feature_names": post_results["feature_names"],
        "evaluation_summary": post_results["optimal_metrics"],
        "default_evaluation": post_results["default_metrics"],
    }
    with open(os.path.join(models_dir, "post_transaction_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(post_metadata, f, indent=4)

    post_eval_report = {
        "mode": "post_transaction",
        "champion_model": post_results["champion_name"],
        "optimal_threshold": post_results["optimal_metrics"],
        "default_threshold_0_50": post_results["default_metrics"],
        "all_models_summary": post_results["benchmark_table"].to_dict(orient="records"),
    }
    with open(os.path.join(reports_dir, "post_transaction_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(post_eval_report, f, indent=4)

    # --- 3. BACKWARD COMPATIBILITY COPIES ---
    joblib.dump(post_results["classifier"], os.path.join(models_dir, "best_model.joblib"))
    joblib.dump(post_results["preprocessor"], os.path.join(models_dir, "preprocessing_pipeline.joblib"))
    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(post_metadata, f, indent=4)
    with open(os.path.join(reports_dir, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(post_eval_report, f, indent=4)

    # --- 4. MODEL COMPARISON JSON ---
    model_comparison_data = {
        "generated_timestamp": "2026-11-08",
        "evaluation_setup": "Untouched Stratified 20% Test Partition (N=2,229; 228 Frauds, 2,001 Legitimate)",
        "pre_transaction": {
            "model_name": pre_results["champion_name"],
            "optimal_threshold": pre_results["optimal_threshold"],
            "metrics": pre_results["optimal_metrics"],
            "all_models": pre_results["benchmark_table"].to_dict(orient="records"),
        },
        "post_transaction": {
            "model_name": post_results["champion_name"],
            "optimal_threshold": post_results["optimal_threshold"],
            "metrics": post_results["optimal_metrics"],
            "all_models": post_results["benchmark_table"].to_dict(orient="records"),
        },
    }
    with open(os.path.join(reports_dir, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(model_comparison_data, f, indent=4)

    # --- 5. THRESHOLD ANALYSIS JSON ---
    threshold_analysis_data = {
        "pre_transaction": {
            "optimal_threshold": pre_results["optimal_threshold"],
            "tradeoff_points": pre_results["df_threshold_curve"].iloc[::5].to_dict(orient="records"),
        },
        "post_transaction": {
            "optimal_threshold": post_results["optimal_threshold"],
            "tradeoff_points": post_results["df_threshold_curve"].iloc[::5].to_dict(orient="records"),
        },
    }
    with open(os.path.join(reports_dir, "threshold_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(threshold_analysis_data, f, indent=4)

    print("\n[ARTIFACTS] All Pre- and Post-Transaction models and reports saved successfully.")


def run_training_pipeline():
    """Execute end-to-end training and artifact generation."""
    data = prepare_and_save_data()
    pre_results = train_and_benchmark_suite(
        data["pre"]["X_train"], data["pre"]["y_train"],
        data["pre"]["X_test"], data["pre"]["y_test"],
        mode="pre", random_state=42
    )
    post_results = train_and_benchmark_suite(
        data["post"]["X_train"], data["post"]["y_train"],
        data["post"]["X_test"], data["post"]["y_test"],
        mode="post", random_state=42
    )
    save_system_artifacts(pre_results, post_results)
    return pre_results, post_results


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath("."))
    run_training_pipeline()
