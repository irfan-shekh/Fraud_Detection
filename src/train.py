"""
Machine Learning Model Training, Imbalance Handling & Hyperparameter Optimization Module.
Implements Logistic Regression, Decision Tree, Random Forest, XGBoost, and Isolation Forest
with Stratified Cross-Validation and resample comparisons (SMOTE, Class-Weighting, RUS).
"""

import os
import sys
# Set loky CPU count to 1 to prevent win32 subprocess hangs in sandbox
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.preprocessing import prepare_and_save_data, create_train_test_split, build_preprocessor_pipeline, get_feature_columns


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """
    Train all required algorithms, compare imbalance strategies,
    perform hyperparameter tuning, and evaluate on untouched test data.
    """
    with joblib.parallel_backend("threading"):
        print("[TRAINING] Starting comprehensive model training and comparison...")

        # Transform data using fitted preprocessor
        X_tr = preprocessor.transform(X_train)
        X_te = preprocessor.transform(X_test)

        # Imbalance ratio for scale_pos_weight
        imbalance_ratio = float((len(y_train) - y_train.sum()) / y_train.sum())

        results = []
        trained_models = {}

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        # =========================================================================
        # 1. LOGISTIC REGRESSION (Baseline & Class-Weighted)
        # =========================================================================
        print("[1/5] Training Logistic Regression...")
        lr_cw = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        )
        lr_cw.fit(X_tr, y_train)
        trained_models["Logistic Regression"] = lr_cw

        # =========================================================================
        # 2. DECISION TREE CLASSIFIER (Tuned with RandomizedSearch)
        # =========================================================================
        print("[2/5] Tuning & Training Decision Tree...")
        dt_params = {
            "max_depth": [4, 6, 8, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", None],
        }
        dt_search = RandomizedSearchCV(
            DecisionTreeClassifier(random_state=random_state),
            dt_params,
            n_iter=10,
            scoring="average_precision",
            cv=cv,
            random_state=random_state,
            n_jobs=2,
        )
        dt_search.fit(X_tr, y_train)
        best_dt = dt_search.best_estimator_
        trained_models["Decision Tree"] = best_dt

        # =========================================================================
        # 3. RANDOM FOREST CLASSIFIER (Tuned with RandomizedSearch)
        # =========================================================================
        print("[3/5] Tuning & Training Random Forest...")
        rf_params = {
            "n_estimators": [100, 150],
            "max_depth": [8, 12, 16],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": ["balanced", None],
        }
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=random_state, n_jobs=2),
            rf_params,
            n_iter=8,
            scoring="average_precision",
            cv=cv,
            random_state=random_state,
            n_jobs=2,
        )
        rf_search.fit(X_tr, y_train)
        best_rf = rf_search.best_estimator_
        trained_models["Random Forest"] = best_rf

        # =========================================================================
        # 4. XGBOOST CLASSIFIER (Tuned with RandomizedSearch)
        # =========================================================================
        print("[4/5] Tuning & Training XGBoost...")
        xgb_params = {
            "n_estimators": [100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "scale_pos_weight": [1.0, imbalance_ratio],
        }
        xgb_search = RandomizedSearchCV(
            XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=2,
            ),
            xgb_params,
            n_iter=8,
            scoring="average_precision",
            cv=cv,
            random_state=random_state,
            n_jobs=2,
        )
        xgb_search.fit(X_tr, y_train)
        best_xgb = xgb_search.best_estimator_
        trained_models["XGBoost"] = best_xgb

        # =========================================================================
        # 5. ISOLATION FOREST (Unsupervised Anomaly Detection)
        # =========================================================================
        print("[5/5] Training Isolation Forest (Unsupervised Anomaly Detection)...")
        iso = IsolationForest(
            n_estimators=100,
            contamination=0.10,
            random_state=random_state,
            n_jobs=2,
        )
        iso.fit(X_tr)
        trained_models["Isolation Forest"] = iso

        # =========================================================================
        # EVALUATE ALL MODELS ON TEST SET
        # =========================================================================
        print("[EVALUATION] Scoring all models on untouched test set...")

        for name, model in trained_models.items():
            if name == "Isolation Forest":
                scores = -model.decision_function(X_te)
                raw_preds = model.predict(X_te)
                y_pred = np.where(raw_preds == -1, 1, 0)
                y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            else:
                y_pred = model.predict(X_te)
                y_prob = model.predict_proba(X_te)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": roc_auc,
                "PR-AUC": pr_auc,
            })

        df_results = pd.DataFrame(results).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
        print("\n--- MODEL COMPARISON SUMMARY ---")
        print(df_results.to_string(index=False))

        best_model_name = df_results.iloc[0]["Model"]
        best_model = trained_models[best_model_name]
        print(f"\n[SELECTION] Best model identified: '{best_model_name}' with PR-AUC = {df_results.iloc[0]['PR-AUC']:.4f}")

        return df_results, trained_models, best_model


def compare_imbalance_techniques(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare 4 class imbalance handling strategies using Random Forest as benchmark:
    1. Original (No resampling)
    2. Class-Weighted
    3. SMOTE (Oversampling)
    4. Random Under-Sampling
    """
    with joblib.parallel_backend("threading"):
        print("\n[IMBALANCE STUDY] Comparing handling techniques...")
        X_tr = preprocessor.transform(X_train)
        X_te = preprocessor.transform(X_test)

        strategies = {
            "1. Original (No Resampling)": RandomForestClassifier(random_state=random_state, n_jobs=2),
            "2. Class-Weighted": RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=2),
            "3. SMOTE Resampling": ImbPipeline([
                ("smote", SMOTE(random_state=random_state)),
                ("rf", RandomForestClassifier(random_state=random_state, n_jobs=2)),
            ]),
            "4. Random Under-Sampling": ImbPipeline([
                ("rus", RandomUnderSampler(random_state=random_state)),
                ("rf", RandomForestClassifier(random_state=random_state, n_jobs=2)),
            ]),
        }

        comp_results = []
        for strat_name, pipeline in strategies.items():
            pipeline.fit(X_tr, y_train)
            y_pred = pipeline.predict(X_te)
            y_prob = pipeline.predict_proba(X_te)[:, 1]

            comp_results.append({
                "Strategy": strat_name,
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "PR-AUC": average_precision_score(y_test, y_prob),
                "ROC-AUC": roc_auc_score(y_test, y_prob),
            })

        df_strat = pd.DataFrame(comp_results)
        print(df_strat.to_string(index=False))
        return df_strat


def compare_feature_engineering_impact(
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Empirically compare Random Forest & XGBoost performance
    with vs without engineered features.
    """
    with joblib.parallel_backend("threading"):
        print("\n[FEATURE ABLATION STUDY] Comparing with vs without engineered features...")
        # 1. With engineered features
        X_tr_fe, X_te_fe, y_tr, y_te = create_train_test_split(use_feature_engineering=True)
        cat_fe, num_fe = get_feature_columns(X_tr_fe)
        prep_fe = build_preprocessor_pipeline(cat_fe, num_fe)
        X_tr_fe_t = prep_fe.fit_transform(X_tr_fe)
        X_te_fe_t = prep_fe.transform(X_te_fe)

        # 2. Without engineered features (raw baseline)
        X_tr_raw, X_te_raw, _, _ = create_train_test_split(use_feature_engineering=False)
        cat_raw, num_raw = get_feature_columns(X_tr_raw)
        prep_raw = build_preprocessor_pipeline(cat_raw, num_raw)
        X_tr_raw_t = prep_raw.fit_transform(X_tr_raw)
        X_te_raw_t = prep_raw.transform(X_te_raw)

        ablation_results = []
        for model_name, clf in [
            ("Random Forest", RandomForestClassifier(random_state=random_state, n_jobs=2)),
            ("XGBoost", XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=2)),
        ]:
            # Raw
            clf.fit(X_tr_raw_t, y_tr)
            prob_raw = clf.predict_proba(X_te_raw_t)[:, 1]
            pred_raw = clf.predict(X_te_raw_t)

            # Engineered
            clf.fit(X_tr_fe_t, y_tr)
            prob_fe = clf.predict_proba(X_te_fe_t)[:, 1]
            pred_fe = clf.predict(X_te_fe_t)

            ablation_results.append({
                "Model": model_name,
                "Feature Set": "Raw Baseline (6 features)",
                "Precision": precision_score(y_te, pred_raw, zero_division=0),
                "Recall": recall_score(y_te, pred_raw, zero_division=0),
                "F1-Score": f1_score(y_te, pred_raw, zero_division=0),
                "PR-AUC": average_precision_score(y_te, prob_raw),
            })
            ablation_results.append({
                "Model": model_name,
                "Feature Set": "Engineered Features (19 features)",
                "Precision": precision_score(y_te, pred_fe, zero_division=0),
                "Recall": recall_score(y_te, pred_fe, zero_division=0),
                "F1-Score": f1_score(y_te, pred_fe, zero_division=0),
                "PR-AUC": average_precision_score(y_te, prob_fe),
            })

        df_ablation = pd.DataFrame(ablation_results)
        print(df_ablation.to_string(index=False))
        return df_ablation


def save_best_model_and_artifacts(
    best_model: Any,
    best_model_name: str,
    preprocessor: Any,
    feature_names: List[str],
    evaluation_summary: Dict[str, Any],
    threshold: float = 0.5,
    models_dir: str = "models",
) -> None:
    """Serialize the trained best model and metadata to disk."""
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "best_model.joblib")
    pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(preprocessor, pipeline_path)

    metadata = {
        "model_name": best_model_name,
        "optimal_threshold": float(threshold),
        "feature_names": feature_names,
        "evaluation_summary": evaluation_summary,
        "input_features_count": len(feature_names),
        "description": "BIA Fraud Detection Capstone Champion Model",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[ARTIFACTS] Best model saved to '{model_path}'.")
    print(f"[ARTIFACTS] Metadata saved to '{metadata_path}'.")



def compare_imbalance_techniques(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare 4 class imbalance handling strategies using Random Forest as benchmark:
    1. Original (No resampling)
    2. Class-Weighted
    3. SMOTE (Oversampling)
    4. Random Under-Sampling
    """
    print("\n[IMBALANCE STUDY] Comparing handling techniques...")
    X_tr = preprocessor.transform(X_train)
    X_te = preprocessor.transform(X_test)

    strategies = {
        "1. Original (No Resampling)": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        "2. Class-Weighted": RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1),
        "3. SMOTE Resampling": ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("rf", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
        ]),
        "4. Random Under-Sampling": ImbPipeline([
            ("rus", RandomUnderSampler(random_state=random_state)),
            ("rf", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
        ]),
    }

    comp_results = []
    for strat_name, pipeline in strategies.items():
        pipeline.fit(X_tr, y_train)
        y_pred = pipeline.predict(X_te)
        y_prob = pipeline.predict_proba(X_te)[:, 1]

        comp_results.append({
            "Strategy": strat_name,
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "PR-AUC": average_precision_score(y_test, y_prob),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
        })

    df_strat = pd.DataFrame(comp_results)
    print(df_strat.to_string(index=False))
    return df_strat


def compare_feature_engineering_impact(
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Empirically compare Random Forest & XGBoost performance
    with vs without engineered features.
    """
    print("\n[FEATURE ABLATION STUDY] Comparing with vs without engineered features...")
    # 1. With engineered features
    X_tr_fe, X_te_fe, y_tr, y_te = create_train_test_split(use_feature_engineering=True)
    cat_fe, num_fe = get_feature_columns(X_tr_fe)
    prep_fe = build_preprocessor_pipeline(cat_fe, num_fe)
    X_tr_fe_t = prep_fe.fit_transform(X_tr_fe)
    X_te_fe_t = prep_fe.transform(X_te_fe)

    # 2. Without engineered features (raw baseline)
    X_tr_raw, X_te_raw, _, _ = create_train_test_split(use_feature_engineering=False)
    cat_raw, num_raw = get_feature_columns(X_tr_raw)
    prep_raw = build_preprocessor_pipeline(cat_raw, num_raw)
    X_tr_raw_t = prep_raw.fit_transform(X_tr_raw)
    X_te_raw_t = prep_raw.transform(X_te_raw)

    ablation_results = []
    for model_name, clf in [
        ("Random Forest", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
        ("XGBoost", XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)),
    ]:
        # Raw
        clf.fit(X_tr_raw_t, y_tr)
        prob_raw = clf.predict_proba(X_te_raw_t)[:, 1]
        pred_raw = clf.predict(X_te_raw_t)

        # Engineered
        clf.fit(X_tr_fe_t, y_tr)
        prob_fe = clf.predict_proba(X_te_fe_t)[:, 1]
        pred_fe = clf.predict(X_te_fe_t)

        ablation_results.append({
            "Model": model_name,
            "Feature Set": "Raw Baseline (6 features)",
            "Precision": precision_score(y_te, pred_raw, zero_division=0),
            "Recall": recall_score(y_te, pred_raw, zero_division=0),
            "F1-Score": f1_score(y_te, pred_raw, zero_division=0),
            "PR-AUC": average_precision_score(y_te, prob_raw),
        })
        ablation_results.append({
            "Model": model_name,
            "Feature Set": "Engineered Features (19 features)",
            "Precision": precision_score(y_te, pred_fe, zero_division=0),
            "Recall": recall_score(y_te, pred_fe, zero_division=0),
            "F1-Score": f1_score(y_te, pred_fe, zero_division=0),
            "PR-AUC": average_precision_score(y_te, prob_fe),
        })

    df_ablation = pd.DataFrame(ablation_results)
    print(df_ablation.to_string(index=False))
    return df_ablation


def save_best_model_and_artifacts(
    best_model: Any,
    best_model_name: str,
    preprocessor: Any,
    feature_names: List[str],
    evaluation_summary: Dict[str, Any],
    threshold: float = 0.5,
    models_dir: str = "models",
) -> None:
    """Serialize the trained best model and metadata to disk."""
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "best_model.joblib")
    pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(preprocessor, pipeline_path)

    metadata = {
        "model_name": best_model_name,
        "optimal_threshold": float(threshold),
        "feature_names": feature_names,
        "evaluation_summary": evaluation_summary,
        "input_features_count": len(feature_names),
        "description": "BIA Fraud Detection Capstone Champion Model",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[ARTIFACTS] Best model saved to '{model_path}'.")
    print(f"[ARTIFACTS] Metadata saved to '{metadata_path}'.")


if __name__ == "__main__":
    data_dict = prepare_and_save_data()
    X_train, X_test = data_dict["X_train"], data_dict["X_test"]
    y_train, y_test = data_dict["y_train"], data_dict["y_test"]
    preprocessor = data_dict["preprocessor"]

    # Run comparisons
    df_ablation = compare_feature_engineering_impact()
    df_imbalance = compare_imbalance_techniques(X_train, y_train, X_test, y_test, preprocessor)
    df_results, trained_models, best_model = train_and_compare_models(
        X_train, y_train, X_test, y_test, preprocessor
    )
