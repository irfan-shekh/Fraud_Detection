"""
Model Explainability Module using SHAP (SHapley Additive exPlanations).
Generates global summary plots, feature importance bar charts, and individual transaction
explanations separately for Pre-Transaction and Post-Transaction models.
"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from typing import List, Any, Optional


def get_preprocessed_feature_names(preprocessor, X_train: pd.DataFrame) -> List[str]:
    """Retrieve output feature names from the fitted preprocessor ColumnTransformer."""
    try:
        names = preprocessor.get_feature_names_out()
        cleaned_names = [n.split("__")[-1] for n in names]
        return cleaned_names
    except Exception:
        return [f"feat_{i}" for i in range(X_train.shape[1])]


def run_shap_analysis(
    model: Any,
    preprocessor: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "reports/figures",
    mode: str = "post",
    sample_size: int = 400,
) -> Any:
    """
    Execute TreeExplainer on test samples, produce global and local SHAP charts for the given model.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[SHAP - {mode.upper()}] Computing SHAP explainability values (sample size = {sample_size})...")

    feature_names = get_preprocessed_feature_names(preprocessor, X_train)

    # Transform test set
    X_test_trans = preprocessor.transform(X_test)
    X_test_sample = X_test_trans[:sample_size]
    y_test_sample = y_test.iloc[:sample_size]

    # Initialize Explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_sample)
    except Exception as e:
        print(f"[SHAP] TreeExplainer fallback to generic Explainer: {e}")
        explainer = shap.Explainer(model.predict, X_test_sample)
        shap_values = explainer(X_test_sample)

    # Extract values for positive class (fraud) if multi-dimensional
    if len(shap_values.shape) == 3:
        shap_values_fraud = shap_values[:, :, 1]
    else:
        shap_values_fraud = shap_values

    shap_values_fraud.feature_names = feature_names

    suffix = f"_{mode}"

    # 1. SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values_fraud.values,
        X_test_sample,
        feature_names=feature_names,
        show=False,
        max_display=12,
    )
    plt.title(f"SHAP Global Feature Impact ({mode.capitalize()}-Transaction)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, f"20_shap_summary{suffix}.png")
    plt.savefig(summary_path, dpi=300)
    if mode == "post":
        plt.savefig(os.path.join(output_dir, "20_shap_summary.png"), dpi=300)
    plt.close()

    # 2. SHAP Feature Importance Bar Plot
    plt.figure(figsize=(9, 6))
    shap.plots.bar(shap_values_fraud, max_display=12, show=False)
    plt.title(f"SHAP Mean Absolute Importance ({mode.capitalize()}-Transaction)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, f"21_shap_importance{suffix}.png")
    plt.savefig(bar_path, dpi=300)
    if mode == "post":
        plt.savefig(os.path.join(output_dir, "21_shap_feature_importance_bar.png"), dpi=300)
    plt.close()

    # 3. Individual Fraud Explanation (Waterfall)
    fraud_indices = np.where(y_test_sample == 1)[0]
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_values_fraud[fraud_idx], max_display=10, show=False)
        plt.title(f"SHAP Waterfall: Fraud Transaction ({mode.capitalize()}-Transaction)", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        fraud_wf_path = os.path.join(output_dir, f"22_shap_waterfall_fraud{suffix}.png")
        plt.savefig(fraud_wf_path, dpi=300)
        if mode == "post":
            plt.savefig(os.path.join(output_dir, "22_shap_individual_fraud_waterfall.png"), dpi=300)
        plt.close()

    # 4. Individual Legitimate Explanation (Waterfall)
    legit_indices = np.where(y_test_sample == 0)[0]
    if len(legit_indices) > 0:
        legit_idx = legit_indices[0]
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_values_fraud[legit_idx], max_display=10, show=False)
        plt.title(f"SHAP Waterfall: Legitimate Transaction ({mode.capitalize()}-Transaction)", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        legit_wf_path = os.path.join(output_dir, f"23_shap_waterfall_legit{suffix}.png")
        plt.savefig(legit_wf_path, dpi=300)
        if mode == "post":
            plt.savefig(os.path.join(output_dir, "23_shap_individual_legit_waterfall.png"), dpi=300)
        plt.close()

    print(f"[SHAP - {mode.upper()}] Visualizations generated successfully.")
    return explainer
