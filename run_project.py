"""
End-to-End Execution Pipeline for BIA Fraud Detection Project.
Orchestrates data audit, EDA, dual pre- and post-transaction preprocessing,
leak-free cross-validation training, threshold calibration, SHAP explainability,
and financial risk modeling.
"""

import os
import sys
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
import json
import joblib
import numpy as np
import pandas as pd

# Add workspace to path
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_raw_data
from src.data_validation import run_data_validation
from src.eda import generate_all_eda_figures
from src.preprocessing import prepare_and_save_data
from src.train import (
    train_and_benchmark_suite,
    save_system_artifacts,
)
from src.evaluate import (
    evaluate_predictions,
    find_optimal_threshold,
    plot_threshold_tradeoff,
    plot_confusion_matrix_heatmap,
    plot_combined_roc_and_pr_curves,
    plot_model_comparison_bars,
)
from src.explainability import run_shap_analysis, get_preprocessed_feature_names
from src.financial_impact import (
    compute_financial_impact,
    evaluate_cost_scenarios,
    plot_financial_impact_visualizations,
)


def run_full_pipeline():
    print("=" * 80)
    print("   BOSTON INSTITUTE OF ANALYTICS (BIA) - CAPSTONE PROJECT")
    print("   FRAUDGUARD AI: DUAL PRE- AND POST-TRANSACTION FRAUD DETECTION SYSTEM")
    print("=" * 80)

    # 1. DATA VALIDATION & QUALITY AUDIT
    print("\n>>> PHASE 1: DATA INGESTION & DATA QUALITY AUDIT")
    data_report = run_data_validation()

    # 2. EXPLORATORY DATA ANALYSIS (EDA)
    print("\n>>> PHASE 2: EXPLORATORY DATA ANALYSIS (15 FIGURES)")
    generate_all_eda_figures()

    # 3. PREPROCESSING & SPLITTING (PRE & POST)
    print("\n>>> PHASE 3 & 4: FEATURE ENGINEERING & DATA PREPROCESSING")
    data_dict = prepare_and_save_data()

    # 4. MODEL TRAINING & LEAK-FREE CROSS-VALIDATION
    print("\n>>> PHASE 5 & 6: MODEL TRAINING & BENCHMARKING (DUAL PIPELINES)")
    pre_results = train_and_benchmark_suite(
        data_dict["pre"]["X_train"], data_dict["pre"]["y_train"],
        data_dict["pre"]["X_test"], data_dict["pre"]["y_test"],
        mode="pre", random_state=42
    )
    post_results = train_and_benchmark_suite(
        data_dict["post"]["X_train"], data_dict["post"]["y_train"],
        data_dict["post"]["X_test"], data_dict["post"]["y_test"],
        mode="post", random_state=42
    )

    # 5. SERIALIZE ARTIFACTS
    print("\n>>> PHASE 7: SAVING MODEL ARTIFACTS AND EVALUATION REPORTS")
    save_system_artifacts(pre_results, post_results)

    # 6. DIAGNOSTIC PLOTS
    print("\n>>> PHASE 8: GENERATING EVALUATION DIAGNOSTIC PLOTS")
    plot_model_comparison_bars(post_results["benchmark_table"])

    # ROC & PR curves for Post-Transaction model
    y_test_post = data_dict["post"]["y_test"].values
    post_probs = post_results["test_probabilities"]
    plot_combined_roc_and_pr_curves(
        {post_results["champion_name"]: post_probs}, y_test_post
    )

    # Confusion Matrices for Post-Transaction model
    plot_confusion_matrix_heatmap(
        post_results["default_metrics"]["confusion_matrix"],
        post_results["champion_name"],
        "reports/figures/16a_confusion_matrix_default.png",
        threshold=0.50,
    )
    plot_confusion_matrix_heatmap(
        post_results["optimal_metrics"]["confusion_matrix"],
        post_results["champion_name"],
        "reports/figures/16b_confusion_matrix_optimal.png",
        threshold=post_results["optimal_threshold"],
    )

    # Threshold curve
    plot_threshold_tradeoff(
        post_results["df_threshold_curve"],
        post_results["optimal_threshold"],
        "reports/figures/16_threshold_analysis.png"
    )

    # 7. SHAP EXPLAINABILITY (BOTH PRE & POST)
    print("\n>>> PHASE 9: MODEL EXPLAINABILITY USING SHAP (PRE & POST)")
    try:
        run_shap_analysis(
            post_results["classifier"],
            post_results["preprocessor"],
            data_dict["post"]["X_train"],
            data_dict["post"]["X_test"],
            data_dict["post"]["y_test"],
            mode="post",
        )
    except Exception as e:
        print(f"[SHAP] Post-transaction SHAP warning: {e}")

    try:
        run_shap_analysis(
            pre_results["classifier"],
            pre_results["preprocessor"],
            data_dict["pre"]["X_train"],
            data_dict["pre"]["X_test"],
            data_dict["pre"]["y_test"],
            mode="pre",
        )
    except Exception as e:
        print(f"[SHAP] Pre-transaction SHAP warning: {e}")

    # 8. FINANCIAL IMPACT ANALYSIS
    print("\n>>> PHASE 10: FINANCIAL RISK & BUSINESS IMPACT ANALYSIS")
    amounts_test = data_dict["post"]["X_test"]["amount"].values
    y_pred_opt = (post_probs >= post_results["optimal_threshold"]).astype(int)

    base_impact = compute_financial_impact(y_test_post, y_pred_opt, amounts_test)
    scenario_impact = evaluate_cost_scenarios(y_test_post, y_pred_opt, amounts_test)
    plot_financial_impact_visualizations(base_impact, scenario_impact)

    full_financial_report = {
        "base_case": base_impact,
        "scenarios": scenario_impact,
    }
    with open("reports/financial_impact_analysis.json", "w", encoding="utf-8") as f:
        json.dump(full_financial_report, f, indent=4)
    print("[FINANCIAL] Financial analysis report saved to 'reports/financial_impact_analysis.json'.")

    print("\n" + "=" * 80)
    print("   END-TO-END PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
