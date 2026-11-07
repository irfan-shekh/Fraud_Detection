"""
End-to-End Execution Pipeline for BIA Fraud Detection Project.
Orchestrates data audit, EDA, preprocessing, model training, evaluation, SHAP, and financial impact.
"""

import os
import sys
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
import json
import numpy as np
import pandas as pd

# Add workspace to path
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_raw_data
from src.data_validation import run_data_validation
from src.eda import generate_all_eda_figures
from src.feature_engineering import engineer_features
from src.preprocessing import prepare_and_save_data, get_feature_columns
from src.train import (
    train_and_compare_models,
    compare_imbalance_techniques,
    compare_feature_engineering_impact,
    save_best_model_and_artifacts,
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
    print("   PROJECT TITLE: FRAUD DETECTION END-TO-END MACHINE LEARNING SYSTEM")
    print("=" * 80)

    # 1. DATA VALIDATION & QUALITY AUDIT
    print("\n>>> PHASE 1: DATA INGESTION & DATA QUALITY AUDIT")
    data_report = run_data_validation()

    # 2. EXPLORATORY DATA ANALYSIS (EDA)
    print("\n>>> PHASE 2: EXPLORATORY DATA ANALYSIS (15 FIGURES)")
    generate_all_eda_figures()

    # 3. PREPROCESSING & SPLITTING
    print("\n>>> PHASE 3 & 4: FEATURE ENGINEERING & DATA PREPROCESSING")
    data_dict = prepare_and_save_data()
    X_train = data_dict["X_train"]
    X_test = data_dict["X_test"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]
    preprocessor = data_dict["preprocessor"]

    # 4. ABLATION STUDY: FEATURE ENGINEERING IMPACT
    print("\n>>> PHASE 3 ABLATION: FEATURE ENGINEERING IMPACT")
    df_ablation = compare_feature_engineering_impact()
    df_ablation.to_csv("reports/feature_engineering_comparison.csv", index=False)

    # 5. CLASS IMBALANCE STUDY
    print("\n>>> PHASE 5: CLASS IMBALANCE HANDLING COMPARISON")
    df_imbalance = compare_imbalance_techniques(X_train, y_train, X_test, y_test, preprocessor)
    df_imbalance.to_csv("reports/imbalance_techniques_comparison.csv", index=False)

    # 6. MODEL TRAINING & HYPERPARAMETER TUNING
    print("\n>>> PHASE 6 & 7: MODEL TRAINING & HYPERPARAMETER OPTIMIZATION")
    df_models, trained_models, best_model = train_and_compare_models(
        X_train, y_train, X_test, y_test, preprocessor
    )
    df_models.to_csv("reports/model_comparison.csv", index=False)

    # 7. MODEL EVALUATION & DIAGNOSTICS
    print("\n>>> PHASE 8: COMPREHENSIVE MODEL EVALUATION")
    plot_model_comparison_bars(df_models)

    # Compute probabilities on test set for all models
    model_probs = {}
    X_te = preprocessor.transform(X_test)
    X_tr = preprocessor.transform(X_train)

    for name, model in trained_models.items():
        if name == "Isolation Forest":
            scores = -model.decision_function(X_te)
            probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        else:
            probs = model.predict_proba(X_te)[:, 1]
        model_probs[name] = probs

    plot_combined_roc_and_pr_curves(model_probs, y_test.values)

    # Optimal threshold for best model: Tuned strictly on Training Out-Of-Fold Cross-Validation
    best_model_name = df_models.iloc[0]["Model"]
    best_probs = model_probs[best_model_name]

    print("[EVALUATION] Calibrating decision threshold on Training Out-of-Fold (OOF) CV predictions...")
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_train))
    for tr_idx, val_idx in cv.split(X_tr, y_train):
        clone_clf = clone(best_model)
        clone_clf.fit(X_tr[tr_idx], y_train.iloc[tr_idx])
        oof_probs[val_idx] = clone_clf.predict_proba(X_tr[val_idx])[:, 1]

    best_th, df_thresh = find_optimal_threshold(y_train.values, oof_probs)
    print(f"[EVALUATION] Optimal validation threshold selected: {best_th:.4f}")
    plot_threshold_tradeoff(df_thresh, best_th)

    # Baseline vs Optimal Threshold Metrics evaluated on untouched Test Set
    metrics_default = evaluate_predictions(y_test.values, best_probs, threshold=0.50)
    metrics_optimal = evaluate_predictions(y_test.values, best_probs, threshold=best_th)

    # Plot Confusion Matrices
    plot_confusion_matrix_heatmap(
        metrics_default["confusion_matrix"],
        best_model_name,
        "reports/figures/16a_confusion_matrix_default.png",
        threshold=0.50,
    )
    plot_confusion_matrix_heatmap(
        metrics_optimal["confusion_matrix"],
        best_model_name,
        "reports/figures/16b_confusion_matrix_optimal.png",
        threshold=best_th,
    )

    # Save comprehensive evaluation report
    eval_report = {
        "best_model": best_model_name,
        "default_threshold_0_50": metrics_default,
        "optimal_threshold": metrics_optimal,
        "all_models_summary": df_models.to_dict(orient="records"),
    }
    with open("reports/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=4)
    print("[EVALUATION] Evaluation report saved to 'reports/evaluation_report.json'.")

    # 8. SHAP EXPLAINABILITY
    print("\n>>> PHASE 9: MODEL EXPLAINABILITY USING SHAP")
    try:
        run_shap_analysis(best_model, preprocessor, X_train, X_test, y_test)
    except Exception as e:
        print(f"[SHAP] Error running SHAP: {e}. Falling back to Random Forest explainer if available.")

    # 9. FINANCIAL IMPACT ANALYSIS
    print("\n>>> PHASE 10: FINANCIAL RISK & BUSINESS IMPACT ANALYSIS")
    amounts_test = X_test["amount"].values
    y_pred_opt = (best_probs >= best_th).astype(int)

    base_impact = compute_financial_impact(y_test.values, y_pred_opt, amounts_test)
    scenario_impact = evaluate_cost_scenarios(y_test.values, y_pred_opt, amounts_test)
    plot_financial_impact_visualizations(base_impact, scenario_impact)

    full_financial_report = {
        "base_case": base_impact,
        "scenarios": scenario_impact,
    }
    with open("reports/financial_impact_analysis.json", "w", encoding="utf-8") as f:
        json.dump(full_financial_report, f, indent=4)
    print("[FINANCIAL] Financial analysis report saved to 'reports/financial_impact_analysis.json'.")

    # 10. SAVE BEST MODEL & METADATA
    print("\n>>> PHASE 12: MODEL SERIALIZATION")
    feature_names = get_preprocessed_feature_names(preprocessor, X_train)
    save_best_model_and_artifacts(
        best_model=best_model,
        best_model_name=best_model_name,
        preprocessor=preprocessor,
        feature_names=feature_names,
        evaluation_summary=metrics_optimal,
        threshold=best_th,
    )

    print("\n" + "=" * 80)
    print("   END-TO-END PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
