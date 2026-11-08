# 🛡️ FraudGuard AI: Comprehensive Project & Feature Audit Report

**Auditor:** Senior Data Scientist, MLOps Engineer & ML Auditor  
**Date:** November 8, 2026  
**Institution:** Boston Institute of Analytics (BIA)  
**Dataset:** Official BIA Benchmark (`Fraud_Analysis_Dataset.csv` — 11,142 records)

---

## 1. Executive Summary

This independent audit evaluates the **FraudGuard AI** machine learning system prior to model separation, pipeline refactoring, and dashboard enhancement. While the previous implementation achieved high benchmark scores on post-transaction data, this audit identifies critical architectural, operational, and data-leakage improvements necessary for authentic enterprise fraud prevention.

---

## 2. Key Audit Findings & Identified Discrepancies

### Finding 1: Training and Inference Feature Mismatches
- **Observation:** `src/feature_engineering.py` constructs 14 engineered features. However, `models/model_metadata.json` records 24 post-encoding features.
- **Root Cause:** One-hot encoding of `type` expands 5 categories into 4 binary indicators (`drop="first"`), increasing dimensionality from 20 numerical + 1 categorical to 24 encoded features.
- **Remediation:** Enforce explicit feature schema tracking with exact column names and types preserved in model metadata for both pre- and post-transaction models.

### Finding 2: Target Leakage and Post-Transaction Feature Dependencies
- **Observation:** The single champion model used `newbalanceOrig`, `newbalanceDest`, `balance_error_orig`, `balance_error_dest`, `origin_balance_zero`, and `destination_balance_zero`.
- **Operational Reality:** In an instantaneous payment gateway or card authorization switch (pre-transaction), updated account balances (`newbalanceOrig` / `newbalanceDest`) **do not exist** because the transaction has not yet cleared or settled.
- **Remediation (Phase 2):** Decouple detection into two dedicated pipelines:
  1. **Pre-Transaction Fraud Screening Model:** Operates solely on data available at transaction initiation (`step`, `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`, and pre-execution derived indicators).
  2. **Post-Transaction Fraud Detection Model:** Leverages post-clearing ledger balances for post-processing risk audits and batch dispute reconciliation.

### Finding 3: Cross-Validation Preprocessing Leakage
- **Observation:** In `src/train.py`, `preprocessor.fit(X_train)` was executed once across the entire training partition before passing `X_tr` to `RandomizedSearchCV` and cross-validation splits.
- **Impact:** Validation folds in CV indirectly had their median imputation and robust scaling medians informed by the training fold data.
- **Remediation (Phase 3):** Encapsulate `ColumnTransformer` and the classifier inside a unified `imblearn.pipeline.Pipeline`. Preprocessing is fitted strictly on the training partition of each CV fold.

### Finding 4: Threshold Selection Methodology
- **Observation:** The previous threshold calibration calculated out-of-fold predictions on already-transformed features.
- **Remediation (Phase 5):** Out-of-fold probability estimation will be performed on raw training folds using fold-isolated pipelines. Threshold selection will be documented and evaluated against a documented business objective before one-time testing on the untouched test partition.

### Finding 5: Inconsistent Feature Counts Across Documentation
- **Observation:** Documentation referenced "12 features", "19 features", and "24 features" interchangeably.
- **Audited Truth:**
  - Raw columns: 10 (`step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`).
  - Raw feature candidates (excluding target & identifiers): 7.
  - Engineered features added: 14.
  - Post-transaction unencoded feature space: 21 features (1 categorical + 20 numerical).
  - Post-transaction encoded feature space: 24 features (4 one-hot encoded + 20 numerical).
  - Pre-transaction feature space: 8 unencoded features (1 categorical + 7 numerical).

### Finding 6: Dashboard Metric Dynamic Loading
- **Observation:** `dashboard/app.py` previously contained static fallback values in certain overview cards.
- **Remediation (Phase 7):** All metrics, confusion matrices, and model comparison parameters must load dynamically from serialized evaluation reports.

### Finding 7: Input Validation Deficiencies
- **Observation:** Single and batch prediction utilities relied on pandas type coercion without defensive checks for negative amounts, negative balances, invalid transaction types, or invalid steps.
- **Remediation (Phase 8):** Build reusable input validation functions in `src/data_validation.py` / `src/validation.py` that validate schema, types, and ranges with clear user-facing error messages.

### Finding 8: Explanation Separation (Rules vs SHAP)
- **Observation:** `predict_single_transaction` mixed hardcoded heuristics (e.g. `origin_balance_zero == 1`) with model risk scores, creating the perception that the rule was the model's reason.
- **Remediation (Phase 7):** Clearly label rule-based checks as deterministic policy rules and reserve feature contribution statements for SHAP value attributions.

### Finding 9: Model Artifact Cross-Version Compatibility
- **Observation:** Cross-version unpickling between scikit-learn versions caused `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`.
- **Remediation:** Retain defensive in-memory patching in `dashboard/utils.py` and serialize independent pre-transaction and post-transaction models with clean metadata.

### Finding 10: Inconsistencies Between Saved Models and Evaluation Reports
- **Observation:** Separate evaluation reports are required for pre-transaction and post-transaction models to prevent comparing dissimilar feature sets without clear documentation.
- **Remediation (Phase 4):** Persist separate reports: `reports/pre_transaction_evaluation.json` and `reports/post_transaction_evaluation.json`.

---

## 3. Approved Implementation Roadmap

```
1. Feature Audit & Dictionary      ──> reports/feature_audit.json [COMPLETED]
2. Pre/Post Feature Engineering     ──> src/feature_engineering.py (Pre vs Post pipelines)
3. Input Validation Module         ──> src/data_validation.py / src/validation.py
4. Preprocessing & Leak-Free CV     ──> src/preprocessing.py & src/train.py
5. Independent Model Training       ──> models/pre_transaction_* & models/post_transaction_*
6. Independent Threshold Tuning    ──> reports/threshold_analysis.json
7. Reports & Evaluation Sync        ──> reports/pre_transaction_evaluation.json & post_...
8. Dashboard 7-Page Overhaul        ──> dashboard/app.py, utils.py, components.py
9. Comprehensive Pytest Suite       ──> tests/ (15+ test cases)
10. Documentation & Final Sync     ──> README.md & submission files
```
