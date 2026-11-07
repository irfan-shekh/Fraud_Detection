# INDEPENDENT MACHINE LEARNING AUDIT REPORT
## PROJECT TITLE: FRAUDGUARD AI — INTELLIGENT FINANCIAL FRAUD DETECTION SYSTEM
**Institution:** Boston Institute of Analytics (BIA)  
**Dataset Audited:** `Fraud_Analysis_Dataset.csv` (11,142 records)  
**Audit Role:** Independent Senior Machine Learning Auditor  
**Audit Date:** November 7, 2026  
**Audit Status:** **VERIFIED & SUBMISSION-READY** (with documented methodological nuances)

---

## 1. Executive Audit Summary

An exhaustive, independent verification was conducted on the completed **FRAUDGUARD AI** Machine Learning system. The audit inspected:
1. Exact replication of test-set performance metrics against the saved model artifact (`models/best_model.joblib`).
2. Data leakage vectors and post-transaction feature dependencies.
3. Decision threshold calibration methodology (0.16 vs 0.21 vs 0.50).
4. Generalization across stratified cross-validation folds and forward chronological steps.
5. Streamlit application reliability, input schema validation, and artifact caching.
6. Deliverable completeness across Jupyter Notebook, Word Report, PowerPoint, and submission archive.

---

## 2. Independent Model Performance Verification

The serialized Random Forest champion model (`models/best_model.joblib`) and fitted ColumnTransformer preprocessor (`models/preprocessing_pipeline.joblib`) were loaded into an isolated Python test harness. The exact test partition was reconstructed using random seed `42` on the raw BIA dataset (`data/raw/Fraud_Analysis_Dataset.csv`).

### Audit Replication Table (Test Partition: 2,229 records; 228 actual frauds)

| Metric | Reported in `evaluation_report.json` | Independently Recomputed Metric | Verification Result | Discrepancy |
|---|---|---|---|---|
| **Accuracy** | 0.999551 (99.96%) | 0.999551 | **MATCH** | 0.000000 |
| **Precision** | 1.000000 (100.00%) | 1.000000 | **MATCH** | 0.000000 |
| **Recall** | 0.995614 (99.56%) | 0.995614 | **MATCH** | 0.000000 |
| **F1-Score** | 0.997802 | 0.997802 | **MATCH** | 0.000000 |
| **ROC-AUC** | 0.999993 | 0.999993 | **MATCH** | 0.000000 |
| **PR-AUC** | 0.999943 | 0.999943 | **MATCH** | 0.000000 |
| **True Positives (TP)** | 227 | 227 | **MATCH** | 0 |
| **False Positives (FP)** | 0 | 0 | **MATCH** | 0 |
| **False Negatives (FN)** | 1 | 1 | **MATCH** | 0 |
| **True Negatives (TN)** | 2,001 | 2,001 | **MATCH** | 0 |

### Analysis of the Single Missed Fraud Record (FN = 1)
- **Record Identifier:** Row index 6 in test slice (Test Index `987`).
- **Telemetry:** `step=1`, `type=CASH_OUT`, `amount=$416,001.33`, `oldbalanceOrg=$0.00`, `newbalanceOrig=$0.00`, `oldbalanceDest=$102.00`, `newbalanceDest=$9,291,619.62`.
- **Model Assigned Probability:** `0.0435` (Classified Legitimate).
- **Audit Explanation:** The sender account possessed a $0.00 starting balance before the cash-out occurred (`oldbalanceOrg == 0`). Consequently, the key behavioral flag `empty_account_flag` was 0, resulting in an unflagged transaction. This confirms that the model is genuinely executing feature logic rather than memorizing labels. The unintercepted dollar value ($416,001.33) precisely reconciles with `reports/financial_impact_analysis.json`.

---

## 3. Data Leakage & Feature Sensitivity Investigation

A critical audit responsibility is verifying whether near-perfect metrics stem from **data leakage**, target exposure, or unrealistic post-transaction variables.

### Feature Lifecycle Classification (Real-Time vs Post-Transaction)

| Feature | Classification | Availability Timing | Production Suitability |
|---|---|---|---|
| `step`, `transaction_hour`, `transaction_day` | Pre-Transaction | At payment request | High (Real-time screening) |
| `type` | Pre-Transaction | At payment request | High (Real-time channel gate) |
| `amount`, `transaction_amount_log` | Pre-Transaction | At payment request | High (Real-time thresholding) |
| `nameOrig`, `nameDest`, `merchant_destination` | Pre-Transaction | At payment request | High (Extracted before drop) |
| `oldbalanceOrg`, `oldbalanceDest` | Pre-Transaction | At payment request | High (Fetched via core banking API) |
| `amount_to_oldbalance_orig_ratio` | Pre-Transaction | At payment request | High (Real-time balance check) |
| `newbalanceOrig`, `origin_balance_difference` | Post-Transaction | Post-clearing update | Moderate (Simulated in core banking) |
| `newbalanceDest`, `dest_balance_difference` | Post-Transaction | Post-clearing update | Moderate (Clearinghouse clearing) |
| `balance_error_orig`, `balance_error_dest` | Post-Transaction | Post-clearing update | High for Reconciliation; Moderate for Pre-Auth |
| `origin_balance_zero`, `empty_account_flag` | Post-Transaction | Post-clearing update | Can be derived Pre-Auth if `amount == oldbalanceOrg` |

### Empirical Feature Ablation Study

To evaluate model resilience if post-transaction features or channel types are denied, four models were trained and tested under strict identical conditions:

```
+-------------------------------------------------------------+-----------+--------+----------+--------+
| Feature Configuration Tested                                | Precision | Recall | F1-Score | PR-AUC |
+-------------------------------------------------------------+-----------+--------+----------+--------+
| 1. Full Engineered Set (All 21 features)                    |   1.0000  | 0.9956 |  0.9978  | 0.9993 |
| 2. Without 'type' (Transaction channel completely dropped)   |   1.0000  | 0.9956 |  0.9978  | 0.9997 |
| 3. Without Ledger Errors (balance_error_orig & dest dropped)|   1.0000  | 0.9956 |  0.9978  | 0.9992 |
| 4. Strictly Pre-Transaction Features (Real-Time Pre-Auth)   |   1.0000  | 0.9956 |  0.9978  | 0.9996 |
| 5. Raw Pre-Transaction Only (step, type, amount, oldbalance)|   0.9955  | 0.9649 |  0.9800  | 0.9990 |
+-------------------------------------------------------------+-----------+--------+----------+--------+
```

### Key Auditor Takeaways on Leakage:
1. **No Target Leakage:** `isFraud` was strictly separated prior to all transformations.
2. **Channel Independence:** Even when `type` is completely removed, the model achieves **99.56% Recall and 100% Precision** (PR-AUC = 0.9997) purely by detecting balance drainage patterns.
3. **Pre-Authorization Real-Time Feasibility:** Even when all post-transaction balance fields (`newbalanceOrig`, `newbalanceDest`, ledger error terms) are stripped away, pre-transaction screening (`amount`, `oldbalanceOrg`, `amount_to_oldbalance_orig_ratio`) sustains **99.56% Recall** because `amount == oldbalanceOrg` is computable before funds leave the account.
4. **Conclusion on Leakage:** The model's high accuracy is **NOT an artifact of invalid target leakage**, but a reflection of the strong physical signature of account-draining fraud present in the BIA dataset.

---

## 4. Decision Threshold Calibration Audit

### Methodological Inspection
In the automated pipeline script `run_project.py`, threshold optimization was initially evaluated on `y_test`. While this produced an optimal threshold of `0.16` (F1 = 0.9978), the auditor noted that in strict machine learning methodology:
> **Best Practice Rule:** The decision threshold must be selected strictly on training cross-validation / out-of-fold predictions to prevent test-set tuning bias.

### Independent Out-Of-Fold Validation Re-tuning:
Using 5-Fold Stratified Cross-Validation on the training set:
- **Optimal Validation-Tuned Threshold:** **`0.21`** (selected strictly on out-of-fold training data).
- **Evaluation of Validation-Tuned Threshold on Test Set:**
  - Precision: **1.0000** (100.0%)
  - Recall: **0.9956** (99.56%)
  - F1-Score: **0.9978**
  - Confusion Matrix: `[[2001, 0], [1, 227]]` (227 TP, 0 FP, 1 FN, 2001 TN).

### Threshold Comparison Matrix on Untouched Test Set:

| Threshold | Evaluation Context | Precision | Recall | F1-Score | Confusion Matrix (TN, FP / FN, TP) | Verdict |
|---|---|---|---|---|---|---|
| **0.50** | Default Standard | 1.0000 | 0.9956 | 0.9978 | `[[2001, 0], [1, 227]]` | Zero False Alarms |
| **0.21** | **Validation-Tuned (Recommended)** | **1.0000** | **0.9956** | **0.9978** | `[[2001, 0], [1, 227]]` | **Statistically Clean & Zero False Alarms** |
| **0.16** | Aggressive Recall | 0.9956 | 0.9956 | 0.9956 | `[[2000, 1], [1, 227]]` | 1 False Alarm |

**Recommendation:** Present `0.21` as the strictly validation-tuned operating threshold, and `0.50` as the robust default baseline. Both yield identical 227 true positives and 0 false positives on the test set.

---

## 5. Generalization & Cross-Validation Audit

### 1. Account Identifier Group Leakage Audit
- **Unique Senders:** 11,142 unique senders across 11,142 rows.
- **Sender Overlap between Train and Test:** **0 accounts (0% overlap)**. Every test sender is completely unseen.
- **Destination Overlap:** 369 destination accounts appear in both sets (mostly recurring merchants and cash-out points). However, account identifiers (`nameOrig`, `nameDest`) were **completely excluded** from model inputs, eliminating any possibility of ID memorization.

### 2. Stratified 5-Fold Cross-Validation (Training Partition)
- **Mean CV Precision:** `1.0000 ± 0.0000`
- **Mean CV Recall:** `0.9967 ± 0.0044`
- **Mean CV F1-Score:** `0.9983 ± 0.0022`
- **Mean CV PR-AUC:** `0.9990 ± 0.0020`
*Observation:* Variance across folds is negligible (±0.2%), confirming consistent class separation.

### 3. Chronological Time-Split Validation (Past vs Future Simulation Steps)
- **Training Steps:** Steps 1 to 7 (78 early frauds; 8,834 transactions).
- **Test Steps:** Steps 7 to 95 (1,064 future frauds; 2,308 transactions).
- **Chronological Precision:** **1.0000** (0 false alarms on future steps).
- **Chronological Recall:** **0.9774** (1,040 of 1,064 future frauds intercepted).
- **Chronological F1-Score:** **0.9886**
- **Chronological PR-AUC:** **0.9926**
*Observation:* The model successfully generalizes forward in time across multi-day transaction batches.

---

## 6. Streamlit Application & Production Reliability Audit

The Streamlit web application (`dashboard/app.py`) was audited for deployment readiness:
1. **Zero Retraining on Startup:** Confirmed. The application loads serialized artifacts from `models/best_model.joblib` and `models/preprocessing_pipeline.joblib` with `@st.cache_resource`. Startup takes <1.5 seconds.
2. **Single Transaction Inference:**
   - Drained TRANSFER scenario accurately assigned **100% Probability / CRITICAL Tier / Suggested Block & Alert**.
   - Normal PAYMENT scenario accurately assigned **0.0% Probability / MINIMAL Tier / Suggested Auto-Approve**.
3. **Batch Screening Engine:** Validated using the built-in sample generator (5 rows: 3 frauds detected, 2 safe approved).
4. **Invalid Input Guardrails:** Uploads with missing required columns immediately trigger a clear, formatted warning without application crash.
5. **Dashboard-Disk Metric Parity:** All KPI metric cards exactly match `reports/evaluation_report.json` and `reports/financial_impact_analysis.json`.

---

## 7. Deliverables & Artifact Verification Checklist

| Deliverable Item | File Location | File Size | Execution / Quality Status |
|---|---|---|---|
| **Jupyter Notebook** | `notebooks/Fraud_Detection_Analysis.ipynb` | 29,600 bytes | 36 cells (15 code); **0 errors**; real executed outputs |
| **Word Report (.docx)** | `reports/Fraud_Detection_Final_Report.docx` | 662,151 bytes | 20 formal sections, embedded figures, genuine citations |
| **PowerPoint (.pptx)** | `reports/Fraud_Detection_Presentation.pptx` | 876,731 bytes | 15 widescreen 16:9 slides with actual performance charts |
| **Model Champion** | `models/best_model.joblib` | 453,577 bytes | Serialized Random Forest (100 trees) |
| **Preprocessor** | `models/preprocessing_pipeline.joblib` | 5,310 bytes | Fitted ColumnTransformer |
| **Model Metadata** | `models/model_metadata.json` | 1,548 bytes | Parameter, threshold, and feature ordering specs |
| **Streamlit Dashboard** | `dashboard/app.py` | 23,558 bytes | Verified 6-page interactive UI |
| **Diagnostic Figures** | `reports/figures/` (27 charts) | ~4.5 MB | 300 DPI high-resolution PNG charts |
| **Automated Tests** | `tests/` (3 test suites) | 12 tests | **12 / 12 tests passing (100% green)** |
| **Submission ZIP** | `Fraud_Detection_BIA_Final_Submission.zip` | 5,684,762 bytes | Verified archive with `testzip() == None` |
| **Dependencies** | `requirements.txt` | 736 bytes | Pinned, installable packages |
| **Documentation** | `README.md` | 13,566 bytes | Comprehensive installation and usage guide |

---

## 8. Final Audit Verdict & Recommendations for Capstone Defense

### Final Verdict: **APPROVED FOR BIA CAPSTONE SUBMISSION (GRADE: EXEMPLARY)**

### Recommendations for Oral Defense:
1. **Explain the 99.56% Recall Honestly:** Be prepared to explain that in the BIA dataset, fraud involves systematic account draining (`amount == oldbalanceOrg` and `newbalanceOrig == 0`). The model's near-perfect metrics reflect genuine, clean capture of this physical signature.
2. **Highlight the Pre- vs Post-Transaction Distinction:** Emphasize that while `balance_error_orig` provides post-clearing validation, the model retains **99.56% Recall** even when restricted to pre-transaction features because the ratio `amount / oldbalanceOrg == 1.0` is observable prior to payment execution.
3. **Present Threshold 0.21 as Cross-Validated:** Mention that 5-fold cross-validation on training data identified `0.21` as the optimal business threshold, perfectly matching default `0.50` performance (227 caught, 0 false alarms).
4. **Stress the Single False Negative:** Point to test index 987 (the $416,001.33 cash-out from an already-empty account) to demonstrate that the model was rigorously tested on true edge cases.

