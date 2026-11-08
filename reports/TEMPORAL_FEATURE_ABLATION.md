# 🛡️ TEMPORAL FEATURE ABLATION EXPERIMENT REPORT — FRAUDGUARD AI
### Senior Data Scientist Independent Investigation
**Date:** November 8, 2026  
**Institution:** Boston Institute of Analytics (BIA)  
**Dataset:** Official BIA Benchmark (`Fraud_Analysis_Dataset.csv`)  
**Evaluation Protocol:** Out-Of-Time (OOT) Chronological Holdout (Train: Steps 1–6; Test: Steps 7–95)

---

## 1. Executive Summary

This independent feature ablation experiment tests the hypothesis:  
> *"Does the high temporal generalization performance of FraudGuard AI depend heavily on dataset-specific temporal features (`step`, `transaction_hour`, `transaction_day`), or is it driven by invariant behavioral and account-state signals?"*

To answer this conclusively, three controlled model configurations were evaluated using the identical chronological training horizon (Steps 1–6: 7,164 records; 66 frauds) and evaluated on the identical untouched out-of-time (OOT) test partition (Steps 7–95: 3,978 records; 1,076 frauds; $1,326,231,582.08 total fraud exposure).

### Summary of Results:
- **Removing temporal features DOES NOT degrade performance.** In fact, it **improves** Out-Of-Time PR-AUC from **0.9898 to 0.9929** and increases fraud recall from **97.77% to 98.14%** (capturing 4 additional fraudulent transactions).
- The model **does not rely on dataset-specific temporal shortcuts**. Its predictive power is overwhelmingly anchored in invariant behavioral mechanics: full account liquidation intent (`empty_account_flag`), channel vulnerability (`TRANSFER` / `CASH_OUT`), and monetary volume (`amount`).
- **Production Recommendation:** **Experiment B (Ablated Temporal Features)** is the most defensible configuration for long-term production deployment because it eliminates vulnerability to time-index drift.

---

## 2. Experimental Design & Methodology

All three experiments adhered strictly to the following leakage-prevention protocol:
1. **Strict Temporal Partitioning:** Historical training window strictly bounded to `step <= 6` (hours 1–6). Out-of-time evaluation horizon strictly bounded to `step >= 7` (hours 7–95). This guarantees $\max(\text{train\_step}) = 6 < \min(\text{test\_step}) = 7$ with zero temporal overlap.
2. **Preprocessing Isolation:** `ColumnTransformer` (median imputation, robust scaling, one-hot encoding) was fitted **strictly on training folds** inside a unified scikit-learn pipeline.
3. **Training-Only OOF Threshold Optimization:** Decision thresholds were calibrated exclusively on 5-fold cross-validation out-of-fold training probabilities using a grid search over F1-score and business recall objectives. **The test set was never used for threshold tuning.**
4. **Untouched OOT Evaluation:** Frozen pipelines and thresholds were evaluated once on the untouched test partition.

### Controlled Configurations:
- **Experiment A (Full Pre-Transaction Features):** Baseline pre-authorization feature set (13 unencoded / 16 encoded features), including `step`, `transaction_hour`, `transaction_day`, and `amount_to_oldbalance_orig_ratio`.
- **Experiment B (Remove Temporal Features):** Excludes all temporal features: `step`, `transaction_hour`, and `transaction_day` (10 unencoded / 13 encoded features).
- **Experiment C (Remove Temporal & Balance Ratio Features):** Excludes `step`, `transaction_hour`, `transaction_day`, AND `amount_to_oldbalance_orig_ratio` (9 unencoded / 12 encoded features).

---

## 3. Empirical Results & Performance Comparison Table

*Out-Of-Time (OOT) Evaluation on Steps 7–95: N = 3,978 transactions (2,902 Legitimate, 1,076 Fraudulent)*  
*Total Out-Of-Time Fraud Capital Exposure: $1,326,231,582.08*

| Metric | Experiment A: Full Pre-Tx | Experiment B: No Temporal | Experiment C: No Temporal & No Ratio | Delta (Exp B vs Exp A) |
|---|---|---|---|---|
| **Features Count (Raw / Encoded)** | 13 / 16 | 10 / 13 | 9 / 12 | -3 features |
| **Excluded Features** | None | `step`, `hour`, `day` | `step`, `hour`, `day`, `ratio` | — |
| **OOF Optimal Threshold** | **0.1500** | **0.2100** | **0.2600** | +0.0600 |
| **OOT Accuracy** | 0.9940 (99.40%) | **0.9947 (99.47%)** | **0.9947 (99.47%)** | **+0.07%** |
| **OOT Precision** | **1.0000 (100.00%)** | 0.9991 (99.91%) | 0.9991 (99.91%) | -0.09% |
| **OOT Recall** | 0.9777 (97.77%) | **0.9814 (98.14%)** | **0.9814 (98.14%)** | **+0.37%** |
| **OOT F1-Score** | 0.9887 | **0.9902** | **0.9902** | **+0.0015** |
| **OOT ROC-AUC** | 0.9927 | **0.9950** | 0.9940 | **+0.0023** |
| **OOT PR-AUC** | 0.9898 | **0.9929** | 0.9916 | **+0.0031** |
| **True Positives (TP)** | 1,052 | **1,056** | **1,056** | **+4 Frauds Caught** |
| **False Positives (FP)** | **0** | 1 | 1 | +1 False Alarm |
| **True Negatives (TN)** | **2,902** | 2,901 | 2,901 | -1 |
| **False Negatives (FN)** | 24 | **20** | **20** | **-4 Missed Frauds** |
| **Frauds Intercepted** | 1,052 / 1,076 | **1,056 / 1,076** | **1,056 / 1,076** | **+4 Caught** |
| **Flagged Fraud Value ($)** | $1,316,538,121.50 | **$1,318,255,376.53** | **$1,318,255,376.53** | **+$1,717,255.03** |
| **Capital Capture Rate** | 99.27% | **99.40%** | **99.40%** | **+0.13%** |

---

## 4. In-Depth Technical & Behavioral Analysis

### A. Do temporal features materially influence predictions?
**No.** The empirical evidence demonstrates that temporal features (`step`, `transaction_hour`, `transaction_day`) contribute negligible unique discriminative power once behavioral and account-state indicators are present. When all temporal features are completely removed (Experiment B), PR-AUC increases from 0.9898 to 0.9929, and recall increases from 97.77% to 98.14%.

### B. Why does removing temporal features improve generalization?
In tree-based ensembles (such as Random Forest), monotonically increasing time counters (`step`) and cyclical hours create partition splits that reflect sample-specific historical schedules (e.g., "frauds occurred at step 4 between 2:00 AM and 4:00 AM").  
When tested on unseen future periods (e.g., Steps 7 through 95), these time-bound rules can cause trees to assign lower probabilities to legitimate high-value transactions or fail to catch attacks occurring at new hours.  
Removing temporal features forces the decision trees to split exclusively on **structural financial anomalies**:
1. `empty_account_flag`: Did the initiator request to transfer exactly 100% of their balance?
2. `type`: Is the transaction routing through high-velocity cash-out channels (`TRANSFER`, `CASH_OUT`)?
3. `amount` & `transaction_amount_log`: Does the monetary magnitude deviate from normal retail baselines?
4. `orig_has_zero_balance` & `oldbalanceOrg`: Is the account being drained from a funded state?

Because these behavioral patterns are intrinsic to the mechanics of unauthorized account takeovers, they remain completely invariant across operational days and hours.

### C. Does the model depend excessively on dataset-specific temporal patterns?
**No.** The ablation experiment proves that FraudGuard AI is not artificially propped up by temporal artifacts. Even with zero temporal information, the model achieves **98.14% OOT recall** and **99.91% precision**, capturing **$1.318 Billion** out of $1.326 Billion in simulated fraud across 88 consecutive forward hours.

### D. What role does the balance-ratio feature play?
In Experiment C, `amount_to_oldbalance_orig_ratio` was ablated in addition to temporal features. The model maintained identical recall (98.14%), identical F1-score (0.9902), and captured the exact same 1,056 frauds. This occurs because `empty_account_flag` (a discrete indicator of 100% balance depletion) already captures the critical point of the continuous ratio distribution.

---

## 5. Production Deployment Recommendation

### Most Defensible Configuration: Experiment B (Ablated Temporal Features)

| Feature Configuration | Recommendation | Production Rationale |
|---|---|---|
| **Experiment A (Full Pre-Tx)** | Acceptable with Monitoring | Zero false positives in baseline OOT holdout, but susceptible to long-term drift if transaction scheduling shifts. |
| **Experiment B (No Temporal)** | **RECOMMENDED FOR PRODUCTION** | **Superior generalization (PR-AUC 0.9929, F1 0.9902). Eliminates time-counter drift completely. Most robust to seasonality and shifting transaction velocity.** |
| **Experiment C (No Temporal / No Ratio)** | Robust Baseline Alternative | High performance (F1 0.9902), slightly lower PR-AUC (0.9916) than Exp B. |

### Architectural Conclusion:
For real-world electronic payment switches, **Experiment B represents the gold-standard design**. By decoupling fraud risk scoring from the wall-clock time and chronological transaction sequence, the financial institution gains a policy engine whose decision boundaries remain stable regardless of daylight savings shifts, seasonal retail volume spikes, or multi-year operational horizons.
