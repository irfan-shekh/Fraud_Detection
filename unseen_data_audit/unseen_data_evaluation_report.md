# Phase 3: Completely Unseen Data Evaluation Report
**FraudGuard AI — Independent Stress Testing Against 12,000 Unseen Transactions**  
*Auditor: Senior Machine Learning Auditor & Production Risk Assessor*  
*Date: November 2026*  
*Project Repository: `c:\Projects\Fraud_Detection`*

---

## 1. Executive Summary

This report delivers the empirical findings of **Test A (Original Benchmark Holdout)** versus **Test B (Independent 12,000 Unseen Transactions)**.

To rigorously determine whether the models generalize or merely exploit synthetic artifacts, an independent evaluation dataset of **12,000 unseen transactions** (`unseen_data_audit/data/unseen_transactions_independent.csv`) was generated using fixed seed `2026`. This dataset relaxes the artificial synthetic shortcuts:
1. **Realistic Drain Ratios:** Legitimate users occasionally transfer their full account balance ($5\%$ account closure rate), while fraudsters execute fractional transfers ($20\% - 75\%$ balance drain) to evade simple threshold rules.
2. **Realistic Time Velocity:** Legitimate transactions continue across all steps ($1$ through $150$), breaking the artificial step segregation.
3. **No Row or Vector Contamination:** Generated completely independently of the training set.

---

## 2. Quantitative Comparison: Test A vs. Test B

The table below contrasts the baseline performance on the original holdout against the independent unseen dataset:

| Metric | Pre-Tx RF (Test A: Original Holdout) | Pre-Tx RF (Test B: 12k Unseen Data) | Post-Tx XGB (Test A: Original Holdout) | Post-Tx XGB (Test B: 12k Unseen Data) | Performance Variance ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Test Samples ($N$)** | 2,229 | 12,000 | 2,229 | 12,000 | $+438\%$ sample scale |
| **Fraud Sample Count** | 228 (10.23%) | 1,008 (8.40%) | 228 (10.23%) | 1,008 (8.40%) | Realistic bank prevalence |
| **Threshold ($\tau$)** | $0.21$ | $0.21$ | $0.50$ | $0.50$ | Frozen production thresholds |
| **Recall (Sensitivity)** | **99.56%** | **97.22%** | **99.56%** | **97.22%** | **-2.34% (Remarkably Stable)** |
| **Precision** | **100.00%** | **8.56%** | **99.56%** | **8.56%** | **-91.44% (Severe False Positive Spike)** |
| **F1-Score** | **0.9978** | **0.1573** | **0.9956** | **0.1573** | **-0.8405** |
| **PR-AUC** | **0.9996** | **0.2207** | **0.9970** | **0.2652** | **-0.7318 to -0.7789** |
| **ROC-AUC** | **1.0000** | **0.7128** | **0.9990** | **0.7973** | **-0.2017 to -0.2872** |
| **True Positives (TP)** | 227 | 980 | 227 | 980 | Detects 980 out of 1,008 frauds |
| **False Negatives (FN)** | 1 | 28 | 1 | 28 | Only 28 frauds missed |
| **False Positives (FP)** | **0** | **10,471** | **1** | **10,471** | **Massive operational false alarm volume** |
| **True Negatives (TN)** | 2,001 | 521 | 2,000 | 521 | Severe legitimate transaction rejection |
| **Total Fraud Exposure ($)**| \$235,846,526.37 | \$83,702,738.65 | \$235,846,526.37 | \$83,702,738.65 | Full dollar volume at risk |
| **Intercepted Fraud ($)** | \$235,430,525.04 | \$82,041,261.54 | \$235,430,525.04 | \$82,041,261.54 | Dollar volume successfully caught |
| **Capital Capture Rate** | **99.82%** | **98.02%** | **99.82%** | **98.02%** | **High financial protection preserved** |

---

## 3. Confusion Matrix Analysis

### Test A: Original Holdout (Pre-Tx RF)
```
                  Predicted Legitimate    Predicted Fraud
Actual Legitimate        2,001 (TN)              0 (FP)
Actual Fraud                 1 (FN)            227 (TP)
```
*Zero false positives; near-perfect discrimination.*

### Test B: Independent Unseen Data (Pre-Tx RF)
```
                  Predicted Legitimate    Predicted Fraud
Actual Legitimate          521 (TN)         10,471 (FP)
Actual Fraud                28 (FN)            980 (TP)
```
*Catastrophic false positive spike. $95.26\%$ of legitimate transactions are flagged as fraudulent.*

---

## 4. Diagnostic Assessment & Root Cause Analysis

### 4.1 Why Does Recall Remain High (97.22%)?
The model successfully identifies fraud features:
- High transaction values relative to typical transfers.
- Transfer channels (`TRANSFER` and `CASH_OUT`).
- Rapid movement of funds.
Out of $1,008$ fraud attacks across diverse amounts and time steps, the model flagged **980 cases**, protecting **$98.02\%$ of all fraudulent capital** (\$82.04M out of \$83.70M).

### 4.2 Why Did Precision Collapse from 100% to 8.56%?
In the training data, legitimate users **never** transferred large portions of their balances, and transactions after step 7 were 100% fraud. When the model encountered unseen data where:
1. Legitimate users legitimately transfer funds in `TRANSFER` channels;
2. Legitimate transactions occur at steps $> 7$;
3. Legitimate users occasionally make large payments;

The model assigned fraud probabilities $> 0.50$ to almost all high-value transfers, resulting in $10,471$ false alarms.

### 4.3 Visual Evidence
Refer to the diagnostic chart:
- `unseen_data_audit/figures/01_original_vs_unseen_comparison.png`

This chart visually illustrates the bifurcation between benchmark perfection and unseen real-world false alarm escalation.

