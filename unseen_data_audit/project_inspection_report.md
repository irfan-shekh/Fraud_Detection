# Phase 1: Complete Project Inspection Report
**FraudGuard AI — Independent Architectural & Pipeline Audit**  
*Auditor: Senior Machine Learning Engineer & Model Validation Specialist*  
*Date: November 2026*  
*Project Repository: `c:\Projects\Fraud_Detection`*

---

## 1. Executive Summary

This inspection report provides a comprehensive architectural baseline of the **FraudGuard AI** system prior to synthetic stress testing and unseen data evaluation. The system was constructed to solve a critical enterprise financial challenge: detecting fraudulent electronic fund transfers in high-throughput payment networks while maintaining sub-second latency and zero post-transaction data leakage.

The audit team performed an end-to-end static code and artifact inspection across data pipelines, feature engineering modules, model serializers, evaluation reports, and production dashboard interfaces.

---

## 2. Dataset Profile & Target Distribution

The project relies on a financial transaction dataset derived from the PaySim synthetic mobile money benchmark:

| Attribute | Audited Value | Description / Integrity Check |
| :--- | :--- | :--- |
| **Primary Dataset Path** | `data/raw/Fraud_Analysis_Dataset.csv` | Original benchmark data file |
| **Backup Mirror Path** | `Fraud_Analysis_Dataset.csv` | Repository root mirror |
| **Total Transaction Records ($N$)** | 11,142 | Complete row count |
| **Dataset File Size** | 1.34 MB | Comma-separated ASCII text |
| **Total Features (Raw Columns)** | 11 columns | `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud` |
| **Target Variable** | `isFraud` | Binary indicator ($1 = \text{Fraud}$, $0 = \text{Legitimate}$) |
| **Legitimate Transactions ($y=0$)** | 10,000 (89.75%) | Majority class |
| **Fraudulent Transactions ($y=1$)** | 1,142 (10.25%) | Minority positive class |
| **Class Imbalance Ratio** | 8.76 : 1 | Moderately imbalanced (unlike real-world $1000:1$, this synthetic slice has high fraud density) |
| **Secondary Flag Column** | `isFlaggedFraud` | Rule-based baseline: flags transactions $> 200,000$ (all values $0$ except 16 records) |

---

## 3. Dual-Model Architecture & Pipeline Separation

To resolve critical post-transaction data leakage, the codebase establishes two decoupled inference pipelines:

```
                      [ Incoming Transaction Request ]
                                     |
              +----------------------+----------------------+
              |                                             |
   [ Pipeline A: Pre-Transaction ]               [ Pipeline B: Post-Transaction ]
   - Execution: In-line Gateway (<10ms)          - Execution: Post-Settlement Ledger
   - Scope: Authorization / Hold / Decline       - Scope: SAR Filing / Clawback / Freeze
   - Features: Available BEFORE settlement       - Features: Ledger balances & delta errors
   - Model: Tuned Random Forest Classifier       - Model: Tuned XGBoost Classifier
   - Target Latency: < 15ms                      - Target Latency: Asynchronous / Batch
```

### 3.1 Pre-Transaction Model (`models/pre_transaction_rf.joblib`)
- **Algorithm:** Scikit-Learn `RandomForestClassifier` (100 estimators, balanced class weights, max depth 12).
- **Authorized Feature Space (16 Features):**
  1. `step` (simulation hour offset)
  2. `amount` (nominal transfer value)
  3. `oldbalanceOrg` (payer initial balance)
  4. `oldbalanceDest` (payee initial balance)
  5. `transaction_amount_log` ($\log(1 + \text{amount})$)
  6. `amount_to_oldbalance_orig_ratio` ($\text{amount} / (\text{oldbalanceOrg} + 1)$)
  7. `transaction_hour` (`step % 24`)
  8. `transaction_day` (`(step // 24) % 7`)
  9. `orig_has_zero_balance` ($\mathbb{I}(\text{oldbalanceOrg} == 0)$)
  10. `dest_has_zero_balance` ($\mathbb{I}(\text{oldbalanceDest} == 0)$)
  11. `empty_account_flag` ($\mathbb{I}(\text{oldbalanceOrg} == 0 \land \text{amount} > 0)$)
  12. `merchant_destination` ($\mathbb{I}(\text{nameDest starts with 'M'})$)
  13. `type_CASH_OUT` (One-hot encoded)
  14. `type_DEBIT` (One-hot encoded)
  15. `type_PAYMENT` (One-hot encoded)
  16. `type_TRANSFER` (One-hot encoded)

### 3.2 Post-Transaction Model (`models/post_transaction_xgb.joblib`)
- **Algorithm:** `XGBClassifier` (`n_estimators=150`, `max_depth=5`, `learning_rate=0.08`, `scale_pos_weight=8.76`).
- **Feature Space (22 Features):**
  - Includes all 16 pre-transaction features plus 6 post-settlement ledger audit signals:
    17. `newbalanceOrig` (payer balance post-settlement)
    18. `newbalanceDest` (payee balance post-settlement)
    19. `balance_error_orig` ($\text{newbalanceOrig} + \text{amount} - \text{oldbalanceOrg}$)
    20. `balance_error_dest` ($\text{oldbalanceDest} + \text{amount} - \text{newbalanceDest}$)
    21. `orig_balance_depleted` ($\mathbb{I}(\text{newbalanceOrig} == 0 \land \text{oldbalanceOrg} > 0)$)
    22. `dest_balance_unchanged` ($\mathbb{I}(\text{newbalanceDest} == \text{oldbalanceDest} \land \text{amount} > 0)$)

---

## 4. Train-Validation-Test Splitting Strategy

The repository employs a chronological split to prevent temporal data leakage:
- **Chronological Split Index:** Partitioned strictly on `step`.
  - **Training / Validation Partition ($80\%$):** Steps $1 \le \text{step} \le 72$ ($N_{\text{train}} = 8,913$ records; $914$ frauds, $7,999$ non-frauds).
  - **Out-of-Time (OOT) Test Partition ($20\%$):** Steps $73 \le \text{step} \le 95$ ($N_{\text{test}} = 2,229$ records; $228$ frauds, $2,001$ non-frauds).
- **Cross-Validation Strategy:** 5-fold TimeSeriesSplit / Stratified temporal cross-validation on the training set.
- **Threshold Selection Mechanism:** Thresholds are optimized strictly via out-of-fold (OOF) cross-validation predictions on the training set ($\tau_{\text{pre}} = 0.21$, $\tau_{\text{post}} = 0.50$). The OOT test set is held untouched for final evaluation.

---

## 5. Serialized Artifacts & Baseline Reported Metrics

| Model Artifact | Serialized File | Pipeline Preprocessor | Threshold $\tau$ | Benchmark Recall | Benchmark Precision | Benchmark F1 | Benchmark PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pre-Tx RF** | `pre_transaction_rf.joblib` | `pre_transaction_preprocessor.joblib` | $0.21$ | $99.56\%$ | $100.00\%$ | $0.9978$ | $0.9996$ |
| **Post-Tx XGB** | `post_transaction_xgb.joblib` | `post_transaction_preprocessor.joblib` | $0.50$ | $99.56\%$ | $99.56\%$ | $0.9956$ | $0.9970$ |

These metrics, while verified as reproducing accurately on the original holdout set (227 TP, 0 or 1 FP, 1 FN out of 228 frauds), are unusually high for real-world financial fraud systems. Phase 2 and subsequent phases investigate the root causes of this near-perfect performance.

