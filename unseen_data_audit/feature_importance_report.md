# Phase 7: Feature Importance & Explainability Report
**FraudGuard AI — Permutation Feature Importance & Error Attribution on Unseen Data**  
*Auditor: Senior Machine Learning Auditor & Interpretability Engineer*  
*Date: November 2026*  
*Project Repository: `c:\Projects\Fraud_Detection`*

---

## 1. Executive Summary

Standard Gini impurity importance (or MDI) in tree ensembles is notoriously biased toward high-cardinality features and can severely misrepresent true predictive power. To establish true model reliance on unseen data, we computed **Permutation Feature Importance** on the 12,000 independent unseen transactions using **PR-AUC degradation** across 5 independent shuffling iterations.

In addition, this report analyzes concrete transaction case studies to explain exactly why the model correctly flags fraud and why it generates false positives.

---

## 2. Permutation Feature Importance Ranking on Unseen Data

The table below lists all 16 pre-transaction features sorted by mean impact on PR-AUC when permuted on the 12,000 unseen transactions:

| Rank | Feature Name | Mean Importance ($\Delta$ PR-AUC) | Std Deviation | Role & Mechanism on Unseen Transactions |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `merchant_destination` | **+0.0534** | $0.0119$ | **Primary Negative Screen:** Merchant destinations (`M...`) are heavily associated with legitimate retail purchases. Permuting this destroys the model's ability to filter out legitimate retail volume. |
| **2** | `empty_account_flag` | **+0.0483** | $0.0074$ | **Account Depletion Signal:** Identifies accounts with near-zero starting balances attempting transactions. |
| **3** | `type_PAYMENT` | **+0.0415** | $0.0126$ | **Channel Exclusivity Filter:** In the training distribution, payment channels had $0\%$ fraud. Permuting it causes immediate false alarm spillover. |
| **4** | `type_CASH_OUT` | **+0.0111** | $0.0013$ | **Attack Funnel Signal:** Critical destination conduit for laundering stolen liquidity. |
| **5** | `transaction_amount_log` | **+0.0086** | $0.0018$ | **Magnitude Scale:** Captures non-linear scale of wire transfers. |
| **6** | `amount` | **+0.0084** | $0.0019$ | **Nominal Exposure:** Corroborates high-risk transaction sizes. |
| **7** | `step` | **+0.0037** | $0.0048$ | **Simulation Clock:** Carries residual temporal bias from the PaySim generator. |
| **8** | `transaction_hour` | **+0.0028** | $0.0081$ | **Diurnal Clock:** Minor time-of-day weighting. |
| **9** | `dest_has_zero_balance` | **+0.0018** | $0.0014$ | **Mule Account Indicator:** Captures newly opened, zero-balance destination accounts. |
| **10** | `type_DEBIT` | **0.0000** | $0.0000$ | **Low Volume:** Minimal transaction frequency in test sets. |
| **11** | `orig_has_zero_balance` | **0.0000** | $0.0000$ | **Redundant:** Subsumed by `empty_account_flag`. |
| **12** | `oldbalanceOrg` | **-0.0007** | $0.0016$ | **Neutral:** Permutation has near-zero net effect on PR-AUC. |
| **13** | `transaction_day` | **-0.0021** | $0.0015$ | **Weak Noise:** Shuffling slightly smooths decision boundaries. |
| **14** | `oldbalanceDest` | **-0.0030** | $0.0046$ | **Minor Overfitting:** Dest balance was noisy across bank payees. |
| **15** | `type_TRANSFER` | **-0.0118** | $0.0032$ | **Negative Permutation Effect:** Shuffling transfer indicator actually reduces false alarms slightly. |
| **16** | `amount_to_oldbalance_orig_ratio` | **-0.0437** | $0.0109$ | **CRITICAL AUDIT FINDING: Misleading Synthetic Shortcut.** In synthetic data, this ratio was exactly $1.0$ for $97.1\%$ of frauds. On unseen data with partial drains and legitimate account closures, this feature induces false positives; shuffling it actually improves PR-AUC! |

---

## 3. Case Studies: Prediction Explanations

### Case Study 1: Correctly Detected Fraud (True Positive)
- **Transaction Details:** `TRANSFER`, Amount: \$412,850.00, Payer Old Balance: \$450,000.00, Payee Old Balance: \$0.00.
- **Model Output:** Fraud Probability: **0.96** (Predicted Fraud; Actual Fraud).
- **Explanation:**
  - `merchant_destination = 0` (Person-to-person transfer).
  - High nominal amount (\$412k) and high log-amount ($12.93$).
  - Payee account was previously un-funded (`dest_has_zero_balance = 1`).
  - High drain ratio ($91.7\%$).
  - **Verdict:** Genuine high-risk fraud pattern accurately captured.

### Case Study 2: Legitimate Customer Flagged as Fraud (False Positive)
- **Transaction Details:** `TRANSFER`, Amount: \$35,000.00, Payer Old Balance: \$35,000.00, Payee Old Balance: \$12,000.00.
- **Customer Context:** Customer closing their checking account and moving funds to an external account.
- **Model Output:** Fraud Probability: **0.82** (Predicted Fraud; Actual Legitimate).
- **Explanation:**
  - `amount_to_oldbalance_orig_ratio = 1.0000` (Exact balance drainage).
  - `type_TRANSFER = 1`.
  - Step $= 48$ (In training data, legitimate transfers past step 7 were virtually non-existent).
  - **Verdict:** Model exploited the synthetic assumption that legitimate customers never transfer $100\%$ of their balance.

### Case Study 3: Subtle Fraud Evading Detection (False Negative)
- **Transaction Details:** `TRANSFER`, Amount: \$8,500.00, Payer Old Balance: \$95,000.00, Payee Old Balance: \$4,200.00.
- **Fraud Context:** Fraudster executing a low-value test transaction ($8.9\%$ of balance).
- **Model Output:** Fraud Probability: **0.14** (Predicted Legitimate; Actual Fraud).
- **Explanation:**
  - Ratio is low ($0.089$).
  - Amount is below typical wire attack sizes.
  - Model failed to trigger because it learned to expect massive balance drain ratios for fraud.

---

## 4. Visual Evidence
Refer to:
- `unseen_data_audit/figures/05_permutation_feature_importance.png`

