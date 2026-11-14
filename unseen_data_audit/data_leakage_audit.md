# Phase 2: Data Leakage & Synthetic Artifact Audit Report
**FraudGuard AI — Deep Mechanistic Investigation of Near-Perfect Benchmark Metrics**  
*Auditor: Senior Machine Learning Auditor & Forensic Data Scientist*  
*Date: November 2026*  
*Project Repository: `c:\Projects\Fraud_Detection`*

---

## 1. Executive Summary

The primary objective of Phase 2 is to answer the central investigative question:  
*Why do the baseline machine learning models achieve nearly 100% Precision, 99.56% Recall, and 0.9996 PR-AUC on the benchmark dataset?*

Through automated forensic auditing and mathematical cross-examination, this audit confirms that the near-perfect metrics are **not** caused by classic data contamination (such as shared row hashes across splits or feature lookahead bugs), but rather by **severe synthetic data generation artifacts**. Specifically, the synthetic generation engine embedded deterministic mathematical signatures into the fraud records that do not exist in legitimate transactions.

---

## 2. Duplicate Record & Identity Collision Audit

The entire 11,142-transaction dataset was audited for identity collisions, row duplication, and cross-split leakage:

| Forensic Audit Test | Method / Criteria | Result | Audit Assessment |
| :--- | :--- | :--- | :--- |
| **Exact Duplicate Rows** | Full row hash matching across all 11 columns | **0 records (0.00%)** | Pass: No identical duplicate transactions exist. |
| **Feature Duplicates (Excl. IDs)** | Exact match across all numerical & categorical values | **0 records (0.00%)** | Pass: Every transaction has a unique vector. |
| **Cross-Split Duplicates** | Matching training feature vectors against test vectors | **0 records (0.00%)** | Pass: Zero partition overlap between Train and Test. |
| **Sender Account Collisions** | Unique `nameOrig` count | **11,142 / 11,142** | Pass: Each sender account appears exactly once. |
| **Recipient Account Re-use** | Unique `nameDest` count | **7,508 / 11,142** | Normal: Payees receive repeat transactions. |

---

## 3. The Synthetic Liquidation Determinism Phenomenon

The single most impactful finding of this audit is the discovery of **Synthetic Liquidation Determinism**:

$$\text{Liquidation Condition}: \quad \text{amount} = \text{oldbalanceOrg}$$

In genuine financial payment systems, account liquidation ($\text{amount} = \text{balance}$) occurs legitimately when customers close bank accounts, transfer funds between personal checking and savings, or wire final escrow balances. Conversely, real-world fraudsters frequently execute micro-charges, testing transactions, or fractional balance transfers ($20\% - 75\%$) to avoid triggering AML (Anti-Money Laundering) thresholds.

However, in this synthetic benchmark:

```
+-----------------------------------------------------------------------------+
| Synthetic Benchmark Liquidation Statistics:                                 |
| - Total Fraud Transactions: 1,142                                           |
|   * Transactions where amount == oldbalanceOrg: 1,109 (97.11%)              |
|                                                                             |
| - Total Legitimate Transactions: 10,000                                     |
|   * Transactions where amount == oldbalanceOrg: 0 (0.00%)                   |
+-----------------------------------------------------------------------------+
```

### Forensic Implication:
For $97.11\%$ of all fraudulent transactions in the dataset, the ratio $\text{amount} / (\text{oldbalanceOrg} + 1) \approx 1.0000$. Because **zero** legitimate transactions exhibit this condition in the dataset, any decision tree split on `amount_to_oldbalance_orig_ratio == 1.0` or `balance_error_orig == 0` instantly achieves near $100\%$ precision without learning subtle fraud behavior.

---

## 4. Temporal Step Segregation Artifact

A secondary synthetic artifact was discovered in the temporal distribution of transaction steps:

```
[ Step 1 through Step 7 ] : Contains ALL 10,000 Legitimate Transactions + 78 Frauds
[ Step 8 through Step 95] : Contains 1,064 Transactions — 100.00% FRAUD (0 Legitimate)
```

| Partition | Step Range | Total Records | Legitimate Records | Fraud Records | Fraud Prevalence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Early Steps** | Steps $1 - 7$ | 10,078 | 10,000 | 78 | $0.77\%$ |
| **Late Steps** | Steps $8 - 95$ | 1,064 | **0** | **1,064** | **100.00%** |
| **Chronological Train Split** | Steps $1 - 72$ | 8,913 | 7,999 | 914 | $10.25\%$ |
| **Chronological Test Split** | Steps $73 - 95$ | 2,229 | 2,001 | 228 | $10.23\%$ |

Because the chronological split divided steps at $72 / 73$, the training set included steps $1$ to $7$ (where all legit records lived) and steps $8$ to $72$ (which contained pure fraud). As a result, the trees learned that high step numbers have an extremely high prior probability of fraud.

---

## 5. Channel Exclusivity

In the raw dataset, transaction types are strictly partitioned:
- **Fraudulent Transactions:** $100.00\%$ occur exclusively in `TRANSFER` ($575$ cases) and `CASH_OUT` ($567$ cases).
- **Legitimate Transactions:** Spread across `PAYMENT` ($4,484$), `CASH_OUT` ($2,510$), `CASH_IN` ($1,843$), `TRANSFER` ($1,093$), and `DEBIT` ($70$).
- **Zero Fraud in `PAYMENT`, `CASH_IN`, and `DEBIT`:** The synthetic generator created exactly zero fraud records in retail payment channels. Consequently, any transaction with `merchant_destination == 1` or `type_PAYMENT == 1` is deterministically classified as legitimate by the model.

---

## 6. Summary of Audit Findings

| Audit Dimension | Finding | Impact on Benchmark Metrics |
| :--- | :--- | :--- |
| **Pipeline Leakage** | Strictly absent | Feature calculations use only point-in-time authorized data. |
| **Duplicate Leakage** | Strictly absent | No train/test overlap or duplicated row vectors. |
| **Deterministic Artifact #1** | $97.11\%$ of frauds drain $100\%$ of account balance | Creates an artificial mathematical cheat code for decision trees. |
| **Deterministic Artifact #2** | Legitimate data generation ceases after Step 7 | Induces unnatural temporal correlation. |
| **Deterministic Artifact #3** | Zero fraud in payment/debit channels | Allows trivial elimination of $45\%$ of false positives. |

**Audit Conclusion:** The model's $99.56\%$ Recall and $100\%$ Precision on the original holdout are mathematically valid for the dataset distribution, but are **heavily inflated by synthetic shortcuts**. When exposed to real-world transactions where legitimate users close accounts and fraudsters drain partial balances, the model's precision will degrade significantly unless audited against independent, un-shortcut data.

