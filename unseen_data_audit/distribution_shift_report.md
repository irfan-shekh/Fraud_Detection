# Phase 4: Distribution Shift Audit Report
**FraudGuard AI — Stress Testing Across 6 Real-World Operational Shift Scenarios**  
*Auditor: Senior Machine Learning Auditor & Production Risk Assessor*  
*Date: November 2026*  
*Project Repository: `c:\Projects\Fraud_Detection`*

---

## 1. Executive Summary

Real-world financial environments are non-stationary: fraud tactics evolve, payment channels expand, macroeconomic transaction volumes surge during holidays, and fraud prevalence is typically orders of magnitude lower than in synthetic academic benchmarks ($0.1\% - 0.5\%$ vs $10.25\%$).

This audit evaluated the frozen **Pre-Transaction RF** and **Post-Transaction XGB** models against six distinct distribution shift scenarios to measure degradation bounds, failure modes, and architectural resilience.

---

## 2. Overview of Tested Distribution Shift Scenarios

| Scenario ID | Name & Description | Shift Vector Tested | Sample Size ($N$) | Fraud Count | Prevalence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Shift 1** | **Normal Retail Shift** | Modest transaction sizes (\$10 - \$5,000), standard retail spend | Amount scale & channel drift | 10,000 | 783 | $7.83\%$ |
| **Shift 2** | **High-Amount B2B Shift** | Large corporate transfers (\$500,000 - \$5,000,000) | Extreme positive skewness | 5,000 | 400 | $8.00\%$ |
| **Shift 3** | **Off-Hours Velocity Shift** | Nighttime & early morning bursts (Hours 00:00 - 05:00) | Temporal distribution shift | 5,000 | 423 | $8.46\%$ |
| **Shift 4** | **Ultra-Low Fraud Prevalence** | Realistic production banking base rate | Severe class imbalance ($500:1$) | 18,447 | 40 | **0.22%** |
| **Shift 5** | **Ambiguous Overlapping Profiles** | Frauds mimicking legitimate balance ratios & amounts | Boundary ambiguity | 8,000 | 685 | $8.56\%$ |
| **Shift 6** | **Novel Channel Attacks** | Fraud occurring in `PAYMENT` and `DEBIT` channels | Unseen channel infiltration | 5,000 | 404 | $8.08\%$ |

---

## 3. Empirical Results Across Scenarios

### 3.1 Pre-Transaction Random Forest Model ($\tau = 0.21$)

| Scenario | Recall | Precision | F1-Score | PR-AUC | ROC-AUC | TP | FP | FN | Audit Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Shift 1: Retail** | $95.53\%$ | $7.86\%$ | $0.1453$ | $0.1457$ | $0.6728$ | 748 | 8,768 | 35 | Flags standard retail transfers as fraud |
| **Shift 2: High Amount** | $95.50\%$ | $7.99\%$ | $0.1475$ | $0.1481$ | $0.6446$ | 382 | 4,399 | 18 | Strong recall on large wires; heavy FP |
| **Shift 3: Off-Hours** | **71.63%** | $9.96\%$ | $0.1748$ | $0.2542$ | $0.6774$ | 303 | 2,740 | **120** | **Recall drops by 25.6%! Temporal bias failure** |
| **Shift 4: Low Prevalence** | **100.00%** | **0.23%** | $0.0045$ | $0.0084$ | $0.6453$ | 40 | **17,621** | **0** | Perfect recall, but 17,621 false alarms |
| **Shift 5: Ambiguous** | $95.77\%$ | $8.45\%$ | $0.1553$ | $0.0944$ | **0.5567** | 656 | 7,107 | 29 | ROC-AUC near random chance ($0.55$) |
| **Shift 6: Novel Channels**| $96.04\%$ | $8.08\%$ | $0.1491$ | $0.0696$ | **0.4391** | 388 | 4,412 | 16 | ROC-AUC degrades below random ($0.44$) |

---

### 3.2 Post-Transaction XGBoost Model ($\tau = 0.50$)

| Scenario | Recall | Precision | F1-Score | PR-AUC | ROC-AUC | TP | FP | FN | Audit Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Shift 1: Retail** | $95.53\%$ | $7.86\%$ | $0.1453$ | **0.2336** | **0.7481** | 748 | 8,768 | 35 | Outperforms RF on ranking metrics |
| **Shift 2: High Amount** | $95.50\%$ | $7.98\%$ | $0.1474$ | **0.2118** | **0.7870** | 382 | 4,402 | 18 | Superior ROC-AUC on high balance wires |
| **Shift 3: Off-Hours** | **71.87%** | $9.99\%$ | $0.1754$ | **0.2601** | **0.7277** | 304 | 2,740 | **119** | Temporal sensitivity present in both models |
| **Shift 4: Low Prevalence** | **100.00%** | **0.23%** | $0.0045$ | **0.0112** | **0.8168** | 40 | **17,621** | **0** | Catches 100% fraud; ROC-AUC is $0.8168$ |
| **Shift 5: Ambiguous** | $95.77\%$ | $8.45\%$ | $0.1553$ | $0.0742$ | **0.4489** | 656 | 7,107 | 29 | Partial drain confuses ledger errors |
| **Shift 6: Novel Channels**| **96.04%** | $8.08\%$ | $0.1491$ | **0.2215** | **0.7016** | 388 | 4,412 | 16 | **XGB handles novel channels far better!** |

---

## 4. Key Engineering Insights

### 4.1 The Off-Hours Vulnerability (Shift 3)
When transactions occurred between 00:00 and 05:00, **Recall dropped precipitously from $97.22\%$ to $71.63\%$**, allowing **120 out of 423 fraud attacks** to bypass detection. This confirms that the model heavily memorized the temporal features (`step`, `transaction_hour`) from the training set, where frauds were concentrated during specific operational simulation steps.

### 4.2 The Base-Rate Fallacy in Banking (Shift 4)
In Shift 4 ($0.22\%$ fraud rate), the model captured **$100\%$ of all fraud attacks** (40 out of 40), but produced **17,621 False Positives**. For every 1 actual fraud caught, human AML analysts would have to review **440 false alerts**. This mathematically illustrates why static thresholds optimized on balanced synthetic data cannot be deployed directly into production without recalibration against real banking base rates.

### 4.3 Why Post-Transaction XGBoost Handled Novel Channels Better (Shift 6)
In Shift 6 (fraud injected into `PAYMENT` and `DEBIT` channels), Pre-Tx RF had a ROC-AUC of $0.4391$ (worse than a coin flip) and PR-AUC of $0.0696$, because it relied heavily on `type_TRANSFER` and `type_CASH_OUT`. Conversely, Post-Tx XGB achieved **PR-AUC of $0.2215$ and ROC-AUC of $0.7016$**. The post-transaction ledger features (`balance_error_orig`, `orig_balance_depleted`) detect illicit fund siphoning **regardless of which channel was used**.

### 4.4 Visual Evidence
Refer to:
- `unseen_data_audit/figures/02_distribution_shift_performance.png`

