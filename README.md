# 🛡️ FRAUDGUARD AI: Intelligent Financial Fraud Detection System
### Boston Institute of Analytics (BIA) — Data Science Capstone Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717.svg?logo=github)](https://github.com/irfan-shekh/Fraud_Detection)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frauddetection-17.streamlit.app)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pytest Passed](https://img.shields.io/badge/pytest-25%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade Machine Learning intelligence platform developed for the **Boston Institute of Analytics (BIA)** Data Science Capstone. The system features a **Dual-Architecture Engine** that rigorously distinguishes between **Pre-Transaction Fraud Screening** (real-time pre-authorization switch with zero leakage) and **Post-Transaction Fraud Detection** (post-clearing ledger reconciliation & audit), evaluates enterprise monetary exposure, provides model explainability through SHAP, and delivers an interactive 7-page analytics dashboard via Streamlit.

🔗 **Live Production Application:** [https://frauddetection-17.streamlit.app](https://frauddetection-17.streamlit.app)  
📁 **GitHub Repository:** [https://github.com/irfan-shekh/Fraud_Detection](https://github.com/irfan-shekh/Fraud_Detection)

---

## 📌 Executive Summary & Key Results

| Metric | Pre-Transaction Screening Model | Post-Transaction Detection Model | Context / Business Impact |
|---|---|---|---|
| **Operational Timing** | **Pre-Authorization (Sub-50ms Gateway)** | **Post-Settlement (Audit & Dispute)** | Separation eliminates target & ledger leakage |
| **Champion Algorithm** | **Random Forest Classifier** | **XGBoost Classifier** | Selected via leak-free 5-Fold Stratified CV |
| **Input Dimensions** | **16 Encoded Features** (13 unencoded) | **26 Encoded Features** (23 unencoded) | Exact dimensions audited via metadata |
| **Calibrated Threshold** | **0.21** (Optimized on Train OOF) | **0.07** (Optimized on Train OOF) | Threshold chosen via business objective |
| **Test Accuracy** | **99.96%** (2,228 / 2,229 correct) | **99.91%** (2,227 / 2,229 correct) | Untouched test partition (2,229 records) |
| **Test Precision** | **100.00%** (0 False Positives) | **99.56%** (1 False Positive) | 100.00% precision at default 0.50 threshold |
| **Test Recall** | **99.56%** (227 / 228 Detected) | **99.56%** (227 / 228 Detected) | Intercepts 99.82% of all fraudulent capital |
| **Test F1-Score** | **0.9978** | **0.9956** | Harmonic mean of Precision and Recall |
| **Test ROC-AUC** | **0.99996** | **0.99899** | Near-perfect class separability |
| **Test PR-AUC** | **0.99961** | **0.99699** | Benchmark metric for imbalanced fraud data |
| **False Positives** | **0** | **1** (0 at th=0.50) | Near-zero customer friction on legitimate accounts |
| **False Negatives** | **1** | **1** | Exactly 1 fraud missed ($416,001.33 exposure) |
| **Gross Intercepted Value**| **$235,430,525.04** | **$235,430,525.04** | Out of $235,846,526.37 total test fraud |
| **Net Financial Benefit** | **$188,337,610.03** | **$188,337,610.03** | Net savings under 80% recovery SLA |

---

## 🏛️ System Architecture: Dual Pre- and Post-Transaction Models

A critical finding in production payment systems is that features derived from final ledger updates (such as `newbalanceOrig`, `newbalanceDest`, and derived ledger errors) are **only available after transaction settlement**. Training a real-time pre-authorization model with post-clearing fields introduces catastrophic target leakage.

FraudGuard AI addresses this with two independent, dedicated pipelines:

```
                                 TRANSACTION FLOW
                                        │
                         [Initiated by Customer/API]
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │    1. PRE-TRANSACTION FRAUD SCREENING       │
                 │          (Real-Time In-Flight Gate)          │
                 ├──────────────────────────────────────────────┤
                 │ • Features: step, type, amount,              │
                 │   oldbalanceOrg, oldbalanceDest, hour, day,  │
                 │   log(amount), amount/balance ratio,         │
                 │   merchant flag, zero starting balance,      │
                 │   empty account liquidation intent flag.     │
                 │ • Excludes: newbalanceOrig, newbalanceDest,  │
                 │   balance_error_*, origin_balance_zero.      │
                 │ • Model: Random Forest (16 encoded features) │
                 │ • Threshold: 0.21 (Train OOF Calibrated)     │
                 │ • Decision: Approve, Step-Up 2FA, or Block   │
                 └──────────────────────┬───────────────────────┘
                                        │
                               [Transaction Executes]
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │     2. POST-TRANSACTION FRAUD DETECTION      │
                 │          (Clearing & Audit Ledger)           │
                 ├──────────────────────────────────────────────┤
                 │ • Features: All pre-transaction features +   │
                 │   newbalanceOrig, newbalanceDest,            │
                 │   origin_balance_difference,                 │
                 │   destination_balance_difference,            │
                 │   balance_error_orig, balance_error_dest,    │
                 │   origin_balance_zero, dest_balance_zero,    │
                 │   error_orig_zero, error_dest_zero.          │
                 │ • Model: XGBoost (26 encoded features)       │
                 │ • Threshold: 0.07 (Train OOF Calibrated)     │
                 │ • Action: Dispute Resolution & AML Filing    │
                 └──────────────────────────────────────────────┘
```

---

## 📐 Feature Engineering & Data Dictionary

Raw dataset contains **10 attributes**: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`.

### 1. Pre-Transaction Feature Pipeline (16 Encoded Dimensions)
Used strictly for pre-authorization screening. No post-settlement fields permitted:
1. `step` (Raw): Temporal hour index since inception.
2. `amount` (Raw): Requested transaction volume in USD.
3. `oldbalanceOrg` (Raw): Sender balance immediately prior to execution.
4. `oldbalanceDest` (Raw): Receiver balance immediately prior to execution.
5. `transaction_hour = step % 24`: Diurnal hour of execution (0–23).
6. `transaction_day = ((step - 1) // 24) + 1`: Operational calendar day (Day 1, 2, ...).
7. `transaction_amount_log = log1p(amount)`: Compresses monetary skewness.
8. `merchant_destination = 1 if nameDest starts with 'M' else 0`: Identifies retail POS payments.
9. `amount_to_oldbalance_orig_ratio = amount / oldbalanceOrg`: Liquidation intensity (clipped to [0, 100]).
10. `orig_has_zero_balance = 1 if oldbalanceOrg == 0 else 0`: Identifies zero-funded source accounts.
11. `dest_has_zero_balance = 1 if oldbalanceDest == 0 else 0`: Identifies unactivated receiver accounts.
12. `empty_account_flag = 1 if amount == oldbalanceOrg and oldbalanceOrg > 0 else 0`: Exact 100% liquidation intent.
13. `type` One-Hot Encoded (4 columns: `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`; `CASH_IN` dropped as reference).

### 2. Post-Transaction Feature Pipeline (26 Encoded Dimensions)
Combines all 16 pre-transaction dimensions plus 10 post-clearing ledger discrepancy indicators:
14. `newbalanceOrig` (Raw): Sender ledger balance after settlement.
15. `newbalanceDest` (Raw): Receiver ledger balance after settlement.
16. `origin_balance_difference = oldbalanceOrg - newbalanceOrig`: Net debit from sender account.
17. `destination_balance_difference = newbalanceDest - oldbalanceDest`: Net credit to receiver account.
18. `balance_error_orig = oldbalanceOrg - amount - newbalanceOrig`: Mathematical anomaly in sender ledger.
19. `balance_error_dest = oldbalanceDest + amount - newbalanceDest`: Mathematical anomaly in receiver ledger.
20. `origin_balance_zero = 1 if newbalanceOrig == 0 else 0`: Sender account verified empty.
21. `destination_balance_zero = 1 if newbalanceDest == 0 else 0`: Receiver account verified empty.
22. `error_orig_zero = 1 if abs(balance_error_orig) < 0.01 else 0`: Sender accounting conforms exactly.
23. `error_dest_zero = 1 if abs(balance_error_dest) < 0.01 else 0`: Receiver accounting conforms exactly.

---

## 🔬 Leak-Free Cross-Validation & Benchmark Comparison

To ensure scientific validity and prevent cross-validation leakage:
1. Preprocessing (`ColumnTransformer`) is bundled directly with the estimator inside a `Pipeline` or `cross_val_predict` workflow.
2. The preprocessor is fitted **strictly inside each training fold** across 5-fold Stratified Cross-Validation.
3. Decision thresholds were calibrated strictly using **out-of-fold training predictions** (`cv=5`).
4. The test set (2,229 transactions; 228 frauds) remained completely untouched until final evaluation.

### Pre-Transaction Model Benchmark (Evaluated on Untouched Test Set)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Optimal Threshold |
|---|---|---|---|---|---|---|---|
| **Random Forest (Champion)** | **0.9996** | **1.0000** | **0.9956** | **0.9978** | **0.99996** | **0.99961** | **0.21** |
| **XGBoost Classifier** | 0.9996 | 1.0000 | 0.9956 | 0.9978 | 0.99971 | 0.99784 | 0.35 |
| **Decision Tree** | 0.9991 | 0.9956 | 0.9956 | 0.9956 | 0.99753 | 0.99127 | 0.50 |
| **Logistic Regression** | 0.9955 | 0.9822 | 0.9737 | 0.9779 | 0.99512 | 0.97230 | 0.40 |

### Post-Transaction Model Benchmark (Evaluated on Untouched Test Set)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Optimal Threshold |
|---|---|---|---|---|---|---|---|
| **XGBoost (Champion)** | **0.9996** | **1.0000** | **0.9956** | **0.9978** | **0.99999** | **0.99994** | **0.07** |
| **Random Forest** | 0.9996 | 1.0000 | 0.9956 | 0.9978 | 0.99998 | 0.99991 | 0.21 |
| **Decision Tree** | 0.9996 | 1.0000 | 0.9956 | 0.9978 | 0.99781 | 0.99606 | 0.50 |
| **Logistic Regression** | 0.9987 | 0.9956 | 0.9912 | 0.9934 | 0.99634 | 0.99610 | 0.45 |

---

## ⏱️ Temporal Generalization & Walk-Forward Validation Experiment

To verify model stability and measure resilience against temporal drift, a multi-tier temporal validation suite was executed across the 95-hour operational horizon ([`src/temporal_experiment.py`](file:///c:/Projects/Fraud_Detection/src/temporal_experiment.py)).

### 1. Macro Out-of-Time (OOT) Holdout (Steps 1–6 vs. 7–95)
- **Training Horizon:** Steps 1 to 6 (7,164 historical transactions; 66 frauds, 7,098 legitimate).
- **Testing Horizon:** Steps 7 to 95 (3,978 future transactions; 1,076 frauds, 2,902 legitimate).
- **Temporal Constraint:** `train_step.max() == 6 < test_step.min() == 7` (Zero overlap).

| Model Architecture | OOT Recall | OOT Precision | OOT F1-Score | OOT PR-AUC | Intercepted Value ($) | Total Fraud Exposure ($) | Capital Capture Rate |
|---|---|---|---|---|---|---|---|
| **Pre-Transaction (Random Forest)** | **97.77%** | **100.00%** | **0.9887** | **0.9898** | **$1,313,912,230.14** | $1,323,605,690.72 | **99.27%** |
| **Post-Transaction (XGBoost)** | **97.77%** | **99.91%** | **0.9882** | **0.9956** | **$1,313,912,230.14** | $1,323,605,690.72 | **99.27%** |

*Key Result:* When trained strictly on past data (Steps 1–6), the pre-transaction model intercepted **99.27% of all fraudulent capital** over the subsequent 89 hours with **zero false alarms** on 2,902 unseen legitimate transactions.

### 2. Walk-Forward Expanding Window Validation (Steps 3 to 7)
In-flight performance was evaluated sequentially across expanding historical windows during active mixed-traffic hours:

| Window Sequence | Training Horizon | Evaluation Step | Test Size | Test Frauds | Recall | Precision | F1-Score | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| **Window 1** | Steps 1–2 | Step 3 | 552 | 4 | **100.0%** | **100.0%** | 1.0000 | 1.0000 |
| **Window 2** | Steps 1–3 | Step 4 | 565 | 10 | **90.0%** | **100.0%** | 0.9474 | 0.9833 |
| **Window 3** | Steps 1–4 | Step 5 | 665 | 6 | **100.0%** | **100.0%** | 1.0000 | 1.0000 |
| **Window 4** | Steps 1–5 | Step 6 | 1,660 | 22 | **100.0%** | **100.0%** | 1.0000 | 1.0000 |
| **Window 5** | Steps 1–6 | Step 7 | 2,914 | 12 | **100.0%** | **100.0%** | 1.0000 | 1.0000 |

### 3. Late-Stage Pure Fraud Stream Holdout (Steps 8 to 95)
- Evaluated on 1,064 pure fraud transactions occurring after step 7.
- **Frauds Intercepted:** 1,040 / 1,064 (**97.74% Recall**).
- **Monetary Interception:** **$1,304,123,427.44** of $1,313,816,888.02 (**99.26% Capital Capture Rate**).
- Full diagnostic report: [`reports/temporal_generalization_report.json`](file:///c:/Projects/Fraud_Detection/reports/temporal_generalization_report.json) and plot [`reports/figures/28_temporal_generalization_analysis.png`](file:///c:/Projects/Fraud_Detection/reports/figures/28_temporal_generalization_analysis.png).

### 4. Temporal Feature Ablation Experiment
To determine whether high Out-Of-Time generalization depends on temporal features, an ablation study was conducted across three controlled models on the identical OOT test partition (Steps 7–95: 1,076 frauds, $1.326B exposure):

| Configuration | Features Used | Threshold (OOF) | OOT Precision | OOT Recall | OOT F1 | OOT PR-AUC | Frauds Caught | Capital Capture |
|---|---|---|---|---|---|---|---|---|
| **Exp A: Full Pre-Tx** | 13 (incl. step, hour, day, ratio) | 0.15 | **100.00%** | 97.77% | 0.9887 | 0.9898 | 1,052 / 1,076 | 99.27% |
| **Exp B: No Temporal** | 10 (drop step, hour, day) | 0.21 | 99.91% | **98.14%** | **0.9902** | **0.9929** | **1,056 / 1,076** | **99.40%** |
| **Exp C: No Temporal / No Ratio** | 9 (drop step, hour, day, ratio) | 0.26 | 99.91% | **98.14%** | **0.9902** | 0.9916 | **1,056 / 1,076** | **99.40%** |

*Key Conclusion:* Removing temporal features actually **improves PR-AUC from 0.9898 to 0.9929** and intercepts 4 additional frauds. This proves that FraudGuard AI's high generalization is not propped up by dataset-specific temporal shortcuts, but is anchored in invariant behavioral dynamics (`empty_account_flag`, channel specificity, and monetary volume). Full report: [`reports/TEMPORAL_FEATURE_ABLATION.md`](file:///c:/Projects/Fraud_Detection/reports/TEMPORAL_FEATURE_ABLATION.md).

---

## 🧠 Explainable AI (SHAP)

To satisfy financial regulatory compliance (FCRA, GDPR Article 22, SR 11-7), SHAP TreeExplainers were integrated for both pipelines:
- **Pre-Transaction Drivers:** In pre-authorization mode, `empty_account_flag` (intent to drain 100% of available funds), `amount_to_oldbalance_orig_ratio`, `amount`, and `oldbalanceOrg` provide the strongest predictive signals.
- **Post-Transaction Drivers:** In post-clearing mode, `balance_error_orig` (accounting discrepancy between stated balance, amount, and final ledger) and `origin_balance_zero` become the dominant confirmation drivers.
- **Rule-Based Heuristic Alerts:** Distinguishes between SHAP statistical feature contributions and explicit policy alerts (e.g., High-Risk Channel, Large Monetary Volume).

---

## 💰 Financial Risk & Scenario Modeling

Financial exposure analysis was computed across 2,229 test transactions ($235,846,526.37 total fraudulent capital):

| Enterprise Scenario | Fraud Recovery Rate | Investigation Cost / Alert | FP Friction Rate | Gross Prevented Loss ($) | Operational Cost ($) | Net Financial Benefit ($) |
|---|---|---|---|---|---|---|
| **High Efficiency (Optimistic)** | 90% | $15.00 | 0.5% | $211,887,472.54 | $3,405.00 | **$211,884,067.54** |
| **Base Case (Realistic)** | **80%** | **$30.00** | **1.5%** | **$188,344,420.03** | **$6,810.00** | **$188,337,610.03** |
| **Conservative (Pessimistic)** | 70% | $60.00 | 3.0% | $164,801,367.53 | $13,620.00 | **$164,787,747.53** |

---

## 🖥️ Streamlit Production Web Application

The live application features 7 specialized navigation pages:
1. **📊 Dashboard Overview:** Dataset KPIs, class balance distribution, channel risk breakdown, and hourly transaction velocity.
2. **⚡ Pre-Transaction Fraud Screening:** Real-time pre-authorization screening engine requiring only pre-execution fields. Provides risk gauges, dynamic policy alerts, and automated decision actions (Approve, Step-Up 2FA, Block).
3. **🔍 Post-Transaction Fraud Detection:** Post-clearing forensic audit engine evaluating ledger discrepancies and settlement mismatches.
4. **📂 Batch Transaction Prediction:** Multi-record batch screening engine supporting both Pre-Transaction and Post-Transaction CSV templates with automated schema validation.
5. **📈 Model Performance Comparison:** Side-by-side benchmark comparison between Pre- and Post-Transaction models with interactive confusion matrices, ROC/PR curves, and threshold simulators.
6. **🧠 Explainable AI (SHAP):** Live TreeExplainer visualizations including summary beeswarm plots, global importance bars, and individual waterfall plots.
7. **💰 Financial Impact Analysis:** Interactive enterprise ROI simulator allowing risk teams to adjust recovery rates, operational review costs, and customer friction multipliers.

---

## 📊 Power BI Executive Analytics Dashboard

In addition to the live Streamlit app, FraudGuard AI includes a **5-page interactive Power BI Desktop report** ([`reports/Fraud_Detection.pbix`](file:///c:/Projects/Fraud_Detection/reports/Fraud_Detection.pbix)) powered by the enriched dataset ([`reports/fraud_detection_bi_dataset.csv`](file:///c:/Projects/Fraud_Detection/reports/fraud_detection_bi_dataset.csv)) and custom DAX measures, documented in [`reports/FraudGuard_AI_PowerBI_Dashboard_Summary.md`](file:///c:/Projects/Fraud_Detection/reports/FraudGuard_AI_PowerBI_Dashboard_Summary.md):

1. **Executive Overview:** Top-line KPI cards (11,142 Total Transactions, 1,142 Frauds, 10.25% Fraud Rate, ~$1.36B Fraud Exposure), transaction breakdowns by type, and fraud activity by simulation step.
2. **Transaction Analysis:** Interactive filtering slicers (`type`, `Risk_Tier`, `Prediction_Status`), transaction value histograms, and channel-wise average amounts.
3. **AI Model Performance:** Held-out test metrics comparison between the Pre-Transaction Random Forest (100.00% Precision, 99.56% Recall, 99.96% PR-AUC) and Post-Transaction XGBoost (99.56% Precision, 99.56% Recall, 99.70% PR-AUC), alongside confusion matrix KPIs.
4. **Risk Intelligence:** Distribution across model-assigned Risk Tiers (`Critical Risk`: 1,139; `High Risk`: 2; `Minimal Risk`: 10,001), probability histogram, and Actual vs. Predicted counts (`1,142` vs `1,141`).
5. **Temporal & Financial Analysis:** Simulation step trend lines (`TRANSFER`: ~$681.6M; `CASH_OUT`: ~$680.38M) with explicit synthetic data and step-7 temporal distribution disclosures.

---

## 🔬 Independent Unseen Data Generalization & Reliability Audit

An independent stress-testing audit was executed to answer a fundamental production question:  
*Does the model genuinely learn fraud patterns and generalize to unseen transactions, or is the high performance (99.56%–100%) driven by synthetic dataset artifacts, data leakage, or overfitting?*

The complete investigation is documented in 8 standalone reports and empirical datasets located in [`unseen_data_audit/`](unseen_data_audit/).

### 1. Benchmark Holdout (Test A) vs. Independent Unseen Data (Test B)

An independent evaluation dataset of **12,000 unseen transactions** (1,008 fraud attacks = 8.40% prevalence; seed `2026`) was generated relaxing synthetic shortcuts: legitimate accounts occasionally execute full balance transfers (e.g. closing accounts), fraudsters execute partial balance drains (20%–75%), and legitimate activity spans all 150 simulation steps.

| Evaluation Metric | Pre-Tx RF (Test A: Holdout) | Pre-Tx RF (Test B: 12k Unseen) | Post-Tx XGB (Test A: Holdout) | Post-Tx XGB (Test B: 12k Unseen) | Production Assessment |
|---|---|---|---|---|---|
| **Sample Size ($N$)** | 2,229 | 12,000 | 2,229 | 12,000 | +438% sample volume |
| **Fraud Attacks Caught** | **227 / 228 (99.56%)** | **980 / 1,008 (97.22%)** | **227 / 228 (99.56%)** | **980 / 1,008 (97.22%)** | **Catches 980 of 1,008 frauds; highly robust sensitivity** |
| **Precision** | **100.00%** (0 FP) | **8.56%** (10,471 FP) | **99.56%** (1 FP) | **8.56%** (10,471 FP) | Severe false alarm spike when synthetic shortcuts relax |
| **F1-Score** | 0.9978 | 0.1573 | 0.9956 | 0.1573 | Precision drag reduces harmonic mean |
| **PR-AUC** | 0.9996 | 0.2207 | 0.9970 | **0.2652** | XGBoost exhibits superior ranking precision |
| **ROC-AUC** | 1.0000 | 0.7128 | 0.9990 | **0.7973** | Post-Tx XGB preserves strong global separation |
| **Capital Intercepted ($)** | $235,430,525.04 | $82,041,261.54 | $235,430,525.04 | $82,041,261.54 | Full dollar protection on caught frauds |
| **Capital Capture Rate** | **99.82%** | **98.02%** | **99.82%** | **98.02%** | **$82.04M out of $83.70M fraud capital protected** |

### 2. Forensic Discovery: Synthetic Liquidation Determinism
The investigation discovered the exact mechanism explaining why benchmark models achieve near-perfect metrics:
- **Synthetic Liquidation Determinism:** In the benchmark training data, **97.11% (1,109 / 1,142) of all frauds** have `amount == oldbalanceOrg` (complete balance drainage), whereas **0.00% (0 / 10,000)** of legitimate transactions drain an account.
- **Temporal Step Segregation:** Legitimate transactions appear **only in steps 1 through 7**. Steps 8 through 95 contain 1,064 transactions, and **100% of them are fraud** (0 legitimate records).
- **Channel Segregation:** 100% of frauds occurred strictly in `TRANSFER` and `CASH_OUT`. Zero in `PAYMENT`, `CASH_IN`, and `DEBIT`.

When tested on unconstrained data where legitimate users close accounts or make large transfers, the model over-flags high-value transfers, resulting in $10,471$ false alarms.

### 3. Distribution Shifts & Noise Stress Testing
- **Off-Hours Shift (Hours 00:00–05:00):** Recall dropped from $97.22\%$ to **$71.63\%$ (120 frauds missed)**, exposing vulnerability to temporal features (`step`, `transaction_hour`).
- **Low Prevalence (0.22% Fraud Base Rate):** Model achieved **100.00% Recall** (40 / 40 caught), but produced 17,621 False Positives, illustrating the **Base-Rate Fallacy in Banking** (440 false alerts per true fraud).
- **Novel Channels (Fraud in Payment/Debit):** Pre-Tx RF ROC-AUC collapsed to $0.4391$, while **Post-Tx XGBoost maintained ROC-AUC of 0.7016 and PR-AUC of 0.2215**, proving post-settlement ledger physics (`balance_error_orig`) detect fraud regardless of channel.
- **Continuous Noise Perturbations (5% to 20% Gaussian Jitter):** Recall remained rock-solid at **96.23%** across all noise tiers, proving the model relies on structural/categorical indicators rather than delicate numerical precision.

### 4. Permutation Importance & Shortcut Exposure on Unseen Data
Permutation feature importance on the 12,000 unseen transactions revealed:
1. `merchant_destination` (+0.0534 $\Delta$ PR-AUC): Primary legitimate transaction filter.
2. `empty_account_flag` (+0.0483 $\Delta$ PR-AUC): Core fraud liquidation signal.
3. `type_PAYMENT` (+0.0415 $\Delta$ PR-AUC): Retail channel filter.
4. `amount_to_oldbalance_orig_ratio` (**-0.0437 $\Delta$ PR-AUC**): **Negative importance.** Permuting this ratio actually *improved* performance on unseen data because it was a synthetic shortcut that caused false alarms on legitimate account closures.

### 5. Final Auditor Verdict & Reliability Rating
- **Production Status:** **Conditionally Viable for Risk Screening / 2FA Triggering, NOT for Autonomous Transaction Blocking.**
- **Generalization:** Model genuinely detects large transfers, attack channels, and liquidation behavior ($97.22\%$ recall), but overfits to synthetic negative distribution assumptions.
- **Deployment Recommendation:** Deploy a three-tier policy: Auto-Approve ($\hat{p} < 0.40$), Step-Up 2FA / Biometrics ($0.40 \le \hat{p} < 0.70$), and Hold for AML Review ($\hat{p} \ge 0.70$). Raising threshold to $0.70$ eliminates **2,368 false alarms (22.6% reduction)** while capturing **89.58% of all fraud attacks**.
- **Overall Reliability Rating:** **MODERATELY RELIABLE (GRADE: B+)**.

### 6. Audit Deliverables Index
- [Audit Synthesis & Technical Interview Defense](unseen_data_audit/final_model_reliability_report.md)
- [Project Inspection & Baseline Report](unseen_data_audit/project_inspection_report.md)
- [Synthetic Leakage & Artifact Forensic Report](unseen_data_audit/data_leakage_audit.md)
- [12,000 Unseen Data Evaluation Report](unseen_data_audit/unseen_data_evaluation_report.md)
- [6 Distribution Shifts Stress-Test Report](unseen_data_audit/distribution_shift_report.md)
- [Noise Perturbation Robustness Report](unseen_data_audit/robustness_test_report.md)
- [Generalization Gap & Threshold Sensitivity Report](unseen_data_audit/model_generalization_report.md)
- [Permutation Feature Importance Report](unseen_data_audit/feature_importance_report.md)

---

## 📁 Repository Structure

```text
Fraud_Detection/
├── data/
│   ├── raw/
│   │   └── Fraud_Analysis_Dataset.csv        # BIA official dataset (11,142 records)
│   └── processed/
│       ├── train_pre.csv / test_pre.csv      # Pre-transaction splits (13 features)
│       └── train_post.csv / test_post.csv    # Post-transaction splits (23 features)
├── dashboard/
│   ├── app.py                                # Main Streamlit app (7 pages)
│   ├── components.py                         # Reusable UI & CSS styling
│   └── utils.py                              # Dual-model loading & prediction engine
├── models/
│   ├── pre_transaction_model.joblib          # Random Forest Pre-Tx Champion
│   ├── pre_transaction_pipeline.joblib       # Pre-Tx Preprocessor ColumnTransformer
│   ├── pre_transaction_metadata.json         # Pre-Tx threshold (0.21) & features (16)
│   ├── post_transaction_model.joblib         # XGBoost Post-Tx Champion
│   ├── post_transaction_pipeline.joblib      # Post-Tx Preprocessor ColumnTransformer
│   ├── post_transaction_metadata.json        # Post-Tx threshold (0.07) & features (26)
│   ├── best_model.joblib                     # Backward-compatible model artifact
│   ├── preprocessing_pipeline.joblib         # Backward-compatible preprocessor
│   └── model_metadata.json                   # Backward-compatible metadata
├── notebooks/
│   └── Fraud_Detection_Analysis.ipynb        # Complete executed capstone notebook
├── unseen_data_audit/                        # Independent Unseen Data Stress-Testing Suite
│   ├── data/                                 # 12,000 unseen txs, permutation importance, JSONs
│   ├── figures/                              # 4 high-res generalization & shift diagnostic figures
│   ├── project_inspection_report.md          # Architectural & pipeline inspection
│   ├── data_leakage_audit.md                 # Forensic synthetic artifact & liquidation analysis
│   ├── unseen_data_evaluation_report.md      # Test A vs. Test B (12k unseen) empirical breakdown
│   ├── distribution_shift_report.md          # 6 operational shift stress tests
│   ├── robustness_test_report.md             # Continuous Gaussian noise jitter tests (5%–20%)
│   ├── model_generalization_report.md        # Generalization gap & threshold sensitivity
│   ├── feature_importance_report.md          # Permutation feature importance on unseen data
│   └── final_model_reliability_report.md     # Master synthesis, Phase 9 verdict & Phase 10 defense
├── reports/
│   ├── figures/                                   # High-resolution diagnostic charts
│   ├── Fraud_Detection.pbix                       # 5-page Power BI Desktop dashboard
│   ├── FraudGuard_AI_PowerBI_Dashboard_Summary.md # Power BI KPI, DAX & layout documentation
│   ├── fraud_detection_bi_dataset.csv             # Enriched BI export (11,142 records)
│   ├── Fraud_Detection_Final_Report.docx          # Formal 14-section academic report
│   ├── AUDIT_REPORT.md                            # Comprehensive 10-point audit report
│   ├── FINAL_INDEPENDENT_AUDIT.md                 # Independent ML Auditor sign-off report
│   ├── feature_audit.json                         # Feature dictionary with availability timing
│   ├── pre_transaction_evaluation.json            # Pre-Tx benchmark metrics
│   ├── post_transaction_evaluation.json           # Post-Tx benchmark metrics
│   ├── model_comparison.json                      # Cross-architecture comparison
│   ├── threshold_analysis.json                    # Trade-off curve data
│   ├── temporal_generalization_report.json        # Multi-tier temporal drift report
│   ├── temporal_generalization_report.csv         # Walk-forward validation metrics
│   ├── TEMPORAL_FEATURE_ABLATION.md               # Independent feature ablation report
│   ├── temporal_feature_ablation.json             # Ablation experiment metrics
│   ├── temporal_feature_ablation.csv              # Ablation experiment summary
│   └── financial_impact_analysis.json             # Enterprise ROI report
├── src/
│   ├── data_loader.py                        # Ingestion & target splitting
│   ├── data_validation.py                    # Quality audit utilities
│   ├── eda.py                                # EDA figure generation
│   ├── feature_engineering.py                # Pre- & Post-Tx feature classes
│   ├── preprocessing.py                      # Leak-free ColumnTransformer pipelines
│   ├── train.py                              # Pipeline CV & threshold calibration
│   ├── evaluate.py                           # Evaluation metrics & curves
│   ├── explainability.py                     # SHAP TreeExplainer integration
│   ├── financial_impact.py                   # Financial risk modeling
│   ├── validation.py                         # Production input validation module
│   ├── temporal_experiment.py                # Temporal generalization suite
│   └── temporal_ablation.py                  # Feature ablation experiment suite
├── tests/
│   ├── test_model.py                         # Model inference & artifact tests
│   ├── test_preprocessing.py                 # Feature engineering & split tests
│   ├── test_dashboard.py                     # Batch prediction & sample tests
│   └── test_validation.py                    # Validation, leakage, & parity tests
├── requirements.txt                          # Pinned production dependencies
├── run_project.py                            # End-to-end execution pipeline
└── README.md                                 # Project documentation
```

---

## 🚀 Installation & Execution Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install required dependencies:
```bash
# Clone repository
git clone https://github.com/irfan-shekh/Fraud_Detection.git
cd Fraud_Detection

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
Run via terminal:
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run the Comprehensive Unit Test Suite
Execute all 25 unit tests across data validation, feature restrictions, model loading, and batch processing:
```bash
python -m pytest -v
```

### 4. Run the Temporal Generalization & Feature Ablation Experiments
Executes macro out-of-time holdouts, walk-forward expanding window validation, and independent feature ablation:
```bash
# Run temporal drift and walk-forward suite
python src/temporal_experiment.py

# Run temporal feature ablation suite (Exp A, Exp B, Exp C)
python src/temporal_ablation.py
```

### 5. Run the Complete End-to-End Pipeline
Executes data quality audit, EDA, feature engineering, pipeline fitting, model training, tuning, SHAP explainability, and financial impact analysis:
```bash
python run_project.py
```

---

## 📜 Academic Integrity & Citation

This capstone project is submitted in partial fulfillment of the requirements for the Post Graduate Diploma in Data Science & Artificial Intelligence at **Boston Institute of Analytics (BIA)**.

All metrics, visualizations, and financial figures reflect genuine model evaluations on the official BIA dataset without fabrication or synthetic manipulation.
