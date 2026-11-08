# 🛡️ FRAUDGUARD AI: Intelligent Financial Fraud Detection System
### Boston Institute of Analytics (BIA) — Data Science Capstone Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frauddetection-17.streamlit.app)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pytest Passed](https://img.shields.io/badge/pytest-23%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade Machine Learning intelligence platform developed for the **Boston Institute of Analytics (BIA)** Data Science Capstone. The system features a **Dual-Architecture Engine** that rigorously distinguishes between **Pre-Transaction Fraud Screening** (real-time pre-authorization switch with zero leakage) and **Post-Transaction Fraud Detection** (post-clearing ledger reconciliation & audit), evaluates enterprise monetary exposure, provides model explainability through SHAP, and delivers an interactive 7-page analytics dashboard via Streamlit.

🔗 **Live Production Application:** [https://frauddetection-17.streamlit.app](https://frauddetection-17.streamlit.app)

---

## 📌 Executive Summary & Key Results

| Metric | Pre-Transaction Screening Model | Post-Transaction Detection Model | Context / Business Impact |
|---|---|---|---|
| **Operational Timing** | **Pre-Authorization (Sub-50ms Gateway)** | **Post-Settlement (Audit & Dispute)** | Separation eliminates target & ledger leakage |
| **Champion Algorithm** | **Random Forest Classifier** | **XGBoost Classifier** | Selected via leak-free 5-Fold Stratified CV |
| **Input Dimensions** | **16 Encoded Features** (13 unencoded) | **26 Encoded Features** (23 unencoded) | Exact dimensions audited via metadata |
| **Calibrated Threshold** | **0.21** (Optimized on Train OOF) | **0.07** (Optimized on Train OOF) | Threshold chosen via business objective |
| **Test Accuracy** | **99.96%** (2,228 / 2,229 correct) | **99.96%** (2,228 / 2,229 correct) | Untouched test partition (2,229 records) |
| **Test Precision** | **100.00%** (0 False Positives) | **100.00%** (0 False Positives) | Zero legitimate transactions falsely blocked |
| **Test Recall** | **99.56%** (227 / 228 Detected) | **99.56%** (227 / 228 Detected) | Intercepts 99.82% of all fraudulent capital |
| **Test F1-Score** | **0.9978** | **0.9978** | Harmonic mean of Precision and Recall |
| **Test ROC-AUC** | **0.99996** | **0.99999** | Near-perfect class separability |
| **Test PR-AUC** | **0.99961** | **0.99994** | Benchmark metric for imbalanced fraud data |
| **False Positives** | **0** | **0** | Zero customer friction on legitimate accounts |
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
├── reports/
│   ├── figures/                              # High-resolution diagnostic charts
│   ├── AUDIT_REPORT.md                       # Comprehensive 10-point audit report
│   ├── feature_audit.json                    # Feature dictionary with availability timing
│   ├── pre_transaction_evaluation.json       # Pre-Tx benchmark metrics
│   ├── post_transaction_evaluation.json      # Post-Tx benchmark metrics
│   ├── model_comparison.json                 # Cross-architecture comparison
│   ├── threshold_analysis.json               # Trade-off curve data
│   └── financial_impact_analysis.json        # Enterprise ROI report
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
│   └── validation.py                         # Production input validation module
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
Execute all 23 unit tests across data validation, feature restrictions, model loading, and batch processing:
```bash
python -m pytest -v
```

### 4. Run the Complete End-to-End Pipeline
Executes data quality audit, EDA, feature engineering, pipeline fitting, model training, tuning, SHAP explainability, and financial impact analysis:
```bash
python run_project.py
```

---

## 📜 Academic Integrity & Citation

This capstone project is submitted in partial fulfillment of the requirements for the Post Graduate Diploma in Data Science & Artificial Intelligence at **Boston Institute of Analytics (BIA)**.

All metrics, visualizations, and financial figures reflect genuine model evaluations on the official BIA dataset without fabrication or synthetic manipulation.
