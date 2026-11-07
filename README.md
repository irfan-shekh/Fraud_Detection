# 🛡️ FRAUDGUARD AI: Intelligent Financial Fraud Detection System
### Boston Institute of Analytics (BIA) — Data Science Capstone Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frauddetection-17.streamlit.app)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Pytest Passed](https://img.shields.io/badge/pytest-12%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning intelligence platform developed for the **Boston Institute of Analytics (BIA)** Data Science Capstone. The system detects fraudulent financial transactions, evaluates enterprise monetary exposure, provides model explainability through SHAP, and delivers an interactive analytics dashboard via Streamlit.

🔗 **Live Production Application:** [https://frauddetection-17.streamlit.app](https://frauddetection-17.streamlit.app)

---

## 📌 Executive Summary & Key Results

| Metric | Result | Context / Business Impact |
|---|---|---|
| **Champion Model** | **Random Forest Classifier** | Hyperparameter-tuned ensemble of 100 decision trees |
| **Optimal Threshold** | **0.21** | Calibrated via 5-fold cross-validation on training data |
| **Test Accuracy** | **99.96%** | Untouched test partition (2,229 transactions) |
| **Test Precision** | **100.00%** | **0 False Positives** (Zero false alarms on test set) |
| **Test Recall** | **99.56%** | **227 out of 228** fraudulent transactions detected |
| **Test F1-Score** | **0.9978** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **0.99999** | Near-perfect class separability |
| **PR-AUC** | **0.99994** | Gold standard evaluation metric for imbalanced fraud data |
| **Total Test Fraud Value** | **$235,846,526.37** | Total monetary fraud volume in test partition |
| **Detected Fraud Value** | **$235,430,525.04** | **99.82%** of all fraudulent dollars intercepted |
| **Missed Fraud Exposure** | **$416,001.33** | Single missed fraud instance |
| **Net Financial Benefit** | **$188,337,610.03** | Estimated net loss prevented under 80% recovery SLA |

---

## 💼 Business Problem & Motivation

Modern electronic funds transfers require instant clearing mechanisms. Fraudulent actors exploit digital banking channels through unauthorized account takeovers (ATO), rapidly draining balances and liquidating capital through secondary cash-out routes.

Financial institutions face an asymmetric risk structure:
1. **False Negatives (Missed Fraud):** Cause catastrophic capital losses, regulatory fines (AML/CFT), chargeback costs, and customer churn. In this dataset, high-value transfers reach up to **$10,000,000**.
2. **False Positives (False Alarms):** Incur unnecessary manual investigation costs (**$15–$60 per alert**) and introduce payment friction that alienates legitimate customers.

**Core Objective:** Maximize fraud detection recall (intercepting maximum fraudulent capital) while maintaining near-zero false alarms.

---

## 📊 Dataset Architecture & Characteristics

The project is built exclusively on the official BIA benchmark dataset (`Fraud_Analysis_Dataset.csv`):
- **Total Records:** 11,142 transactions
- **Legitimate Records:** 10,000 (89.75%)
- **Fraudulent Records:** 1,142 (10.25%)
- **Class Imbalance:** Approximately 1:8.76
- **Data Hygiene:** 0 missing values, 0 duplicate records, 0 negative amounts/balances
- **Temporal Span:** 95 hours (~4 operating days)

### Critical Domain Insights Uncovered during EDA:
- **100% Channel Specificity:** All 1,142 frauds occurred exclusively in `TRANSFER` (564 frauds; 38.5% channel rate) and `CASH_OUT` (578 frauds; 30.9% channel rate). Exactly zero frauds occurred in `PAYMENT`, `CASH_IN`, or `DEBIT`.
- **Systematic Account Drainage:** In **99.2%** of fraudulent transactions, the sender balance was liquidated to **$0.00**. In **97.1%**, the transaction amount exactly equaled the starting balance.
- **Monetary Disparity:** Fraudulent transactions averaged **$1,192,629** (median $441,496), compared to **$101,339** (median $11,940) for legitimate transactions.

---

## 📐 Feature Engineering

To provide deep discriminative signals to tree models, 12 domain features were constructed:
1. `origin_balance_difference = oldbalanceOrg - newbalanceOrig`
2. `destination_balance_difference = newbalanceDest - oldbalanceDest`
3. `balance_error_orig = oldbalanceOrg - amount - newbalanceOrig` (captures ledger accounting mismatch)
4. `balance_error_dest = oldbalanceDest + amount - newbalanceDest`
5. `origin_balance_zero = (newbalanceOrig == 0)` (flag for complete balance drainage)
6. `empty_account_flag = (amount == oldbalanceOrg)` (flag for 100% account drainage)
7. `transaction_hour = step % 24`
8. `transaction_day = ((step - 1) // 24) + 1`
9. `transaction_amount_log = log1p(amount)`
10. `merchant_destination = (nameDest starts with 'M')`
11. `error_orig_zero = (abs(balance_error_orig) < 0.01)`
12. `error_dest_zero = (abs(balance_error_dest) < 0.01)`

**Impact:** Incorporating engineered features improved Random Forest test recall from **96.49% to 99.56%** with zero precision penalty.

---

## 🔬 Multi-Model Benchmark Comparison

All models were evaluated on the untouched 20% test partition (2,229 records; 228 actual frauds):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Random Forest (Champion)** | **0.9996** | **1.0000** | **0.9956** | **0.9978** | **0.99999** | **0.99994** |
| **XGBoost Classifier** | 0.9996 | 1.0000 | 0.9956 | 0.9978 | 0.99972 | 0.99844 |
| **Decision Tree Classifier** | 0.9996 | 1.0000 | 0.9956 | 0.9978 | 0.99781 | 0.99606 |
| **Logistic Regression (Class-Weighted)** | 0.9987 | 0.9956 | 0.9912 | 0.9934 | 0.99634 | 0.99610 |
| **Isolation Forest (Unsupervised)** | 0.9480 | 0.7617 | 0.7149 | 0.7376 | 0.97470 | 0.81951 |

---

## 🧠 Explainable AI (SHAP)

To satisfy financial compliance (FCRA, GDPR Article 22), SHAP tree explainers were deployed:
- **Primary Driver:** `balance_error_orig` (accounting mismatch) contributes the strongest global positive attribution to fraud probability.
- **Secondary Drivers:** `empty_account_flag` and `origin_balance_zero` strongly confirm malicious intent.
- **Transaction Amount:** Amplifies risk score non-linearly above $200,000.

---

## 💰 Financial Risk & Scenario Modeling

| Enterprise Scenario | Fraud Recovery Rate | Investigation Cost / Alert | FP Friction Rate | Gross Prevented Loss ($) | Operational Cost ($) | Net Financial Benefit ($) |
|---|---|---|---|---|---|---|
| **Low Cost / High Efficiency** | 90% | $15.00 | 0.5% | $211,887,472.54 | $3,405.00 | **$211,884,067.54** |
| **Medium (Base Case)** | **80%** | **$30.00** | **1.5%** | **$188,344,420.03** | **$6,810.00** | **$188,337,610.03** |
| **High Cost / Conservative** | 70% | $60.00 | 3.0% | $164,801,367.53 | $13,620.00 | **$164,787,747.53** |

---

## 🖥️ Streamlit Web Application (FRAUDGUARD AI)

The production dashboard contains 6 interactive modules:
1. **📊 Dashboard Overview:** Executive KPIs, interactive Plotly donut & bar charts, hourly velocity line chart.
2. **⚡ Single Transaction Prediction:** Form with pre-loaded benchmark scenarios, calibrated risk gauge, contributing factor bullets, and suggested actions.
3. **📂 Batch Transaction Prediction:** Multi-row CSV screening engine with validation, summary KPIs, and downloadable enriched CSV report.
4. **📈 Model Performance Hub:** Live confusion matrix, multi-model ROC & PR curves, and dynamic decision threshold slider.
5. **🧠 Explainable AI Hub:** SHAP global summary beeswarm, feature importance bars, and individual fraud/legitimate waterfall plots.
6. **💰 Financial Impact Analysis:** Interactive ROI simulator allowing risk officers to adjust recovery and review costs dynamically.

---

## 📁 Repository Structure

```text
Fraud_Detection/
├── data/
│   ├── raw/
│   │   └── Fraud_Analysis_Dataset.csv      # BIA official dataset
│   └── processed/
│       ├── train.csv                       # Stratified train split
│       └── test.csv                        # Untouched test split
├── notebooks/
│   └── Fraud_Detection_Analysis.ipynb      # Complete 20-section executed notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                      # Data ingestion utilities
│   ├── data_validation.py                  # Quality audit & JSON report
│   ├── eda.py                              # Generates 15+ EDA figures
│   ├── feature_engineering.py              # Domain ledger feature engineering
│   ├── preprocessing.py                    # Leak-free ColumnTransformer pipeline
│   ├── train.py                            # Model training & hyperparameter search
│   ├── evaluate.py                         # Evaluation metrics, ROC, PR, threshold
│   ├── explainability.py                   # SHAP summary & waterfall charts
│   └── financial_impact.py                 # Financial ROI & scenario modeling
├── models/
│   ├── best_model.joblib                   # Serialized Random Forest champion
│   ├── preprocessing_pipeline.joblib       # Fitted preprocessor ColumnTransformer
│   └── model_metadata.json                 # Model hyperparameters & threshold (0.21)
├── reports/
│   ├── figures/                            # 27 high-resolution diagnostic charts
│   ├── model_comparison.csv                # Benchmark evaluation metrics
│   ├── feature_engineering_comparison.csv  # Ablation study metrics
│   ├── imbalance_techniques_comparison.csv # SMOTE / RUS / Weighting study
│   ├── evaluation_report.json              # Full evaluation JSON
│   ├── data_quality_report.json            # Quality audit JSON
│   ├── financial_impact_analysis.json      # Financial ROI report JSON
│   ├── FINAL_INDEPENDENT_AUDIT.md          # Independent audit verification report
│   ├── BI_DASHBOARD_GUIDE.md               # Power BI & Tableau DAX schema guide
│   ├── EXECUTIVE_PITCH_SHEET.md            # 1-page executive board brief
│   └── fraud_detection_bi_dataset.csv      # Enriched BI export dataset
├── dashboard/
│   ├── app.py                              # Main Streamlit application
│   ├── components.py                       # Reusable Plotly & CSS components
│   └── utils.py                            # Inference & batch processing logic
├── tests/
│   ├── test_preprocessing.py               # Unit tests for data & preprocessing
│   ├── test_model.py                       # Unit tests for model inference
│   └── test_dashboard.py                   # Unit tests for batch & sample CSVs
├── requirements.txt                        # Pinned dependencies
├── README.md                               # Project documentation
├── .gitignore                              # Git exclusion rules
├── run_project.py                          # Master end-to-end execution pipeline
└── Fraud_Analysis_Dataset.csv              # Working copy of BIA dataset
```

### 🎓 Submission Deliverables Location
All finalized academic submission deliverables are archived in [`C:\Projects\BIA_Capstone\`](file:///C:/Projects/BIA_Capstone):
- `Fraud_Detection_Final_Report.docx` (Formal 20-section report, 662 KB)
- `Fraud_Detection_Presentation.pptx` (15-slide presentation, 876 KB)
- `Fraud_Detection_BIA_Final_Submission.zip` (Full submission archive, 6.08 MB)
- `FINAL_INDEPENDENT_AUDIT.md` (Audit verification report)
- `EXECUTIVE_PITCH_SHEET.md` (Executive brief)

---

## 🚀 Installation & Execution Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install required dependencies:
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
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

### 3. Run the Unit Test Suite
Verify that all 12 unit tests pass:
```bash
python -m pytest -v
```

### 4. Run the Complete End-to-End Pipeline
Executes data quality audit, EDA, feature engineering, pipeline fitting, model training, tuning, SHAP explainability, and financial impact analysis:
```bash
python run_project.py
```

---

## ⚠️ Limitations & Strategic Roadmap

### Limitations:
- **Simulated Multi-Agent Dynamics:** Synthetic PaySim benchmarks lack real-world device fingerprints, browser User-Agent strings, and IP geolocations.
- **Channel Concentration:** Fraud occurs exclusively in `TRANSFER` and `CASH_OUT`. Card-not-present (CNP) e-commerce fraud is unrepresented.
- **Account Non-Recurrence:** All sender identifiers are unique, preventing recursive longitudinal customer profiling.

### Strategic Roadmap:
- **Graph Neural Networks (GNN):** Construct inter-account topology graphs to detect multi-hop laundering syndicates.
- **Sub-10ms Streaming Pipelines:** Implement Apache Kafka and Apache Flink for real-time transaction scoring.
- **Active Feedback Loops:** Integrate analyst dispute dispositions to continuously adapt to adversarial fraud drift.

---

## 📜 Academic Integrity & Citation

This capstone project is submitted in partial fulfillment of the requirements for the Post Graduate Diploma in Data Science & Artificial Intelligence at **Boston Institute of Analytics (BIA)**.

All metrics, visualizations, and financial figures reflect genuine model evaluations on the provided BIA dataset without fabrication or synthetic manipulation.
