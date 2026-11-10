# FraudGuard AI --- Power BI Dashboard

**Financial Fraud Detection & Analytics Dashboard**\
**Project:** FraudGuard AI --- Dual-Engine Financial Fraud Detection
System\
**Report:** Five-page interactive Power BI dashboard\
**Dataset:** Synthetic financial transaction dataset

## 1. Project Overview

FraudGuard AI is a financial fraud detection project supported by a
five-page Power BI dashboard. The report brings together transaction
analytics and previously reported machine-learning evaluation results.

The five pages are: 1. Executive Overview 2. Transaction Analysis 3. AI
Model Performance 4. Risk Intelligence 5. Temporal & Financial Analysis

### Dataset summary

Measure Value

---

Total transactions 11,142
Actual fraud transactions 1,142
Actual fraud rate 10.25%
Actual fraudulent transaction value Approximately 1.36B
Simulation steps 95

> **Dataset disclosure:** The dataset is synthetic. Results should not
> be presented as evidence of performance on live banking transactions.

## 2. Dashboard Design

The report uses a consistent dark navy theme.

Element Theme

---

Main page background `#071629`
Visual background `#10243D`
Cyan `#28BFFF`
Red `#FF5263`
Purple `#8767F5`
Green `#35D49A`
Amber `#F0B44C`
Main text White

## 3. Page 1 --- Executive Overview

**Purpose:** Provide a high-level summary of transaction activity and
fraud.

### KPI cards

KPI Value

---

Total Transactions 11,142
Fraud Transactions 1,142
Fraud Rate 10.25%
Fraud Value Approximately 1.36B

### Visuals

1.  **Transactions by Type** --- donut chart.
2.  **Fraud Transactions by Type** --- bar chart.
3.  **Fraud Transactions by Step** --- column chart.

**What it communicates:** The scale of transaction activity, the
proportion of transactions labelled fraudulent, and how fraud is
distributed across transaction types and simulation steps.

## 4. Page 2 --- Transaction Analysis

**Purpose:** Explore transaction behavior and financial patterns.

### Slicers

- Transaction Type
- Risk Tier
- Prediction Status

### Visuals

1.  **Total Transaction Value by Type** --- compares total transaction
    value across transaction types.
2.  **Fraudulent Transaction Value by Type** --- compares fraudulent
    transaction value by type.
3.  **Transaction Amount Distribution** --- histogram of transaction
    amounts.
4.  **Average Transaction Amount by Type** --- compares average
    transaction amounts by type.

**What it communicates:** Which transaction types account for the
largest transaction values, how fraudulent transaction value is
distributed, and how transaction amounts differ by type.

## 5. Page 3 --- AI Model Performance

**Purpose:** Present the reported evaluation results for the two
machine-learning engines and compare their metrics.

### Random Forest confusion matrix

Actual / Predicted Fraud Legitimate

---

Fraud 227 1
Legitimate 0 2,001

### Random Forest metrics

Metric Reported result

---

Precision 100.00%
Recall 99.56%
F1 Score 99.78%
ROC-AUC 99.9965% (approximately 99.997%)
PR-AUC 99.9614% (approximately 99.96%)

### Reported model comparison

Metric Pre-Transaction Random Forest Post-Transaction XGBoost

---

Precision 100.00% 99.56%
Recall 99.56% 99.56%
F1 Score 99.78% 99.56%
ROC-AUC 99.9965% 99.8995%
PR-AUC 99.9614% 99.6999%

### Visuals

1.  True Positives, False Positives, True Negatives, and False Negatives
    KPI cards.
2.  Random Forest confusion matrix.
3.  Precision, Recall, and F1 Score cards.
4.  ROC-AUC and PR-AUC cards.
5.  Random Forest versus XGBoost comparison table.
6.  Clustered column chart comparing Precision, Recall, and F1 Score.

### Interpretation and caveats

- Random Forest is the **pre-transaction** model.
- XGBoost is the **post-transaction** model.
- The models use different feature availability. Their metrics should
  be interpreted in the context of their different roles, not as a
  perfectly like-for-like production comparison.
- These are previously reported held-out test results. They are not
  recalculated by the Power BI report from the full transaction
  dataset.

## 6. Page 4 --- Risk Intelligence

**Purpose:** Show model-assigned risk categories, fraud-probability
distribution, and the relationship between risk tiers and actual fraud
labels.

### Risk-tier distribution

Risk tier Transactions

---

Critical Risk 1,139
High Risk 2
Minimal Risk 10,001
**Total** **11,142**

### Actual versus predicted fraud counts

Measure Count

---

Actual fraud transactions 1,142
Predicted fraud transactions 1,141

The Predicted Fraud KPI remains 1,141 because diagnostic DAX measures
returned 1,141 even after removing the Risk Tier filter. Do not manually
change it to 1,142 without reconciling the underlying prediction data.

### Visuals

1.  Critical Risk KPI.
2.  High Risk KPI.
3.  Minimal Risk KPI.
4.  Predicted Fraud KPI.
5.  Transaction Distribution by Risk Tier --- donut chart.
6.  Fraud Probability Distribution --- histogram.
7.  Actual Fraud Transactions by Risk Tier --- bar chart.
8.  Actual vs Predicted Fraud by Risk Tier --- clustered bar chart.
9.  Fraudulent Transaction Value by Risk Tier --- bar chart.

### Interpretation and caveats

- Risk tiers are model-assigned classifications; they are not
  independently verified fraud outcomes.
- A high fraud probability is a model score, not proof that a
  transaction is fraudulent.
- Matching aggregate counts by risk tier does not establish that the
  same individual transactions were correctly classified.

## 7. Page 5 --- Temporal & Financial Analysis

**Purpose:** Analyze how fraud frequency and fraudulent transaction
value vary across simulation steps, and compare financial exposure by
transaction type.

### KPI cards

KPI Value

---

Fraud Transactions 1,142
Fraudulent Value Approximately 1.36B
Fraud Rate 10.25%
Simulation Steps 95

### Visuals

1.  **Fraud Transactions Over Time** --- line chart showing fraud counts
    by simulation step.
2.  **Fraud Value Over Time** --- area chart showing fraudulent
    transaction value by simulation step.
3.  **Fraud Transactions by Type Over Time** --- stacked column chart
    comparing fraud counts by type and simulation step.
4.  **Fraud Exposure by Transaction Type** --- horizontal bar chart
    comparing fraudulent transaction value by type.

### Reported financial exposure by transaction type

Transaction type Fraudulent transaction value

---

TRANSFER Approximately 681.6M
CASH_OUT Approximately 680.38M

### Temporal-data limitation

The supplied synthetic dataset has an unusual temporal pattern:
legitimate transactions largely stop appearing after simulation step 7,
while later steps contain fraud activity. This limits claims about
real-world temporal generalization. Simulation steps are not calendar
dates.

## 8. Key DAX Measures

The following are representative measures used in the report. Confirm
that the table and column names match the Power BI model before using
them.

### Total transactions

```dax
Total Transactions =
COUNTROWS(FactTransactions)
```

### Actual fraud transactions

```dax
Fraud Transactions =
CALCULATE(
    COUNTROWS(FactTransactions),
    FactTransactions[isFraud] = 1
)
```

### Actual fraud rate

```dax
Fraud Rate =
DIVIDE(
    [Fraud Transactions],
    [Total Transactions],
    0
)
```

Format as Percentage.

### Actual fraudulent transaction value

```dax
Fraudulent Transaction Value =
CALCULATE(
    SUM(FactTransactions[amount]),
    FactTransactions[isFraud] = 1
)
```

### Predicted fraud transactions

```dax
Predicted Fraud Count =
CALCULATE(
    COUNTROWS(FactTransactions),
    FactTransactions[Predicted_Fraud] = 1
)
```

### Fraud transactions over time

```dax
Temporal Fraud Count =
CALCULATE(
    COUNTROWS(FactTransactions),
    FactTransactions[isFraud] = 1
)
```

Use `step` on the chart axis.

### Fraud value over time

```dax
Temporal Fraud Value =
CALCULATE(
    SUM(FactTransactions[amount]),
    FactTransactions[isFraud] = 1
)
```

Use `step` on the chart axis.

### Fraud exposure by transaction type

```dax
Fraud Exposure by Type =
CALCULATE(
    SUM(FactTransactions[amount]),
    FactTransactions[isFraud] = 1
)
```

Use `type` as the chart category.

## 9. Final Validation Checklist

Before publishing the `.pbix` report, verify:

- [ ] Actual fraud count is 1,142.
- [ ] Actual fraud rate is 10.25%.
- [ ] Predicted fraud KPI is 1,141 unless the source prediction data
      is reconciled and confirms a different value.
- [ ] Slicers filter the intended visuals correctly.
- [ ] Page 3 displays held-out test metrics, not metrics recalculated
      from the full dataset.
- [ ] Pre-transaction Random Forest and post-transaction XGBoost are
      clearly distinguished.
- [ ] Risk tiers are described as model-assigned classifications.
- [ ] Fraudulent transaction value is not described as money recovered
      or blocked.
- [ ] The synthetic nature of the dataset is disclosed.
- [ ] The temporal limitation after simulation step 7 is disclosed.
- [ ] Titles, fonts, colors, and footers are consistent across all
      five pages.

## 10. Project Presentation Summary

FraudGuard AI is supported by a five-page Power BI dashboard built to
analyze 11,142 synthetic financial transactions. The report provides an
executive overview, transaction behavior analysis, machine-learning
model evaluation, risk intelligence, and temporal and financial exposure
analysis.

It brings together Power BI visualizations and DAX measures to make
transaction patterns, reported model performance, risk classifications,
and financial exposure easier to explore.

```

Check dataset and internship/provider sharing terms before uploading the
dataset or restricted files. If the `.pbix` contains sensitive data or
credentials, remove them before publishing.

------------------------------------------------------------------------

**Project:** FraudGuard AI --- Dual-Engine Financial Fraud Detection
System\
**Dashboard:** Power BI --- Five-page Financial Fraud Detection &
Analytics Report\
**Data:** Synthetic transaction dataset
```
