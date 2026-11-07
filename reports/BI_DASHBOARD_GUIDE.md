# POWER BI & TABLEAU IMPLEMENTATION GUIDE
## BIA Capstone Project: Fraud Detection Dashboard
**Data Source:** `reports/fraud_detection_bi_dataset.csv` (11,142 rows, 28 columns)  
**System:** FRAUDGUARD AI Business Intelligence Kit  

---

## 1. Dataset Overview for Business Intelligence Tools

The exported file [`reports/fraud_detection_bi_dataset.csv`](file:///c:/Projects/Fraud_Detection/reports/fraud_detection_bi_dataset.csv) contains the complete BIA transaction dataset augmented with model predictions, probabilities, prediction status, risk categories, and engineered features:

| Column Name | Data Type | Usage in BI Dashboard |
|---|---|---|
| `step` | Whole Number | Time dimension (hours 1 to 95) |
| `type` | Text | Dimension for channel slicer / bar charts |
| `amount` | Decimal / Currency | Core monetary metric |
| `nameOrig`, `nameDest` | Text | Account identifiers (drill-through tables) |
| `oldbalanceOrg`, `newbalanceOrig` | Decimal / Currency | Sender financial balances |
| `oldbalanceDest`, `newbalanceDest`| Decimal / Currency | Recipient financial balances |
| `isFraud` | Whole Number (0/1) | Actual Ground Truth class |
| `Predicted_Fraud` | Whole Number (0/1) | Model classification outcome |
| `Fraud_Probability` | Decimal (0.0 to 1.0) | Gauge / probability distribution |
| `Prediction_Status` | Text | `True Positive`, `False Positive`, `True Negative`, `False Negative` |
| `Risk_Tier` | Text | `Critical Risk`, `High Risk`, `Moderate Risk`, `Minimal Risk` |
| `balance_error_orig` | Decimal / Currency | Ledger accounting mismatch metric |
| `empty_account_flag` | Whole Number (0/1) | 100% balance depletion indicator |

---

## 2. Power BI Setup & DAX Measures

### Step 1: Import Data
1. Open **Power BI Desktop**.
2. Click **Get Data** -> **Text/CSV**.
3. Select `reports/fraud_detection_bi_dataset.csv` and click **Load**.

### Step 2: Create Core DAX Measures
Create a new measure table or add the following DAX formulas:

```dax
// 1. Total Transaction Count
Total Transactions = COUNTROWS('fraud_detection_bi_dataset')

// 2. Total Fraudulent Transactions (Actual)
Actual Frauds = CALCULATE(COUNTROWS('fraud_detection_bi_dataset'), 'fraud_detection_bi_dataset'[isFraud] = 1)

// 3. Model Detected Frauds (True Positives)
True Positives = CALCULATE(COUNTROWS('fraud_detection_bi_dataset'), 'fraud_detection_bi_dataset'[Prediction_Status] = "True Positive")

// 4. Model False Alarms (False Positives)
False Positives = CALCULATE(COUNTROWS('fraud_detection_bi_dataset'), 'fraud_detection_bi_dataset'[Prediction_Status] = "False Positive")

// 5. Missed Frauds (False Negatives)
False Negatives = CALCULATE(COUNTROWS('fraud_detection_bi_dataset'), 'fraud_detection_bi_dataset'[Prediction_Status] = "False Negative")

// 6. Model Precision
Model Precision = DIVIDE([True Positives], [True Positives] + [False Positives], 1)

// 7. Model Recall
Model Recall = DIVIDE([True Positives], [True Positives] + [False Negatives], 0)

// 8. Total Fraud Volume ($)
Total Fraud Amount = CALCULATE(SUM('fraud_detection_bi_dataset'[amount]), 'fraud_detection_bi_dataset'[isFraud] = 1)

// 9. Intercepted Fraud Volume ($)
Intercepted Fraud Amount = CALCULATE(SUM('fraud_detection_bi_dataset'[amount]), 'fraud_detection_bi_dataset'[Prediction_Status] = "True Positive")

// 10. Net Financial Benefit ($ Base Case: 80% Recovery, $30 Review Cost)
Net Financial Benefit = ([Intercepted Fraud Amount] * 0.80) - (([True Positives] + [False Positives]) * 30)
```

### Step 3: Recommended Power BI Dashboard Layout
- **Top Row (KPI Cards):** `Total Transactions`, `Actual Frauds`, `Model Recall (99.56%)`, `Model Precision (100%)`, `Net Financial Benefit ($188.3M)`.
- **Left Column:** Donut chart of `Risk_Tier` breakdown; Bar chart of `amount` by `type`.
- **Center:** Clustered column chart showing `Total Transactions` vs `Actual Frauds` across `step` (Simulation Hours).
- **Right Column:** Matrix table of `Prediction_Status` (True Positive, True Negative, False Negative, False Positive) with counts and dollar sums.
- **Slicers:** Slicer on `type` (`TRANSFER`, `CASH_OUT`, etc.) and `Risk_Tier`.

---

## 3. Tableau Setup & Calculated Fields

### Step 1: Connect to Data
1. Open **Tableau Desktop** or **Tableau Public**.
2. Connect to **Text File** -> select `reports/fraud_detection_bi_dataset.csv`.

### Step 2: Create Calculated Fields
- **Fraud Status Label:**
  ```tableau
  IF [isFraud] = 1 THEN "Fraudulent" ELSE "Legitimate" END
  ```
- **Dollar Detection Rate:**
  ```tableau
  SUM(IF [Prediction_Status] = "True Positive" THEN [amount] ELSE 0 END) / 
  SUM(IF [isFraud] = 1 THEN [amount] ELSE 0 END)
  ```
- **Net Prevented Losses:**
  ```tableau
  (SUM(IF [Prediction_Status] = "True Positive" THEN [amount] ELSE 0 END) * 0.80) - 
  (COUNT(IF [Prediction_Status] = "True Positive" OR [Prediction_Status] = "False Positive" THEN 1 END) * 30)
  ```

### Step 3: Dashboard Assembly
1. **Sheet 1 (Executive KPIs):** Text table with `Model Recall`, `Model Precision`, and `Net Prevented Losses`.
2. **Sheet 2 (Channel Vulnerability):** Stacked horizontal bar chart with `type` on Rows, `SUM(amount)` on Columns, colored by `isFraud`.
3. **Sheet 3 (Confusion Matrix Heatmap):** Rows = `isFraud`, Columns = `Predicted_Fraud`, Text = `COUNT(fraud_detection_bi_dataset)`, colored by count.
4. **Sheet 4 (Hourly Velocity):** Line chart of `step` on Columns, `COUNT(fraud_detection_bi_dataset)` on Rows, colored by `isFraud`.
5. Assemble into a single cohesive interactive dashboard with interactive filters on `type` and `Risk_Tier`.

