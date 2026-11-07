# BOSTON INSTITUTE OF ANALYTICS (BIA) — CAPSTONE VIVA VOCE & ORAL DEFENSE GUIDE
## Project Title: FRAUDGUARD AI — Intelligent Financial Fraud Detection System
**Domain:** Financial Technology / Banking Security / Applied Machine Learning  
**Student Reference:** Post Graduate Diploma in Data Science & Artificial Intelligence  

---

## 1. Top 15 Likely Evaluator Questions & High-Scoring Model Answers

### Q1: "Your model achieved 99.56% Recall and 100% Precision. Isn't this too good to be true? Are you sure there is no data leakage?"
**Model Answer:**
> "That was the primary hypothesis we stress-tested during our independent audit. We confirmed that this high performance is **not due to data leakage**, but rather the physical nature of the attack signatures in this dataset:
> 1. In **99.2%** of fraudulent transactions, the sender balance was liquidated to exactly **$0.00**.
> 2. In **97.1%**, the transaction amount exactly equaled the starting balance.
> 3. To prove the model doesn't rely on post-transaction leakage, we conducted an ablation study removing all post-clearing fields (`newbalanceOrig`, `newbalanceDest`, and ledger error terms). Even when restricted purely to **pre-transaction variables** (`amount`, `oldbalanceOrg`, and the ratio `amount / oldbalanceOrg == 1.0`), the model maintains **99.56% Recall** because account draining is observable *before* funds leave the bank.
> 4. Furthermore, there is **0% sender overlap** between train and test sets, and forward chronological testing on future time steps (hours 7 to 95) achieved **97.74% Recall** and **100% Precision**."

---

### Q2: "Why did you select Random Forest over XGBoost as your champion model?"
**Model Answer:**
> "Both Random Forest and XGBoost performed exceptionally well, achieving identical test-set recall (99.56%) and precision (100.00%). However, Random Forest achieved a slightly higher **PR-AUC (0.99994 vs 0.99844)** and **ROC-AUC (0.99999 vs 0.99972)**. 
> Operationally, Random Forest offers lower inference latency jitter, requires fewer hyperparameter sensitivities in production, and provides straightforward TreeSHAP explainability without the extreme gradient sensitivity that boosted trees exhibit on out-of-distribution tabular edge cases."

---

### Q3: "Why did Isolation Forest perform significantly worse (PR-AUC = 0.8195) compared to supervised models?"
**Model Answer:**
> "Isolation Forest is an **unsupervised anomaly detection algorithm** that isolates observations by randomly selecting features and split values. Its fundamental assumption is that anomalies are 'few and different' in geometric space. 
> However, many high-value legitimate transactions (e.g., enterprise payroll or large commercial purchases) look geometrically unusual in dollar terms without being fraudulent. Conversely, fraudulent transactions often mimic normal amounts. Because Isolation Forest lacks ground-truth supervision, it cannot learn the exact mathematical constraint that `balance_error_orig != 0` defines fraud, yielding a higher false alarm rate (Precision = 76.17%)."

---

### Q4: "How did you handle the class imbalance in the dataset?"
**Model Answer:**
> "The raw dataset has an imbalance of **10.25% fraud (1,142) vs 89.75% legitimate (10,000)** (~1:8.76 ratio). We empirically benchmarked four strategies using Scikit-Learn and Imbalanced-Learn:
> 1. **Original Data (No Resampling):** PR-AUC = 0.9993
> 2. **Class-Weighted Learning (`class_weight='balanced'`):** PR-AUC = 0.9998
> 3. **SMOTE (Synthetic Minority Over-sampling Technique):** PR-AUC = 0.9999
> 4. **Random Under-Sampling (RUS):** PR-AUC = 0.9996
> 
> We ensured that SMOTE was applied **strictly within training cross-validation pipelines** and never to validation or test sets to prevent synthetic leakage. Ultimately, class-weighted learning was selected because it achieves near-identical discrimination without synthesizing artificial financial records."

---

### Q5: "How did you choose your decision threshold, and why not use the standard 0.50?"
**Model Answer:**
> "In banking fraud, the cost matrix is heavily asymmetric:
> - A **False Negative** (missing a $10M fraud) causes catastrophic capital loss.
> - A **False Positive** costs only ~$30 to investigate.
> 
> Therefore, we calibrated the decision threshold using **Stratified 5-Fold Cross-Validation strictly on the training partition**. The optimal F1 threshold identified was **`0.21`**. When tested on the untouched test partition, threshold `0.21` delivers the same 227 true positives and 0 false alarms as `0.50`, but provides greater sensitivity buffer against borderline transactions."

---

### Q6: "Tell me about the single transaction your model missed (False Negative = 1)."
**Model Answer:**
> "The model detected 227 of 228 frauds, missing exactly one. 
> - **Details:** Step 1, `CASH_OUT`, Amount = $416,001.33, `oldbalanceOrg = $0.00`, `newbalanceOrig = $0.00`.
> - **Root Cause:** The sender's account was *already at $0.00* before the transaction occurred. Because `oldbalanceOrg` was 0, our primary behavioral indicator `empty_account_flag` (which looks for balance draining) did not trigger, and the model assigned a probability of 0.0435.
> - **Business Value:** In a production setting, this transaction would have been caught by standard core-banking validation rules (overdraft prevention) before reaching the ML scoring layer."

---

### Q7: "How does SHAP work, and what were the top fraud indicators?"
**Model Answer:**
> "SHAP (SHapley Additive exPlanations) is based on cooperative game theory. It calculates the marginal contribution of each feature across all possible feature subsets (coalitions) to explain the difference between the baseline prediction and the actual model output.
> 
> In FRAUDGUARD AI, TreeSHAP revealed:
> 1. `balance_error_orig`: The strongest global positive contributor. When the sender's account balance equation fails to balance, fraud probability spikes.
> 2. `empty_account_flag` & `origin_balance_zero`: Confirm malicious account draining.
> 3. `type_TRANSFER` & `type_CASH_OUT`: Channel weights that isolate the attack vector.
> 4. `amount`: Amplifies risk non-linearly for transfers above $200,000."

---

### Q8: "How does your system address regulatory compliance (e.g., GDPR Article 22, FCRA)?"
**Model Answer:**
> "Under FCRA and GDPR Article 22, individuals have the right not to be subject to solely automated decisions with legal or significant effects without meaningful explanation. 
> FRAUDGUARD AI incorporates:
> 1. **Explainable AI (SHAP):** Every prediction generates a local waterfall breakdown showing exactly which features increased or decreased the risk score.
> 2. **Human-in-the-Loop Tiering:** Transactions between 0.20 and 0.80 probability are routed for step-up two-factor authentication or manual analyst review rather than immediate automated denial.
> 3. **Non-Discrimination:** Account identifiers, demographics, and personal attributes are excluded from the feature space."

---

### Q9: "How did you calculate your financial impact figures?"
**Model Answer:**
> "We clearly separated **measured dataset values** from **hypothetical operational assumptions**:
> - **Measured:** In the test set, total fraud was **$235,846,526.37**. Our model detected **$235,430,525.04 (99.82%)**, missing **$416,001.33 (0.18%)**, with **$0.00** in false alarms.
> - **Assumptions (Base Case):** 80% recovery rate for detected fraud, $30 investigation cost per alert, 1.5% friction cost on false positives.
> - **Calculation:** Gross Loss Prevented = $235.43M × 80% = **$188.34M**. Operational Cost = (227 TP + 0 FP) × $30 = **$6,810**. Net Benefit = **$188,337,610.03**."

---

### Q10: "What happens if a new transaction type like 'CRYPTO_BUY' is introduced? How will your model handle it?"
**Model Answer:**
> "Our `ColumnTransformer` pipeline specifies `OneHotEncoder(handle_unknown='ignore')`. If an unseen categorical channel arrives, the one-hot columns for all known types evaluate to 0 rather than crashing. 
> Furthermore, because our ablation study proved that the model maintains 99.56% recall even without the `type` column, the model would still detect the transaction based on monetary amount and balance drainage signatures."

---

## 2. Key Architecture Diagram (For Whiteboard / Slide Presentation)

```
[Incoming Raw Transaction]
         │
         ▼
[Step 1: Ingestion & Schema Validation]
         │ ── Validate required columns, data types, non-negative values
         ▼
[Step 2: Feature Engineering (No Identifiers)]
         │ ── Derive balance errors, draining flags, intraday cycle
         ▼
[Step 3: Preprocessing Pipeline (ColumnTransformer)]
         │ ── OneHotEncoder (handle_unknown='ignore')
         │ ── RobustScaler (fitted on training data only)
         ▼
[Step 4: Machine Learning Inference (Random Forest Champion)]
         │ ── Generate calibrated probability score [0.00 – 1.00]
         ▼
[Step 5: Decision Threshold Engine (Tuned Threshold = 0.21)]
         │
    ┌────┴─────────────────────────────┬───────────────────────────┐
    ▼                                  ▼                           ▼
[Probability < 0.20]          [0.20 <= Prob < 0.80]        [Probability >= 0.80]
  MINIMAL RISK                  HIGH / MODERATE RISK         CRITICAL RISK
  Auto-Approve                  Step-up 2FA / Review         Block & Dispatch Alert
         │                                  │                           │
         └──────────────────────────────────┼───────────────────────────┘
                                            ▼
                       [Step 6: Explainability Layer (SHAP)]
                       Feature attribution & waterfall reason codes
```

---

## 3. High-Impact Capstone Presentation Tips

1. **Start with the Dollar Impact:** Begin your opening slide by stating: *"Our model successfully shields $235.4M in fraudulent transfers on the benchmark test set, delivering over $188M in net loss mitigation."* This immediately anchors your project in real business value.
2. **Emphasize Methodological Cleanliness:** Mention that your preprocessing pipeline was fitted strictly on training data, SMOTE was quarantined inside cross-validation folds, and threshold calibration was performed on out-of-fold training data.
3. **Live Streamlit Demo:** When demonstrating the dashboard, show:
   - Scenario A (Drained Account Transfer) triggering **100% Critical Risk**.
   - Scenario B (Legitimate Merchant Payment) triggering **0% Minimal Risk**.
   - The Batch CSV upload processing 5 rows and generating an exported CSV report in real-time.

