# 🛡️ FRAUDGUARD AI: 2-MINUTE EXECUTIVE PITCH SHEET
## Boston Institute of Analytics (BIA) — Capstone Presentation
**Target Audience:** BIA Examination Panel, Lead Data Scientist, Head of Fraud Risk  
**Delivery Time:** 120 Seconds (2 Minutes)  

---

### [00:00 – 00:30] The Hook: The Real-World Crisis of Instant Payment Fraud
> *"Respected panel members and faculty, instant clearing networks have revolutionized modern commerce, but they have also handed fraudsters an unprecedented weapon: the ability to execute unauthorized account takeovers and liquidate life savings within seconds.*
> 
> *In our audited financial dataset of over 11,000 transactions, individual fraudulent attacks reached up to **$10,000,000**. Traditional rule-based banking firewalls are too rigid—they either let multi-million dollar frauds slip through or generate thousands of false alarms that alienate good customers.*
> 
> *My capstone project introduces **FRAUDGUARD AI**—an end-to-end Machine Learning intelligence system engineered to solve this asymmetric risk challenge."*

---

### [00:30 – 01:00] The Technical Innovation: Domain Feature Engineering
> *"Rather than relying blindly on black-box algorithms, our breakthrough came from rigorous domain feature engineering:*
> 1. *We discovered that **100% of frauds** occurred exclusively in `TRANSFER` and `CASH_OUT` channels.*
> 2. *In **99.2%** of attacks, fraudsters completely liquidated sender accounts down to **$0.00**.*
> 3. *We engineered mathematical ledger indicators—specifically `balance_error_orig`—which flags discrepancies where the sender's account accounting equation fails to reconcile.*
> 
> *These domain features elevated our model's test-set recall from **96.49% to 99.56%**, proving that financial behavioral context is paramount."*

---

### [01:00 – 01:30] The Results: Enterprise-Grade Discrimination
> *"When evaluated on an untouched 20% test partition of 2,229 transactions:
> - Our champion **Random Forest model** detected **227 out of 228 fraudulent transactions**, achieving a **99.56% Recall**.
> - Remarkably, it achieved **100.00% Precision** with **zero false positives** on legitimate customers.
> - Its Precision-Recall AUC reached **0.99994**, decisively outperforming gradient boosted trees and unsupervised isolation methods.
> 
> *Using SHAP (SHapley Additive exPlanations), we established complete regulatory transparency, proving that ledger errors and balance draining flags are the primary drivers of model decisions."*

---

### [01:30 – 02:00] The Bottom-Line ROI & Live Prototype
> *"From a financial risk perspective:
> - On the test set alone, our system successfully intercepted **$235.4 Million** in fraudulent capital—a **99.82% dollar recovery rate**.
> - Factoring in enterprise investigation overheads, the system delivers an estimated **$188.3 Million net financial benefit** under standard banking recovery SLAs.
> 
> *To put this intelligence into the hands of fraud analysts, I developed and deployed **FRAUDGUARD AI**—a 6-page interactive Streamlit web dashboard featuring real-time transaction risk scoring, batch CSV screening, dynamic threshold sliders, and automated compliance reporting.*
> 
> *I am now excited to demonstrate the live application for you. Thank you, and I welcome your questions."*

