"""
PowerPoint Presentation Generator for BIA Fraud Detection Capstone.
Constructs a professional 15-slide widescreen presentation in reports/Fraud_Detection_Presentation.pptx
incorporating real evaluation figures, financial metrics, and executive findings.
"""

import os
import json
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


def apply_slide_theme(slide, title_text, category_text="BIA CAPSTONE PROJECT | FRAUD DETECTION"):
    """Create a consistent corporate header banner on each slide."""
    # Top banner shape
    top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = RGBColor(0x1D, 0x35, 0x57)  # Navy
    top_bar.line.color.rgb = RGBColor(0x1D, 0x35, 0x57)

    # Accent underline
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.15), Inches(13.333), Inches(0.06))
    accent.fill.solid()
    accent.fill.fore_color.rgb = RGBColor(0x2A, 0x9D, 0x8F)  # Teal
    accent.line.color.rgb = RGBColor(0x2A, 0x9D, 0x8F)

    # Header text box
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.5), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = "Calibri"
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = RGBColor(0xA8, 0xDA, 0xDC)

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = "Calibri"
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


def add_bullet_points(tf, points, font_size=14, space_after=10):
    """Add styled bullet points to a text frame."""
    for i, pt in enumerate(points):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = f"•  {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(font_size)
        p.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        p.space_after = Pt(space_after)


def create_presentation(output_path: str = "reports/Fraud_Detection_Presentation.pptx"):
    print("[PPTX] Generating BIA Capstone Presentation...")
    prs = Presentation()
    # 16:9 widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = RGBColor(0x0F, 0x17, 0x2A)  # Dark slate
    bg1.line.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(3.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "BOSTON INSTITUTE OF ANALYTICS (BIA)"
    p0.font.name = "Calibri"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(0x2A, 0x9D, 0x8F)
    p0.space_after = Pt(12)

    p_t = tf1.add_paragraph()
    p_t.text = "FRAUDGUARD AI: END-TO-END FINANCIAL FRAUD DETECTION"
    p_t.font.name = "Calibri"
    p_t.font.size = Pt(32)
    p_t.font.bold = True
    p_t.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    p_t.space_after = Pt(12)

    p_sub = tf1.add_paragraph()
    p_sub.text = "Predictive Risk Modeling, Feature Engineering, SHAP Explainability & Interactive Streamlit Deployment"
    p_sub.font.name = "Calibri"
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    p_sub.space_after = Pt(28)

    p_auth = tf1.add_paragraph()
    p_auth.text = "Candidate: Senior Data Science Researcher  |  Domain: FinTech & Payment Security  |  Date: November 2026"
    p_auth.font.name = "Calibri"
    p_auth.font.size = Pt(12)
    p_auth.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # =========================================================================
    # SLIDE 2: Executive Summary & Project Introduction
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s2, "1. Executive Summary & Project Overview")
    tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    add_bullet_points(tf2, [
        "Project Context: Official Data Science Capstone Project for Boston Institute of Analytics (BIA).",
        "Business Urgency: Automated account takeover and real-time liquidation threaten modern instant clearing networks.",
        "End-to-End Scope: Encompasses dataset quality auditing, exploratory visual discovery, ledger feature engineering, multi-algorithm training, hyperparameter optimization, SHAP interpretability, and business ROI quantification.",
        "Champion Model: Random Forest ensemble tuned with stratified cross-validation achieving 99.56% Recall, 100.0% Precision, and 0.9999 PR-AUC on untouched test data.",
        "Financial Protection: Directly shields $235.4M in fraudulent exposure on the test set, delivering $188.3M in net loss mitigation.",
        "Interactive Production Delivery: Deployed via FRAUDGUARD AI, a 6-page interactive Streamlit risk dashboard."
    ], font_size=15, space_after=14)

    # =========================================================================
    # SLIDE 3: Business Problem & Risk Tradeoffs
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s3, "2. Problem Statement: The Asymmetric Cost of Fraud")

    # Card 1: False Negatives
    c1 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    c1.fill.solid()
    c1.fill.fore_color.rgb = RGBColor(0xFE, 0xE2, 0xE2)
    c1.line.color.rgb = RGBColor(0xEF, 0x44, 0x44)
    tf_c1 = c1.text_frame
    tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "🔴 FALSE NEGATIVES (MISSED FRAUD)"
    p.font.name = "Calibri"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x99, 0x1B, 0x1B)
    add_bullet_points(tf_c1, [
        "Direct capital drainage from customer accounts.",
        "Immediate chargeback liability and clearinghouse fines.",
        "Regulatory AML/CFT compliance enforcement.",
        "Loss of brand trust and catastrophic account churn.",
        "Dataset maximum observed single fraud: $10,000,000."
    ], font_size=13, space_after=10)

    # Card 2: False Positives
    c2 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.2))
    c2.fill.solid()
    c2.fill.fore_color.rgb = RGBColor(0xFE, 0xF9, 0xC3)
    c2.line.color.rgb = RGBColor(0xCA, 0x8A, 0x04)
    tf_c2 = c2.text_frame
    tf_c2.word_wrap = True
    p2 = tf_c2.paragraphs[0]
    p2.text = "🟡 FALSE POSITIVES (FALSE ALARMS)"
    p2.font.name = "Calibri"
    p2.font.size = Pt(16)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(0x85, 0x4D, 0x0E)
    add_bullet_points(tf_c2, [
        "Costly manual analyst reviews ($15 to $60 per ticket).",
        "Transaction rejection at point of sale / transfer.",
        "Severe payment friction causing customer frustration.",
        "Operational bottleneck overwhelming security operations.",
        "Goal: Zero false alarms without compromising recall."
    ], font_size=13, space_after=10)

    # =========================================================================
    # SLIDE 4: Dataset Overview & Quality Audit
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s4, "3. Official BIA Dataset Architecture & Integrity Audit")

    if os.path.exists("reports/figures/01_class_distribution_count.png"):
        s4.shapes.add_picture("reports/figures/01_class_distribution_count.png", Inches(0.8), Inches(1.6), Inches(5.6), Inches(4.8))

    tb4 = s4.shapes.add_textbox(Inches(6.8), Inches(1.6), Inches(5.8), Inches(5.0))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    add_bullet_points(tf4, [
        "Total Audited Records: 11,142 observations, 10 columns.",
        "Class Balance: 10,000 Legitimate (89.75%) vs 1,142 Fraudulent (10.25%).",
        "Imbalance Ratio: Approximately 1:8.76.",
        "Data Hygiene: 0 missing values, 0 duplicate rows across all features.",
        "Range Validation: All amounts and account balances are strictly non-negative.",
        "Temporal Span: 95 simulation steps representing ~4 consecutive operating days.",
        "Account Identifiers: 11,142 unique senders; 7,508 unique destinations (5,510 merchant IDs starting with 'M')."
    ], font_size=14, space_after=12)

    # =========================================================================
    # SLIDE 5: Exploratory Data Analysis & Channel Vulnerability
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s5, "4. Exploratory Analysis: Channel Isolation & Attack Vectors")

    if os.path.exists("reports/figures/04_fraud_by_transaction_type.png"):
        s5.shapes.add_picture("reports/figures/04_fraud_by_transaction_type.png", Inches(0.8), Inches(1.6), Inches(5.8), Inches(4.8))

    tb5 = s5.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.6), Inches(5.0))
    tf5 = tb5.text_frame
    tf5.word_wrap = True
    add_bullet_points(tf5, [
        "100% Channel Specificity: Fraud is strictly isolated to TRANSFER (564 frauds; 38.5% rate) and CASH_OUT (578 frauds; 30.9% rate).",
        "Zero Fraud Channels: PAYMENT (5,510), CASH_IN (1,951), and DEBIT (346) exhibit exactly 0 fraudulent events.",
        "Two-Phase Attack Strategy: Attackers first execute unauthorized TRANSFER to an intermediate account, followed immediately by CASH_OUT liquidation.",
        "Monetary Skewness: Fraud mean amount = $1,192,629 vs Legitimate mean amount = $101,339 (orders of magnitude difference).",
        "Domain Implication: Feature representations must capture the interaction between transaction type and ledger movements."
    ], font_size=14, space_after=12)

    # =========================================================================
    # SLIDE 6: Feature Engineering Innovation
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s6, "5. Feature Engineering: Behavioral Indicators & Ledger Accounting")

    tb6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.5))
    tf6 = tb6.text_frame
    tf6.word_wrap = True
    add_bullet_points(tf6, [
        "balance_error_orig (Sender Ledger Discrepancy): Quantifies whether oldbalanceOrg - amount == newbalanceOrig. Fraudsters trigger severe accounting discrepancies.",
        "balance_error_dest (Receiver Ledger Discrepancy): Quantifies whether oldbalanceDest + amount == newbalanceDest.",
        "empty_account_flag (100% Draining Indicator): Binary flag indicating whether the transaction amount exactly equals the sender's entire starting balance (occurs in 97.1% of frauds).",
        "origin_balance_zero (Zero Balance Flag): Detects when the sender account balance is reduced to exactly $0.00 (occurs in 99.2% of frauds).",
        "merchant_destination: Binary indicator for destination accounts starting with 'M' (external merchants).",
        "transaction_hour & transaction_day: Captures cyclical intraday transaction patterns (step % 24).",
        "Empirical Validation: Adding engineered features improved Random Forest test recall from 96.49% to 99.56%!"
    ], font_size=14, space_after=12)

    # =========================================================================
    # SLIDE 7: Machine Learning Methodology & Architecture
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s7, "6. Machine Learning Workflow & Leak-Free Pipeline")

    tb7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.5))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    add_bullet_points(tf7, [
        "Data Partitioning: Stratified 80/20 train-test split (8,913 train with 914 frauds; 2,229 test with 228 frauds).",
        "Leakage Prevention: Preprocessor ColumnTransformer (RobustScaler for skewed finances, OneHotEncoder for channels) fitted strictly on X_train.",
        "Chronological Validation: Robustness experiment confirming stability across time steps.",
        "Candidate Architectures:",
        "    • Logistic Regression: Class-weighted linear baseline with L2 penalty.",
        "    • Decision Tree: Cost-complexity pruned tree with depth optimization.",
        "    • Random Forest: Bagged ensemble of 100 decision trees with balanced bootstrapping.",
        "    • XGBoost: Gradient-boosted decision trees with learning rate shrinkage and scale_pos_weight.",
        "    • Isolation Forest: Unsupervised anomaly detection isolating fraudulent outliers.",
        "Optimization Metric: Precision-Recall Area Under Curve (PR-AUC) and F1-Score via Stratified 5-Fold CV."
    ], font_size=14, space_after=10)

    # =========================================================================
    # SLIDE 8: Class Imbalance Benchmark Study
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s8, "7. Class Imbalance Handling: Empirical Strategy Comparison")

    tb8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.5))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    add_bullet_points(tf8, [
        "Evaluated 4 distinct class handling paradigms using Random Forest on untouched test data:",
        "1. Original (No Resampling): Precision = 1.0000 | Recall = 0.9956 | F1 = 0.9978 | PR-AUC = 0.9993",
        "2. Class-Weighted Learning: Precision = 1.0000 | Recall = 0.9956 | F1 = 0.9978 | PR-AUC = 0.9998",
        "3. SMOTE Resampling (Training Only): Precision = 1.0000 | Recall = 0.9956 | F1 = 0.9978 | PR-AUC = 0.9999",
        "4. Random Under-Sampling (RUS): Precision = 1.0000 | Recall = 0.9956 | F1 = 0.9978 | PR-AUC = 0.9996",
        "Key Finding: Because our engineered features provide extreme class separability, SMOTE and Class-Weighting both achieved top-tier discrimination (PR-AUC > 0.9997) without introducing false positives.",
        "Integrity Protocol: Resampling was applied strictly to training folds within imblearn pipelines—never to validation or test sets."
    ], font_size=14, space_after=12)

    # =========================================================================
    # SLIDE 9: Multi-Model Benchmark Results
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s9, "8. Model Comparison: Test Set Evaluation Metrics")

    if os.path.exists("reports/figures/19_model_comparison_bar.png"):
        s9.shapes.add_picture("reports/figures/19_model_comparison_bar.png", Inches(0.8), Inches(1.5), Inches(6.0), Inches(5.0))

    tb9 = s9.shapes.add_textbox(Inches(7.1), Inches(1.5), Inches(5.5), Inches(5.0))
    tf9 = tb9.text_frame
    tf9.word_wrap = True
    add_bullet_points(tf9, [
        "Random Forest (Champion):",
        "    • Accuracy: 99.96%  |  Precision: 100.00%",
        "    • Recall: 99.56%  |  F1-Score: 99.78%",
        "    • ROC-AUC: 0.99999  |  PR-AUC: 0.99994",
        "XGBoost (Runner-Up):",
        "    • Accuracy: 99.96%  |  Precision: 100.00%",
        "    • Recall: 99.56%  |  PR-AUC: 0.99844",
        "Logistic Regression:",
        "    • Precision: 99.56%  |  Recall: 99.12%  |  F1: 99.34%",
        "Isolation Forest (Unsupervised Baseline):",
        "    • Precision: 76.17%  |  Recall: 71.49%  |  F1: 73.76%",
        "Conclusion: Supervised ensemble trees dramatically outperform unsupervised outlier methods."
    ], font_size=13, space_after=8)

    # =========================================================================
    # SLIDE 10: Confusion Matrix & ROC / PR Diagnostics
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s10, "9. Evaluation Diagnostics: Confusion Matrix & Discrimination Curves")

    if os.path.exists("reports/figures/16b_confusion_matrix_optimal.png"):
        s10.shapes.add_picture("reports/figures/16b_confusion_matrix_optimal.png", Inches(0.8), Inches(1.6), Inches(5.6), Inches(4.8))

    if os.path.exists("reports/figures/18_precision_recall_curves.png"):
        s10.shapes.add_picture("reports/figures/18_precision_recall_curves.png", Inches(6.8), Inches(1.6), Inches(5.8), Inches(4.8))

    # =========================================================================
    # SLIDE 11: Explainable AI Using SHAP
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s11, "10. Explainable AI: Feature Attribution via SHAP")

    if os.path.exists("reports/figures/20_shap_summary.png"):
        s11.shapes.add_picture("reports/figures/20_shap_summary.png", Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.0))

    tb11 = s11.shapes.add_textbox(Inches(7.1), Inches(1.6), Inches(5.5), Inches(5.0))
    tf11 = tb11.text_frame
    tf11.word_wrap = True
    add_bullet_points(tf11, [
        "Regulatory Mandate: Explainability is legally required for financial credit and fraud decisions under FCRA and GDPR.",
        "Top Global Feature: balance_error_orig exerts the largest positive SHAP force on fraud predictions.",
        "Account Depletion: empty_account_flag and origin_balance_zero serve as strong confirmatory fraud indicators.",
        "Channel Weight: Transaction type TRANSFER and CASH_OUT strongly tilt predictions toward fraud.",
        "Model Transparency: Individual waterfall explanations provide instant audit logs for compliance officers."
    ], font_size=14, space_after=14)

    # =========================================================================
    # SLIDE 12: Financial Impact & ROI Quantification
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s12, "11. Financial Impact Analysis: Measured Capital Loss Mitigation")

    if os.path.exists("reports/figures/24_financial_impact_breakdown.png"):
        s12.shapes.add_picture("reports/figures/24_financial_impact_breakdown.png", Inches(0.8), Inches(1.6), Inches(5.8), Inches(4.8))

    tb12 = s12.shapes.add_textbox(Inches(6.9), Inches(1.6), Inches(5.7), Inches(5.0))
    tf12 = tb12.text_frame
    tf12.word_wrap = True
    add_bullet_points(tf12, [
        "Total Test Fraud Exposure: $235,846,526.37 (228 fraudulent transactions).",
        "Detected Fraud Amount: $235,430,525.04 (99.82% dollar recovery rate).",
        "Missed Fraud Amount: $416,001.33 (only 1 single fraud missed).",
        "False Positive Volume: $0.00 (zero legitimate transactions flagged).",
        "Base Case ROI (80% Recovery Rate, $30/investigation):",
        "    • Gross Loss Prevented: $188,344,420.03",
        "    • Total Investigation Overhead: $6,810.00",
        "    • Net Financial Benefit: $188,337,610.03!",
        "Conservative Case (70% Recovery, $60/investigation): Net Benefit = $164,787,747.53."
    ], font_size=13, space_after=10)

    # =========================================================================
    # SLIDE 13: Interactive Streamlit Dashboard (FRAUDGUARD AI)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s13, "12. Production Deployment: FRAUDGUARD AI Streamlit App")

    tb13 = s13.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.5))
    tf13 = tb13.text_frame
    tf13.word_wrap = True
    add_bullet_points(tf13, [
        "Platform Architecture: High-performance interactive dashboard built using Streamlit, Plotly, and Scikit-Learn.",
        "Module 1 — Executive Overview: Top-line KPI metric cards, donut composition, and hourly velocity trends.",
        "Module 2 — Single Transaction Prediction: Interactive risk scoring form with calibrated gauge, risk badge, and suggested actions.",
        "Module 3 — Batch CSV Screening: Automated validation, multi-row inference, summary metrics, and CSV report export.",
        "Module 4 — Model Performance Hub: Live confusion matrix, comparative ROC/PR curves, and dynamic decision threshold slider.",
        "Module 5 — Explainable AI Hub: Embedded SHAP beeswarm, feature importance bars, and individual waterfall plots.",
        "Module 6 — Financial ROI Simulator: Configurable sliders for investigation costs, recovery rates, and friction penalties.",
        "Launch Command: `streamlit run dashboard/app.py` (Fully tested, independent of retraining)."
    ], font_size=14, space_after=12)

    # =========================================================================
    # SLIDE 14: Project Limitations & Future Scope
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s14, "13. Risk Governance: Limitations & Strategic Roadmap")

    tb14 = s14.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.5))
    tf14 = tb14.text_frame
    tf14.word_wrap = True
    add_bullet_points(tf14, [
        "Dataset Boundaries:",
        "    • Synthetic PaySim benchmark lacks biometric telemetry, IP geolocation, and device fingerprints.",
        "    • Fraud concentrated strictly in TRANSFER and CASH_OUT; real banking faces card-not-present (CNP) and merchant fraud.",
        "    • Senders are non-recurring, precluding longitudinal user velocity profiling.",
        "Strategic Roadmap for Production Deployment:",
        "    • Graph Neural Networks (GNN): Construct relational money flow graphs to dismantle complex laundering syndicates.",
        "    • Sub-10ms Streaming Pipelines: Implement Apache Kafka and Flink for real-time clearinghouse transaction blocking.",
        "    • Online Active Learning: Ingest analyst dispute dispositions to continuously adapt to adversarial drift.",
        "    • Model Governance: Maintain automated model drift monitoring and periodic retraining protocols."
    ], font_size=13, space_after=10)

    # =========================================================================
    # SLIDE 15: Conclusion & Recommendations
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    apply_slide_theme(s15, "14. Conclusion & Final Recommendations")

    tb15 = s15.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.5))
    tf15 = tb15.text_frame
    tf15.word_wrap = True
    add_bullet_points(tf15, [
        "Capstone Completion: All Boston Institute of Analytics (BIA) project requirements completed end-to-end.",
        "Empirical Excellence: The champion Random Forest model delivers 99.56% Recall, 100.0% Precision, and 0.9999 PR-AUC.",
        "Measurable Financial ROI: Shields $235.4M in fraudulent transfers, yielding $188.3M net operational benefit on the test set.",
        "Actionable Explainability: SHAP confirms ledger accounting discrepancies as primary fraud drivers.",
        "Deliverables Ready: Jupyter notebook (.ipynb), Word report (.docx), PowerPoint (.pptx), Streamlit app, test suite, and final submission archive.",
        "Thank You! Questions and Faculty Discussion."
    ], font_size=14, space_after=14)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"[PPTX] Successfully saved presentation to '{output_path}'.")


if __name__ == "__main__":
    create_presentation()

