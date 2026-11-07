"""
FRAUDGUARD AI — Intelligent Financial Fraud Detection System
Streamlit Capstone Dashboard for Boston Institute of Analytics (BIA).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json

from dashboard.components import (
    apply_custom_styles,
    render_metric_card,
    create_fraud_distribution_pie,
    create_type_fraud_bar,
    create_hourly_trend_line,
    create_risk_gauge,
)
from dashboard.utils import (
    load_model_artifacts,
    load_dataset,
    load_evaluation_metrics,
    load_financial_metrics,
    predict_single_transaction,
    predict_batch_transactions,
    generate_sample_csv,
)

# Page configuration
st.set_page_config(
    page_title="FRAUDGUARD AI | Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply CSS
apply_custom_styles()

# Load models and data
@st.cache_resource
def get_artifacts():
    return load_model_artifacts()

@st.cache_data
def get_data():
    return load_dataset()

@st.cache_data
def get_eval_data():
    return load_evaluation_metrics()

@st.cache_data
def get_fin_data():
    return load_financial_metrics()

try:
    artifacts = get_artifacts()
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    model_metadata = artifacts["metadata"]
    default_threshold = artifacts["threshold"]
except Exception as e:
    st.error(f"⚠️ Error loading model artifacts: {e}. Please ensure models are trained via `run_project.py`.")
    st.stop()

# Sidebar Navigation
st.sidebar.markdown("## 🛡️ **FRAUDGUARD AI**")
st.sidebar.markdown(
    "<small style='color: #64748b;'>Enterprise Risk Intelligence Platform<br>Boston Institute of Analytics</small>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

navigation = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard Overview",
        "⚡ Single Transaction Prediction",
        "📂 Batch Transaction Prediction",
        "📈 Model Performance",
        "🧠 Explainable AI (SHAP)",
        "💰 Financial Impact Analysis",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** `{model_metadata.get('model_name', 'Random Forest')}`")
st.sidebar.markdown(f"**Decision Threshold:** `{default_threshold:.2f}`")
st.sidebar.markdown(
    "<div style='font-size: 0.75rem; color: #94a3b8; margin-top: 20px;'>"
    "CapStone Submission: BIA Data Science<br>© 2026 FraudGuard AI"
    "</div>",
    unsafe_allow_html=True,
)

# ==============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ==============================================================================
if navigation == "📊 Dashboard Overview":
    st.markdown('<div class="brand-title">FRAUDGUARD AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Executive Risk Overview & Historical Transaction Analytics</div>',
        unsafe_allow_html=True,
    )

    df_raw = get_data()
    eval_metrics = get_eval_data()
    opt_metrics = eval_metrics.get("optimal_threshold", {})

    # Top KPI Metrics Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Total Audited Records", f"{len(df_raw):,}", "Historical transactions", "#1D3557")
    with c2:
        render_metric_card("Fraud Incidents", f"{int(df_raw['isFraud'].sum()):,}", "Confirmed fraudulent cases", "#E63946")
    with c3:
        render_metric_card("Legitimate Volume", f"{int((df_raw['isFraud'] == 0).sum()):,}", "Normal customer operations", "#2A9D8F")
    with c4:
        rec_val = opt_metrics.get("recall", 0.9956)
        render_metric_card("Model Test Recall", f"{rec_val:.1%}", "227 of 228 frauds caught", "#E76F51")
    with c5:
        prec_val = opt_metrics.get("precision", 1.0)
        render_metric_card("Model Test Precision", f"{prec_val:.1%}", "Zero false alarms", "#457B9D")

    st.markdown("---")

    # Row 1: Donut & Grouped Bar
    col_chart1, col_chart2 = st.columns([1, 1.2])
    with col_chart1:
        st.plotly_chart(create_fraud_distribution_pie(df_raw), use_container_width=True)
    with col_chart2:
        st.plotly_chart(create_type_fraud_bar(df_raw), use_container_width=True)

    # Row 2: Hourly Velocity & Summary Insights
    col_chart3, col_insight = st.columns([1.3, 1])
    with col_chart3:
        st.plotly_chart(create_hourly_trend_line(df_raw), use_container_width=True)
    with col_insight:
        st.markdown("### 📋 Executive Key Findings")
        st.info(
            """
            - **Target Class Distribution**: 10.25% fraud (1,142) vs 89.75% legitimate (10,000).
            - **Attack Channel Isolation**: 100% of frauds are restricted to **TRANSFER** and **CASH_OUT** channels.
            - **Account Draining Pattern**: In **99.2%** of frauds, the sender account is drained to zero ($0.00).
            - **Monetary Disparity**: Average fraud amount is **$1.19M** compared to **$101K** for legitimate transfers.
            - **Model Capability**: Random Forest champion achieves **99.56% Recall** and **100.0% Precision** on test data.
            """
        )

    st.markdown(
        '<div class="disclaimer-box"><b>Notice:</b> Historical metrics represent the Boston Institute of Analytics (BIA) official benchmark dataset. Live predictions are executed in real time via the Single/Batch inference sections.</div>',
        unsafe_allow_html=True,
    )

# ==============================================================================
# PAGE 2: SINGLE TRANSACTION PREDICTION
# ==============================================================================
elif navigation == "⚡ Single Transaction Prediction":
    st.markdown('<div class="brand-title">Single Transaction Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Simulate individual transaction telemetry and inspect real-time ML inference</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### ⚡ Quick Load Benchmark Scenarios")
    s_col1, s_col2, s_col3 = st.columns(3)

    if "preset_type" not in st.session_state:
        st.session_state["preset_type"] = "TRANSFER"
        st.session_state["preset_amount"] = 181.0
        st.session_state["preset_old_org"] = 181.0
        st.session_state["preset_new_org"] = 0.0
        st.session_state["preset_old_dst"] = 0.0
        st.session_state["preset_new_dst"] = 0.0
        st.session_state["preset_step"] = 1

    with s_col1:
        if st.button("🔴 Scenario A: Account Draining Transfer (Fraud)"):
            st.session_state["preset_type"] = "TRANSFER"
            st.session_state["preset_amount"] = 350000.0
            st.session_state["preset_old_org"] = 350000.0
            st.session_state["preset_new_org"] = 0.0
            st.session_state["preset_old_dst"] = 0.0
            st.session_state["preset_new_dst"] = 0.0
            st.session_state["preset_step"] = 14

    with s_col2:
        if st.button("🟢 Scenario B: Legitimate Merchant Payment"):
            st.session_state["preset_type"] = "PAYMENT"
            st.session_state["preset_amount"] = 1250.0
            st.session_state["preset_old_org"] = 15000.0
            st.session_state["preset_new_org"] = 13750.0
            st.session_state["preset_old_dst"] = 0.0
            st.session_state["preset_new_dst"] = 0.0
            st.session_state["preset_step"] = 10

    with s_col3:
        if st.button("🔴 Scenario C: Fast Cash-Out Draining (Fraud)"):
            st.session_state["preset_type"] = "CASH_OUT"
            st.session_state["preset_amount"] = 850000.0
            st.session_state["preset_old_org"] = 850000.0
            st.session_state["preset_new_org"] = 0.0
            st.session_state["preset_old_dst"] = 25000.0
            st.session_state["preset_new_dst"] = 875000.0
            st.session_state["preset_step"] = 22

    st.markdown("---")

    with st.form("single_prediction_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            trans_type = st.selectbox(
                "Transaction Type",
                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                index=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"].index(st.session_state["preset_type"]),
            )
            step = st.number_input("Step (Hour of Simulation)", min_value=1, max_value=744, value=st.session_state["preset_step"])
        with f2:
            amount = st.number_input("Transaction Amount ($ USD)", min_value=0.01, max_value=10000000.0, value=float(st.session_state["preset_amount"]), step=100.0)
            old_org = st.number_input("Sender Initial Balance ($)", min_value=0.0, value=float(st.session_state["preset_old_org"]), step=100.0)
        with f3:
            new_org = st.number_input("Sender New Balance ($)", min_value=0.0, value=float(st.session_state["preset_new_org"]), step=100.0)
            old_dst = st.number_input("Receiver Initial Balance ($)", min_value=0.0, value=float(st.session_state["preset_old_dst"]), step=100.0)
            new_dst = st.number_input("Receiver New Balance ($)", min_value=0.0, value=float(st.session_state["preset_new_dst"]), step=100.0)

        predict_btn = st.form_submit_button("🔍 Evaluate Transaction Risk")

    if predict_btn:
        result = predict_single_transaction(
            model=model,
            preprocessor=preprocessor,
            step=step,
            trans_type=trans_type,
            amount=amount,
            oldbalanceOrg=old_org,
            newbalanceOrig=new_org,
            oldbalanceDest=old_dst,
            newbalanceDest=new_dst,
            threshold=default_threshold,
        )

        st.markdown("### 🎯 Real-Time ML Decision Output")
        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            st.plotly_chart(create_risk_gauge(result["probability"]), use_container_width=True)

            if result["is_fraud"]:
                st.markdown(
                    f"""
                    <div class="risk-banner risk-critical">
                        🚨 TRANSACTION CLASSIFIED AS FRAUDULENT<br>
                        <span style="font-size: 0.9rem; font-weight: 400;">
                        Risk Tier: <b>{result['risk_category']}</b> | Confidence: <b>{result['probability']:.1%}</b>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-banner risk-minimal">
                        ✅ TRANSACTION CLASSIFIED AS LEGITIMATE<br>
                        <span style="font-size: 0.9rem; font-weight: 400;">
                        Risk Tier: <b>{result['risk_category']}</b> | Confidence: <b>{(1 - result['probability']):.1%} Safe</b>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with res_col2:
            st.markdown("#### 🛡️ Decision Support & Suggested Action")
            st.warning(f"**Recommended Action:** {result['suggested_action']}")

            st.markdown("#### 🔍 Explainability & Domain Rule Triggers")
            for factor in result["contributing_factors"]:
                st.markdown(f"- {factor}")

            st.markdown("#### 📐 Auto-Engineered Financial Metrics")
            eng = result["engineered_summary"]
            em1, em2 = st.columns(2)
            with em1:
                st.metric("Sender Balance Error", f"${eng['balance_error_orig']:,.2f}")
                st.metric("Net Sender Deduction", f"${eng['origin_balance_diff']:,.2f}")
            with em2:
                st.metric("Receiver Balance Error", f"${eng['balance_error_dest']:,.2f}")
                st.metric("Net Receiver Addition", f"${eng['dest_balance_diff']:,.2f}")

    st.markdown(
        '<div class="disclaimer-box"><b>Academic Prototype Disclaimer:</b> FraudGuard AI is an academic decision-support prototype developed for BIA capstone evaluation and does not constitute an automated banking clearing switch.</div>',
        unsafe_allow_html=True,
    )

# ==============================================================================
# PAGE 3: BATCH TRANSACTION PREDICTION
# ==============================================================================
elif navigation == "📂 Batch Transaction Prediction":
    st.markdown('<div class="brand-title">Batch Transaction Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Upload multi-record CSV files for automated enterprise screening</div>',
        unsafe_allow_html=True,
    )

    col_up, col_samp = st.columns([2, 1])
    with col_samp:
        st.markdown("##### 📥 Need a Test CSV?")
        sample_csv = generate_sample_csv()
        st.download_button(
            label="Download Benchmark Sample CSV",
            data=sample_csv,
            file_name="sample_fraud_batch.csv",
            mime="text/csv",
        )

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Transaction File (CSV)",
            type=["csv"],
            help="Ensure CSV includes: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest",
        )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully ingested {len(batch_df)} transaction rows.")

            with st.spinner("Processing batch feature engineering and model inference..."):
                results_df = predict_batch_transactions(
                    model=model,
                    preprocessor=preprocessor,
                    batch_df=batch_df,
                    threshold=default_threshold,
                )

            st.markdown("---")
            st.markdown("### 📊 Screening Summary")
            b1, b2, b3, b4 = st.columns(4)

            total_proc = len(results_df)
            flagged_cnt = int(results_df["Predicted_Fraud"].sum())
            flagged_pct = (flagged_cnt / total_proc) * 100 if total_proc > 0 else 0
            flagged_amt = float(results_df[results_df["Predicted_Fraud"] == 1]["amount"].sum())

            with b1:
                render_metric_card("Total Ingested", f"{total_proc:,}", "Processed rows", "#1D3557")
            with b2:
                render_metric_card("Flagged Frauds", f"{flagged_cnt:,}", f"{flagged_pct:.1f}% flag rate", "#E63946")
            with b3:
                render_metric_card("Approved Safe", f"{total_proc - flagged_cnt:,}", "Low risk score", "#2A9D8F")
            with b4:
                render_metric_card("Flagged Volume", f"${flagged_amt:,.2f}", "Total flagged value", "#F4A261")

            st.markdown("### 📋 Screened Transaction Records")
            st.dataframe(results_df, use_container_width=True)

            csv_out = results_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Export Screened Batch Predictions (CSV)",
                data=csv_out,
                file_name="screened_fraud_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")

# ==============================================================================
# PAGE 4: MODEL PERFORMANCE
# ==============================================================================
elif navigation == "📈 Model Performance":
    st.markdown('<div class="brand-title">Model Evaluation & Benchmark Diagnostics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Empirical test-set results and multi-model benchmark comparisons</div>',
        unsafe_allow_html=True,
    )

    eval_data = get_eval_data()
    all_models = eval_data.get("all_models_summary", [])
    if all_models:
        df_comp = pd.DataFrame(all_models)
        st.markdown("### 🏆 Comprehensive Model Comparison Table (Test Set)")
        st.dataframe(
            df_comp.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}",
            }).highlight_max(subset=["PR-AUC", "F1-Score", "Recall"], color="#dcfce7"),
            use_container_width=True,
        )

    st.markdown("---")

    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        st.markdown("#### 🔲 Confusion Matrix (Optimal Decision Threshold)")
        if os.path.exists("reports/figures/16b_confusion_matrix_optimal.png"):
            st.image("reports/figures/16b_confusion_matrix_optimal.png", use_container_width=True)
        else:
            st.info("Confusion matrix figure available in reports/figures.")

    with col_eval2:
        st.markdown("#### ⚖️ Decision Threshold Optimization Curve")
        if os.path.exists("reports/figures/16_threshold_analysis.png"):
            st.image("reports/figures/16_threshold_analysis.png", use_container_width=True)
        else:
            st.info("Threshold analysis curve available in reports/figures.")

    st.markdown("---")
    col_eval3, col_eval4 = st.columns(2)
    with col_eval3:
        st.markdown("#### 📈 Comparative ROC Curves")
        if os.path.exists("reports/figures/17_roc_curves.png"):
            st.image("reports/figures/17_roc_curves.png", use_container_width=True)
    with col_eval4:
        st.markdown("#### 🎯 Comparative Precision-Recall Curves")
        if os.path.exists("reports/figures/18_precision_recall_curves.png"):
            st.image("reports/figures/18_precision_recall_curves.png", use_container_width=True)

# ==============================================================================
# PAGE 5: EXPLAINABLE AI
# ==============================================================================
elif navigation == "🧠 Explainable AI (SHAP)":
    st.markdown('<div class="brand-title">Explainable AI & Feature Attribution</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Interpreting model decisions using SHAP (SHapley Additive exPlanations)</div>',
        unsafe_allow_html=True,
    )

    col_shap1, col_shap2 = st.columns(2)
    with col_shap1:
        st.markdown("#### 🐝 Global SHAP Summary Beeswarm Plot")
        if os.path.exists("reports/figures/20_shap_summary.png"):
            st.image("reports/figures/20_shap_summary.png", use_container_width=True)
        st.caption("Shows the directional distribution of feature impact on fraud probability.")

    with col_shap2:
        st.markdown("#### 📊 Global Mean |SHAP| Importance Bar")
        if os.path.exists("reports/figures/21_shap_feature_importance_bar.png"):
            st.image("reports/figures/21_shap_feature_importance_bar.png", use_container_width=True)
        st.caption("Ranks features by average absolute contribution to predictions.")

    st.markdown("---")
    st.markdown("### 🔎 Local Transaction Attribution (Waterfalls)")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("#### 🔴 Fraudulent Transaction Explanation")
        if os.path.exists("reports/figures/22_shap_individual_fraud_waterfall.png"):
            st.image("reports/figures/22_shap_individual_fraud_waterfall.png", use_container_width=True)
        st.caption("Key positive drivers: complete balance depletion (`origin_balance_zero`), high amount, ledger error.")

    with col_w2:
        st.markdown("#### 🟢 Legitimate Transaction Explanation")
        if os.path.exists("reports/figures/23_shap_individual_legit_waterfall.png"):
            st.image("reports/figures/23_shap_individual_legit_waterfall.png", use_container_width=True)
        st.caption("Key negative drivers: balanced accounting (`balance_error_orig = 0`), non-zero remaining balances.")

# ==============================================================================
# PAGE 6: FINANCIAL IMPACT
# ==============================================================================
elif navigation == "💰 Financial Impact Analysis":
    st.markdown('<div class="brand-title">Financial Risk & Loss Mitigation Modeling</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Translating machine learning metrics into measurable enterprise ROI</div>',
        unsafe_allow_html=True,
    )

    fin_data = get_fin_data()
    base_case = fin_data.get("base_case", {})
    meas = base_case.get("measured_metrics", {})
    est = base_case.get("estimated_business_impact", {})

    # Top measured values
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Audited Fraud", f"${meas.get('total_fraud_amount', 235846526):,.0f}", "Measured in test set", "#1D3557")
    with m2:
        render_metric_card("Detected Fraud Volume", f"${meas.get('detected_fraud_amount', 235430525):,.0f}", f"{meas.get('dollar_detection_rate_pct', 99.82)}% caught", "#2A9D8F")
    with m3:
        render_metric_card("Missed Fraud Exposure", f"${meas.get('missed_fraud_amount', 416001):,.0f}", "1 single missed fraud", "#E63946")
    with m4:
        render_metric_card("Net Financial Benefit", f"${est.get('net_financial_benefit_usd', 188337610):,.0f}", "After operational costs", "#457B9D")

    st.markdown("---")

    col_fin1, col_fin2 = st.columns([1.2, 1])
    with col_fin1:
        st.markdown("#### 📊 Loss Breakdown & Scenario Simulations")
        if os.path.exists("reports/figures/24_financial_impact_breakdown.png"):
            st.image("reports/figures/24_financial_impact_breakdown.png", use_container_width=True)
        if os.path.exists("reports/figures/25_financial_scenarios.png"):
            st.image("reports/figures/25_financial_scenarios.png", use_container_width=True)

    with col_fin2:
        st.markdown("#### ⚙️ Interactive ROI Simulator")
        st.markdown("Adjust enterprise cost assumptions to simulate net business impact:")

        sim_cost = st.slider("Investigation Cost per Flag ($)", min_value=5.0, max_value=150.0, value=30.0, step=5.0)
        sim_rec = st.slider("Fraud Recovery / Prevention Rate (%)", min_value=40, max_value=100, value=80, step=5) / 100.0
        sim_fric = st.slider("False Positive Friction Rate (%)", min_value=0.0, max_value=5.0, value=1.5, step=0.5) / 100.0

        det_amt = meas.get("detected_fraud_amount", 235430525.0)
        tp_cnt = meas.get("detected_fraud_count", 227)
        fp_cnt = meas.get("false_positive_count", 0)
        flag_legit = meas.get("flagged_legit_amount", 0.0)

        sim_prevented = det_amt * sim_rec
        sim_op_cost = (tp_cnt + fp_cnt) * sim_cost + flag_legit * sim_fric
        sim_net = sim_prevented - sim_op_cost

        st.metric("Simulated Loss Prevented", f"${sim_prevented:,.2f}")
        st.metric("Simulated Operational Cost", f"${sim_op_cost:,.2f}")
        st.metric("Simulated Net Benefit", f"${sim_net:,.2f}", delta=f"{sim_net - est.get('net_financial_benefit_usd', 0):+,.0f} vs base")

        st.info(
            """
            **Financial Assumptions Notice:**
            Actual recovery rates and investigation costs depend on bank-specific SLAs.
            These figures demonstrate risk modeling under configurable operational parameters.
            """
        )

