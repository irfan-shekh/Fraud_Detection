"""
FRAUDGUARD AI — Intelligent Financial Fraud Detection System
Streamlit Capstone Dashboard for Boston Institute of Analytics (BIA).
Implements Dual-Pipeline Fraud Intelligence:
1. Pre-Transaction Fraud Screening (Authorization Switch)
2. Post-Transaction Fraud Detection (Clearing & Settlement Audit)
"""

import os
import sys

# Ensure repository root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

try:
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
        load_model_comparison,
        load_evaluation_metrics,
        load_financial_metrics,
        load_threshold_analysis,
        load_feature_audit,
        predict_pre_transaction,
        predict_post_transaction,
        predict_batch_transactions,
        generate_sample_csv,
    )
except ImportError:
    from components import (
        apply_custom_styles,
        render_metric_card,
        create_fraud_distribution_pie,
        create_type_fraud_bar,
        create_hourly_trend_line,
        create_risk_gauge,
    )
    from utils import (
        load_model_artifacts,
        load_dataset,
        load_model_comparison,
        load_evaluation_metrics,
        load_financial_metrics,
        load_threshold_analysis,
        load_feature_audit,
        predict_pre_transaction,
        predict_post_transaction,
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


# Cached loaders
@st.cache_resource
def get_model(mode: str = "post"):
    return load_model_artifacts(mode=mode)

@st.cache_data
def get_data():
    return load_dataset()

@st.cache_data
def get_comparison():
    return load_model_comparison()

@st.cache_data
def get_eval(mode: str = "post"):
    return load_evaluation_metrics(mode=mode)

@st.cache_data
def get_financial():
    return load_financial_metrics()


# Load model artifacts
try:
    post_artifacts = get_model(mode="post")
    pre_artifacts = get_model(mode="pre")
except Exception as e:
    st.error(f"⚠️ Error loading model artifacts: {e}. Please ensure models are trained via `python run_project.py`.")
    st.stop()


# Sidebar Navigation
st.sidebar.markdown("## 🛡️ **FRAUDGUARD AI**")
st.sidebar.markdown(
    "<small style='color: #64748b;'>Enterprise Risk Intelligence Platform<br>Boston Institute of Analytics</small>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

navigation = st.sidebar.radio(
    "System Navigation",
    [
        "📊 Dashboard Overview",
        "⚡ Pre-Transaction Fraud Screening",
        "🔍 Post-Transaction Fraud Detection",
        "📂 Batch Transaction Prediction",
        "📈 Model Performance Comparison",
        "🧠 Explainable AI (SHAP)",
        "💰 Financial Impact Analysis",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
st.sidebar.markdown(f"**Pre-Tx Model:** `{pre_artifacts['model_name']}` (Th: `{pre_artifacts['threshold']:.2f}`)")
st.sidebar.markdown(f"**Post-Tx Model:** `{post_artifacts['model_name']}` (Th: `{post_artifacts['threshold']:.2f}`)")
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
        '<div class="brand-subtitle">Executive Risk Overview & Dual-Pipeline Fraud Intelligence</div>',
        unsafe_allow_html=True,
    )

    df_raw = get_data()
    comp_data = get_comparison()
    post_eval = get_eval("post")
    pre_eval = get_eval("pre")

    total_tx = len(df_raw)
    fraud_tx = int((df_raw["isFraud"] == 1).sum())
    legit_tx = int((df_raw["isFraud"] == 0).sum())
    fraud_pct = (fraud_tx / total_tx) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Transactions", f"{total_tx:,}", "Ingested benchmark records")
    with col2:
        render_metric_card("Legitimate Volume", f"{legit_tx:,}", f"{100 - fraud_pct:.2f}% normal volume", "#1D3557")
    with col3:
        render_metric_card("Fraudulent Volume", f"{fraud_tx:,}", f"{fraud_pct:.2f}% class prevalence", "#E63946")
    with col4:
        pre_rec = pre_eval.get("optimal_threshold", {}).get("recall", 0.9956) * 100
        render_metric_card("Pre-Tx Screening Recall", f"{pre_rec:.1f}%", "Interception rate prior to settlement", "#2A9D8F")

    st.markdown("---")

    # Interactive charts row
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        st.plotly_chart(create_fraud_distribution_pie(df_raw), use_container_width=True)
    with ch_col2:
        st.plotly_chart(create_type_fraud_bar(df_raw), use_container_width=True)

    st.plotly_chart(create_hourly_trend_line(df_raw), use_container_width=True)

    st.markdown("### 🔬 Dual-Pipeline Architecture Summary")
    st.info(
        "**Why Two Models?**\n"
        "- **Pre-Transaction Fraud Screening:** Evaluates transaction requests in real time at the authorization gateway using only initiating features (`amount`, `oldbalanceOrg`, `type`, etc.). It catches fraud attempts before funds leave the account.\n"
        "- **Post-Transaction Fraud Detection:** Audits cleared transactions during settlement reconciliation, using post-clearing ledger balances (`newbalanceOrig`, `balance_error_orig`) to catch account liquidations and ledger manipulation."
    )


# ==============================================================================
# PAGE 2: PRE-TRANSACTION FRAUD SCREENING
# ==============================================================================
elif navigation == "⚡ Pre-Transaction Fraud Screening":
    st.markdown('<div class="brand-title">Pre-Transaction Fraud Screening</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Real-Time Authorization Switch Scoring (Strictly Zero Post-Transaction Information)</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-box" style="margin-top:0px; margin-bottom: 20px;">
        ℹ️ <b>Pre-Authorization Mode:</b> This model evaluates incoming payment requests strictly <i>before</i> funds are cleared.
        It does <b>not</b> request or use updated account balances (<code>newbalanceOrig</code> / <code>newbalanceDest</code>),
        preventing target leakage and reflecting genuine payment gateway deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    scenario = st.selectbox(
        "Select Benchmark Test Scenario (or enter custom transaction below):",
        [
            "Custom Transaction Entry",
            "Scenario 1: High-Risk Draining Transfer ($100k requested with $100k balance)",
            "Scenario 2: Normal Merchant Payment ($150 to retailer)",
            "Scenario 3: Normal Peer-to-Peer Transfer ($2,000 with ample funds)",
        ],
    )

    # Defaults
    def_type = "TRANSFER"
    def_step = 1
    def_amount = 100000.0
    def_old_orig = 100000.0
    def_old_dest = 0.0
    def_dest_id = "C553264065"

    if scenario.startswith("Scenario 1"):
        def_type = "TRANSFER"
        def_step = 1
        def_amount = 100000.0
        def_old_orig = 100000.0
        def_old_dest = 0.0
        def_dest_id = "C553264065"
    elif scenario.startswith("Scenario 2"):
        def_type = "PAYMENT"
        def_step = 2
        def_amount = 150.0
        def_old_orig = 4500.0
        def_old_dest = 0.0
        def_dest_id = "M1979787155"
    elif scenario.startswith("Scenario 3"):
        def_type = "TRANSFER"
        def_step = 5
        def_amount = 2000.0
        def_old_orig = 25000.0
        def_old_dest = 12000.0
        def_dest_id = "C20012345"

    with st.form("pre_tx_form"):
        col1, col2 = st.columns(2)
        with col1:
            trans_type = st.selectbox("Transaction Channel", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], index=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"].index(def_type))
            step = st.number_input("Transaction Step / Hour (1–95)", min_value=1, max_value=500, value=int(def_step))
            amount = st.number_input("Requested Transaction Amount ($)", min_value=0.01, max_value=50000000.0, value=float(def_amount), step=100.0)
        with col2:
            oldbalanceOrg = st.number_input("Sender Starting Balance ($)", min_value=0.0, max_value=50000000.0, value=float(def_old_orig), step=100.0)
            oldbalanceDest = st.number_input("Recipient Starting Balance ($)", min_value=0.0, max_value=50000000.0, value=float(def_old_dest), step=100.0)
            nameDest = st.text_input("Recipient Account ID (prefix M for Merchant)", value=def_dest_id)

        threshold_override = st.slider("Pre-Transaction Decision Threshold", min_value=0.05, max_value=0.95, value=float(pre_artifacts["threshold"]), step=0.01)
        submitted = st.form_submit_button("⚡ Evaluate Pre-Transaction Risk")

    if submitted:
        try:
            res = predict_pre_transaction(
                model=pre_artifacts["model"],
                preprocessor=pre_artifacts["preprocessor"],
                step=step,
                trans_type=trans_type,
                amount=amount,
                oldbalanceOrg=oldbalanceOrg,
                oldbalanceDest=oldbalanceDest,
                nameDest=nameDest,
                threshold=threshold_override,
            )

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.plotly_chart(create_risk_gauge(res["probability"]), use_container_width=True)
            with res_col2:
                css_class = "risk-critical" if res["is_fraud"] else "risk-minimal"
                verdict = "🚨 HIGH RISK: SUSPECTED FRAUD DETECTED" if res["is_fraud"] else "✅ LOW RISK: TRANSACTION VERIFIED"
                st.markdown(
                    f"""
                    <div class="risk-banner {css_class}">
                        <div style="font-size: 1.25rem;">{verdict}</div>
                        <div style="margin-top: 5px;">Risk Tier: <b>{res['risk_category']}</b> | Evaluated at Threshold: <b>{res['threshold']:.2f}</b></div>
                        <div style="margin-top: 8px; font-weight: normal;">Suggested Gateway Action: <b>{res['suggested_action']}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### 📋 Pre-Authorization Policy Triggers")
                for alert in res["policy_alerts"]:
                    st.markdown(f"- {alert}")

                st.markdown(
                    "<small style='color: #64748b;'>*Note: Policy triggers indicate pre-execution rule conditions. For statistical feature attributions, refer to the SHAP Explainability tab.</small>",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Validation Error: {e}")


# ==============================================================================
# PAGE 3: POST-TRANSACTION FRAUD DETECTION
# ==============================================================================
elif navigation == "🔍 Post-Transaction Fraud Detection":
    st.markdown('<div class="brand-title">Post-Transaction Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Settlement Audit & Ledger Accounting Verification (Post-Execution)</div>',
        unsafe_allow_html=True,
    )

    scenario_post = st.selectbox(
        "Select Post-Transaction Audit Scenario:",
        [
            "Custom Transaction Entry",
            "Scenario 1: Completed Account Drainage ($181 transferred, balance reduced to exactly $0)",
            "Scenario 2: Accounting Discrepancy (Sender balance reduced by more than amount)",
            "Scenario 3: Normal Cleared Merchant Purchase ($50 payment with expected balance reduction)",
        ],
    )

    def_type = "TRANSFER"
    def_step = 1
    def_amount = 181.0
    def_old_orig = 181.0
    def_new_orig = 0.0
    def_old_dest = 0.0
    def_new_dest = 0.0
    def_dest_id = "C553264065"

    if scenario_post.startswith("Scenario 1"):
        def_type = "TRANSFER"
        def_step = 1
        def_amount = 181.0
        def_old_orig = 181.0
        def_new_orig = 0.0
        def_old_dest = 0.0
        def_new_dest = 0.0
    elif scenario_post.startswith("Scenario 2"):
        def_type = "TRANSFER"
        def_step = 3
        def_amount = 50000.0
        def_old_orig = 80000.0
        def_new_orig = 10000.0  # discrepancy: 80k - 50k should be 30k
        def_old_dest = 5000.0
        def_new_dest = 55000.0
    elif scenario_post.startswith("Scenario 3"):
        def_type = "PAYMENT"
        def_step = 2
        def_amount = 50.0
        def_old_orig = 1000.0
        def_new_orig = 950.0
        def_old_dest = 0.0
        def_new_dest = 0.0
        def_dest_id = "M12345678"

    with st.form("post_tx_form"):
        col1, col2 = st.columns(2)
        with col1:
            p_type = st.selectbox("Transaction Channel", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], index=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"].index(def_type))
            p_step = st.number_input("Transaction Step (Hour)", min_value=1, max_value=500, value=int(def_step))
            p_amount = st.number_input("Executed Amount ($)", min_value=0.01, max_value=50000000.0, value=float(def_amount), step=100.0)
            p_old_orig = st.number_input("Sender Balance BEFORE Execution ($)", min_value=0.0, max_value=50000000.0, value=float(def_old_orig), step=100.0)
            p_new_orig = st.number_input("Sender Balance AFTER Execution ($)", min_value=0.0, max_value=50000000.0, value=float(def_new_orig), step=100.0)
        with col2:
            p_dest_id = st.text_input("Recipient ID", value=def_dest_id)
            p_old_dest = st.number_input("Receiver Balance BEFORE Execution ($)", min_value=0.0, max_value=50000000.0, value=float(def_old_dest), step=100.0)
            p_new_dest = st.number_input("Receiver Balance AFTER Execution ($)", min_value=0.0, max_value=50000000.0, value=float(def_new_dest), step=100.0)
            p_th = st.slider("Post-Transaction Decision Threshold", min_value=0.05, max_value=0.95, value=float(post_artifacts["threshold"]), step=0.01)

        submitted_post = st.form_submit_button("🔍 Run Post-Transaction Audit")

    if submitted_post:
        try:
            res_p = predict_post_transaction(
                model=post_artifacts["model"],
                preprocessor=post_artifacts["preprocessor"],
                step=p_step,
                trans_type=p_type,
                amount=p_amount,
                oldbalanceOrg=p_old_orig,
                newbalanceOrig=p_new_orig,
                oldbalanceDest=p_old_dest,
                newbalanceDest=p_new_dest,
                nameDest=p_dest_id,
                threshold=p_th,
            )

            st.markdown("---")
            rcol1, rcol2 = st.columns([1, 2])
            with rcol1:
                st.plotly_chart(create_risk_gauge(res_p["probability"]), use_container_width=True)
            with rcol2:
                css_class = "risk-critical" if res_p["is_fraud"] else "risk-minimal"
                verdict = "🚨 FRAUD CONFIRMED: LEDGER DISCREPANCY DETECTED" if res_p["is_fraud"] else "✅ TRANSACTION VERIFIED & RECONCILED"
                st.markdown(
                    f"""
                    <div class="risk-banner {css_class}">
                        <div style="font-size: 1.25rem;">{verdict}</div>
                        <div style="margin-top: 5px;">Risk Tier: <b>{res_p['risk_category']}</b> | Evaluated at Threshold: <b>{res_p['threshold']:.2f}</b></div>
                        <div style="margin-top: 8px; font-weight: normal;">Settlement Disposition: <b>{res_p['suggested_action']}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### ⚖️ Ledger Accounting Diagnostics")
                led = res_p["ledger_metrics"]
                lcol1, lcol2 = st.columns(2)
                with lcol1:
                    st.metric("Sender Ledger Imbalance", f"${led['balance_error_orig']:,.2f}")
                with lcol2:
                    st.metric("Receiver Ledger Imbalance", f"${led['balance_error_dest']:,.2f}")

                for alert in res_p["policy_alerts"]:
                    st.markdown(f"- {alert}")

        except Exception as e:
            st.error(f"Validation Error: {e}")


# ==============================================================================
# PAGE 4: BATCH TRANSACTION PREDICTION
# ==============================================================================
elif navigation == "📂 Batch Transaction Prediction":
    st.markdown('<div class="brand-title">Batch Transaction Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Bulk CSV Screening Engine with Strict Input Validation</div>',
        unsafe_allow_html=True,
    )

    batch_mode = st.radio(
        "Select Batch Screening Mode:",
        ["Pre-Transaction Screening (Pre-Authorization)", "Post-Transaction Detection (Clearing Audit)"],
        horizontal=True,
    )
    is_pre = "Pre-Transaction" in batch_mode
    mode_str = "pre" if is_pre else "post"
    active_artifacts = pre_artifacts if is_pre else post_artifacts

    col_dl, col_up = st.columns([1, 2])
    with col_dl:
        st.markdown("#### Download Benchmark CSV")
        sample_csv = generate_sample_csv(mode=mode_str)
        st.download_button(
            label=f"📥 Download {mode_str.capitalize()} Sample CSV",
            data=sample_csv,
            file_name=f"sample_{mode_str}_transactions.csv",
            mime="text/csv",
        )
        st.markdown(
            f"<small style='color: #64748b;'>Expected columns for {mode_str}:<br>"
            + ("<code>step, type, amount, oldbalanceOrg, oldbalanceDest</code>" if is_pre else "<code>step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest</code>")
            + "</small>",
            unsafe_allow_html=True,
        )

    with col_up:
        st.markdown("#### Upload Batch File")
        uploaded_file = st.file_uploader(f"Choose a CSV file ({mode_str}-transaction)", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)
            st.write(f"Uploaded file contains **{len(batch_raw):,}** transactions.")

            scored_df = predict_batch_transactions(
                model=active_artifacts["model"],
                preprocessor=active_artifacts["preprocessor"],
                batch_df=batch_raw,
                mode=mode_str,
                threshold=float(active_artifacts["threshold"]),
            )

            flagged_cnt = int((scored_df["Predicted_Fraud"] == 1).sum())
            legit_cnt = len(scored_df) - flagged_cnt

            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                render_metric_card("Processed Rows", f"{len(scored_df):,}", "Transactions evaluated")
            with bcol2:
                render_metric_card("Flagged Suspicious", f"{flagged_cnt:,}", f"{flagged_cnt / len(scored_df) * 100:.1f}% risk rate", "#E63946")
            with bcol3:
                render_metric_card("Verified Normal", f"{legit_cnt:,}", f"{legit_cnt / len(scored_df) * 100:.1f}% auto-approved", "#2A9D8F")

            st.markdown("### Scored Transaction Output")
            st.dataframe(
                scored_df.style.apply(
                    lambda row: ["background-color: #fee2e2" if row["Predicted_Fraud"] == 1 else "" for _ in row],
                    axis=1,
                ),
                use_container_width=True,
            )

            csv_out = scored_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Scored Batch Results (CSV)",
                data=csv_out,
                file_name=f"fraud_scored_{mode_str}_results.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ Batch Validation Failure: {e}")


# ==============================================================================
# PAGE 5: MODEL PERFORMANCE COMPARISON
# ==============================================================================
elif navigation == "📈 Model Performance Comparison":
    st.markdown('<div class="brand-title">Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Rigorous Benchmark Metrics Across Pre- and Post-Transaction Models</div>',
        unsafe_allow_html=True,
    )

    comp = get_comparison()
    pre_eval = get_eval("pre")
    post_eval = get_eval("post")

    st.markdown("### 🏆 Champion Architecture Side-by-Side")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ⚡ Pre-Transaction Fraud Screening Model")
        st.markdown(f"**Model Architecture:** `{pre_eval.get('champion_model', 'Random Forest')}`")
        opt_pre = pre_eval.get("optimal_threshold", {})
        m_df_pre = pd.DataFrame([{
            "Accuracy": f"{opt_pre.get('accuracy', 0):.4f}",
            "Precision": f"{opt_pre.get('precision', 0):.4f}",
            "Recall": f"{opt_pre.get('recall', 0):.4f}",
            "F1-Score": f"{opt_pre.get('f1_score', 0):.4f}",
            "ROC-AUC": f"{opt_pre.get('roc_auc', 0):.5f}",
            "PR-AUC": f"{opt_pre.get('pr_auc', 0):.5f}",
            "Optimal Threshold": f"{opt_pre.get('threshold', 0):.2f}",
        }])
        st.table(m_df_pre.T.rename(columns={0: "Pre-Tx Metric Value"}))

    with c2:
        st.markdown("#### 🔍 Post-Transaction Fraud Detection Model")
        st.markdown(f"**Model Architecture:** `{post_eval.get('champion_model', 'XGBoost')}`")
        opt_post = post_eval.get("optimal_threshold", {})
        m_df_post = pd.DataFrame([{
            "Accuracy": f"{opt_post.get('accuracy', 0):.4f}",
            "Precision": f"{opt_post.get('precision', 0):.4f}",
            "Recall": f"{opt_post.get('recall', 0):.4f}",
            "F1-Score": f"{opt_post.get('f1_score', 0):.4f}",
            "ROC-AUC": f"{opt_post.get('roc_auc', 0):.5f}",
            "PR-AUC": f"{opt_post.get('pr_auc', 0):.5f}",
            "Optimal Threshold": f"{opt_post.get('threshold', 0):.2f}",
        }])
        st.table(m_df_post.T.rename(columns={0: "Post-Tx Metric Value"}))

    st.markdown("---")
    st.markdown("### 📊 Diagnostic Evaluation Charts")
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        cm_opt = os.path.join(PROJECT_ROOT, "reports", "figures", "16b_confusion_matrix_optimal.png")
        if os.path.exists(cm_opt):
            st.image(cm_opt, caption="Post-Transaction Calibrated Confusion Matrix", use_container_width=True)
    with fig_col2:
        roc_fig = os.path.join(PROJECT_ROOT, "reports", "figures", "17_roc_curves.png")
        if os.path.exists(roc_fig):
            st.image(roc_fig, caption="ROC & PR-AUC Curves on Untouched Test Partition", use_container_width=True)


# ==============================================================================
# PAGE 6: EXPLAINABLE AI (SHAP)
# ==============================================================================
elif navigation == "🧠 Explainable AI (SHAP)":
    st.markdown('<div class="brand-title">Explainable AI Hub (SHAP)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Shapley Additive Explanations for Regulatory Compliance & Audit Transparency</div>',
        unsafe_allow_html=True,
    )

    shap_mode = st.radio("Select Model Explanation:", ["Pre-Transaction Screening Model", "Post-Transaction Detection Model"], horizontal=True)
    is_pre = "Pre" in shap_mode
    mode_str = "pre" if is_pre else "post"

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"#### Global Feature Impact Beeswarm ({mode_str.capitalize()})")
        beeswarm_path = os.path.join(PROJECT_ROOT, "reports", "figures", f"20_shap_summary_{mode_str}.png")
        if not os.path.exists(beeswarm_path):
            beeswarm_path = os.path.join(PROJECT_ROOT, "reports", "figures", "20_shap_summary.png")
        if os.path.exists(beeswarm_path):
            st.image(beeswarm_path, use_container_width=True)

    with col_s2:
        st.markdown(f"#### Mean Absolute Feature Importance ({mode_str.capitalize()})")
        bar_path = os.path.join(PROJECT_ROOT, "reports", "figures", f"21_shap_importance_{mode_str}.png")
        if not os.path.exists(bar_path):
            bar_path = os.path.join(PROJECT_ROOT, "reports", "figures", "21_shap_feature_importance_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Individual Transaction Waterfall Attributions")
    wf1, wf2 = st.columns(2)
    with wf1:
        fraud_wf = os.path.join(PROJECT_ROOT, "reports", "figures", f"22_shap_waterfall_fraud_{mode_str}.png")
        if not os.path.exists(fraud_wf):
            fraud_wf = os.path.join(PROJECT_ROOT, "reports", "figures", "22_shap_individual_fraud_waterfall.png")
        if os.path.exists(fraud_wf):
            st.image(fraud_wf, caption=f"SHAP Waterfall: High-Risk Fraud ({mode_str.capitalize()})", use_container_width=True)

    with wf2:
        legit_wf = os.path.join(PROJECT_ROOT, "reports", "figures", f"23_shap_waterfall_legit_{mode_str}.png")
        if not os.path.exists(legit_wf):
            legit_wf = os.path.join(PROJECT_ROOT, "reports", "figures", "23_shap_individual_legit_waterfall.png")
        if os.path.exists(legit_wf):
            st.image(legit_wf, caption=f"SHAP Waterfall: Legitimate Transaction ({mode_str.capitalize()})", use_container_width=True)


# ==============================================================================
# PAGE 7: FINANCIAL IMPACT ANALYSIS
# ==============================================================================
elif navigation == "💰 Financial Impact Analysis":
    st.markdown('<div class="brand-title">Financial Risk & ROI Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Simulate Monetary Loss Mitigation & Operational Review Overhead</div>',
        unsafe_allow_html=True,
    )

    fin_data = get_financial()
    base_case = fin_data.get("base_case", {})
    meas = base_case.get("measured_metrics", {})

    st.markdown(
        """
        <div class="disclaimer-box" style="margin-top:0px; margin-bottom: 20px;">
        ⚠️ <b>Hypothetical Business Model Notice:</b> All recovery amounts and cost mitigations are modeled estimates based on stated SLA parameters.
        Detected fraud values reflect genuine empirical test partition measurements ($235.4M), while net financial benefits depend on your enterprise recovery SLA.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Configurable simulator
    st.sidebar.markdown("### 🎛️ Enterprise SLA Parameters")
    recovery_rate = st.sidebar.slider("Fraud Recovery Rate (%)", min_value=10, max_value=100, value=80, step=5) / 100.0
    investigation_cost = st.sidebar.number_input("Investigation Cost per Alert ($)", min_value=5.0, max_value=200.0, value=30.0, step=5.0)
    fp_friction = st.sidebar.slider("False Positive Customer Friction Rate (%)", min_value=0.1, max_value=5.0, value=1.5, step=0.1) / 100.0

    # Measured values from test set
    detected_fraud_amt = meas.get("detected_fraud_amount", 235430525.04)
    missed_fraud_amt = meas.get("missed_fraud_amount", 416001.33)
    flagged_alerts = meas.get("detected_fraud_count", 227) + meas.get("false_positive_count", 0)
    flagged_legit_amt = meas.get("flagged_legit_amount", 0.0)

    # Dynamic calculation
    gross_prevented = detected_fraud_amt * recovery_rate
    total_review_cost = flagged_alerts * investigation_cost
    friction_cost = flagged_legit_amt * fp_friction
    net_benefit = gross_prevented - total_review_cost - friction_cost

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        render_metric_card("Detected Fraud Volume", f"${detected_fraud_amt:,.2f}", "Total monetary value intercepted")
    with fcol2:
        render_metric_card("Modeled Net Benefit", f"${net_benefit:,.2f}", f"At {recovery_rate*100:.0f}% recovery SLA", "#2A9D8F")
    with fcol3:
        render_metric_card("Missed Fraud Exposure", f"${missed_fraud_amt:,.2f}", "Capital loss from false negatives", "#E63946")

    st.markdown("---")
    st.markdown("### 📈 Enterprise Scenario Breakdown")
    f_chart1 = os.path.join(PROJECT_ROOT, "reports", "figures", "24_financial_impact_breakdown.png")
    f_chart2 = os.path.join(PROJECT_ROOT, "reports", "figures", "25_financial_scenarios.png")

    f1, f2 = st.columns(2)
    with f1:
        if os.path.exists(f_chart1):
            st.image(f_chart1, use_container_width=True)
    with f2:
        if os.path.exists(f_chart2):
            st.image(f_chart2, use_container_width=True)
