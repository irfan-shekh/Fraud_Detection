"""
Dashboard Utility Module for Dual-Model Fraud Detection.
Handles cached loading of Pre- and Post-Transaction model artifacts, input validation,
feature engineering, single/batch prediction pipelines, and dynamic report loading.
"""

import os
import sys
import io
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import (
    engineer_pre_transaction_features,
    engineer_post_transaction_features,
    engineer_features,
)
from src.validation import (
    validate_single_transaction_inputs,
    validate_batch_dataframe,
    ValidationError,
)

# Paths to artifacts
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Fraud_Analysis_Dataset.csv")
DATA_PATH_FALLBACK = os.path.join(PROJECT_ROOT, "Fraud_Analysis_Dataset.csv")

PRE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pre_transaction_model.joblib")
PRE_PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "pre_transaction_pipeline.joblib")
PRE_METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "pre_transaction_metadata.json")

POST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "post_transaction_model.joblib")
POST_PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "post_transaction_pipeline.joblib")
POST_METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "post_transaction_metadata.json")

# Fallback legacy paths
LEGACY_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
LEGACY_PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessing_pipeline.joblib")
LEGACY_METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

EVAL_PATH = os.path.join(PROJECT_ROOT, "reports", "evaluation_report.json")
PRE_EVAL_PATH = os.path.join(PROJECT_ROOT, "reports", "pre_transaction_evaluation.json")
POST_EVAL_PATH = os.path.join(PROJECT_ROOT, "reports", "post_transaction_evaluation.json")
COMPARISON_PATH = os.path.join(PROJECT_ROOT, "reports", "model_comparison.json")
THRESHOLD_PATH = os.path.join(PROJECT_ROOT, "reports", "threshold_analysis.json")
FEATURE_AUDIT_PATH = os.path.join(PROJECT_ROOT, "reports", "feature_audit.json")
FIN_PATH = os.path.join(PROJECT_ROOT, "reports", "financial_impact_analysis.json")


def _patch_preprocessor_compatibility(preprocessor: Any) -> Any:
    """Defensively patch scikit-learn cross-version unpickling compatibility."""
    if preprocessor is None:
        return preprocessor
    try:
        transformers = getattr(preprocessor, "transformers_", [])
        for name, trans, cols in transformers:
            if hasattr(trans, "named_steps"):
                for step_name, step in trans.named_steps.items():
                    if "Imputer" in step.__class__.__name__:
                        if not hasattr(step, "_fill_dtype"):
                            stats = getattr(step, "statistics_", None)
                            step._fill_dtype = stats.dtype if stats is not None and hasattr(stats, "dtype") else np.float64
            elif "Imputer" in trans.__class__.__name__:
                if not hasattr(trans, "_fill_dtype"):
                    stats = getattr(trans, "statistics_", None)
                    trans._fill_dtype = stats.dtype if stats is not None and hasattr(stats, "dtype") else np.float64
    except Exception:
        pass
    return preprocessor


def load_model_artifacts(mode: str = "post") -> Dict[str, Any]:
    """
    Load serialized model, preprocessor, and metadata for 'pre' or 'post' transaction mode.
    """
    mode_clean = str(mode).lower()
    if mode_clean not in ["pre", "post"]:
        raise ValueError(f"Invalid model mode '{mode}'. Must be either 'pre' or 'post'.")

    if mode_clean == "pre":
        m_path = PRE_MODEL_PATH
        p_path = PRE_PIPELINE_PATH
        meta_path = PRE_METADATA_PATH
    else:
        m_path = POST_MODEL_PATH if os.path.exists(POST_MODEL_PATH) else LEGACY_MODEL_PATH
        p_path = POST_PIPELINE_PATH if os.path.exists(POST_PIPELINE_PATH) else LEGACY_PIPELINE_PATH
        meta_path = POST_METADATA_PATH if os.path.exists(POST_METADATA_PATH) else LEGACY_METADATA_PATH

    if not os.path.exists(m_path) or not os.path.exists(p_path):
        raise FileNotFoundError(
            f"{mode.upper()}-Transaction model artifacts not found at '{m_path}'. Please run 'python run_project.py'."
        )

    model = joblib.load(m_path)
    preprocessor = _patch_preprocessor_compatibility(joblib.load(p_path))

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "threshold": float(metadata.get("optimal_threshold", 0.50)),
        "model_name": metadata.get("model_name", "Random Forest"),
        "mode": mode,
    }


def load_dataset() -> pd.DataFrame:
    """Load raw dataset for overview dashboard metrics."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    elif os.path.exists(DATA_PATH_FALLBACK):
        return pd.read_csv(DATA_PATH_FALLBACK)
    elif os.path.exists("Fraud_Analysis_Dataset.csv"):
        return pd.read_csv("Fraud_Analysis_Dataset.csv")
    else:
        raise FileNotFoundError("Raw dataset not found.")


def load_model_comparison() -> Dict[str, Any]:
    """Load model comparison report JSON."""
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_evaluation_metrics(mode: str = "post") -> Dict[str, Any]:
    """Load evaluation report JSON for pre or post mode."""
    target_path = PRE_EVAL_PATH if mode.lower() == "pre" else POST_EVAL_PATH
    if not os.path.exists(target_path):
        target_path = EVAL_PATH

    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_financial_metrics() -> Dict[str, Any]:
    """Load financial analysis report JSON."""
    if os.path.exists(FIN_PATH):
        with open(FIN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_threshold_analysis() -> Dict[str, Any]:
    """Load threshold tradeoff curve data."""
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_feature_audit() -> Dict[str, Any]:
    """Load feature audit report JSON."""
    if os.path.exists(FEATURE_AUDIT_PATH):
        with open(FEATURE_AUDIT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def predict_pre_transaction(
    model: Any,
    preprocessor: Any,
    step: int,
    trans_type: str,
    amount: float,
    oldbalanceOrg: float,
    oldbalanceDest: float,
    nameDest: str = "C2000000000",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Score incoming transaction using PRE-TRANSACTION model.
    Only uses features available prior to authorization (NO new balances).
    """
    cleaned = validate_single_transaction_inputs(
        step=step,
        trans_type=trans_type,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        oldbalanceDest=oldbalanceDest,
        mode="pre",
    )

    input_df = pd.DataFrame([{
        "step": cleaned["step"],
        "type": cleaned["type"],
        "amount": cleaned["amount"],
        "oldbalanceOrg": cleaned["oldbalanceOrg"],
        "oldbalanceDest": cleaned["oldbalanceDest"],
        "nameDest": str(nameDest),
    }])

    fe_df = engineer_pre_transaction_features(input_df, drop_identifiers=True)

    _patch_preprocessor_compatibility(preprocessor)
    X_trans = preprocessor.transform(fe_df)

    prob = float(model.predict_proba(X_trans)[0, 1])
    th = threshold if threshold is not None else 0.50
    is_fraud = bool(prob >= th)

    if prob >= 0.80:
        risk_category = "CRITICAL"
        risk_color = "#E63946"
        action = "BLOCK TRANSACTION & TRIGGER HIGH-PRIORITY AML ESCALATION"
    elif prob >= 0.50:
        risk_category = "HIGH"
        risk_color = "#F4A261"
        action = "HOLD TRANSACTION FOR STEP-UP TWO-FACTOR AUTHENTICATION"
    elif prob >= 0.20:
        risk_category = "MODERATE"
        risk_color = "#E9C46A"
        action = "ALLOW TRANSACTION UNDER PASSIVE HEURISTIC MONITORING"
    else:
        risk_category = "MINIMAL"
        risk_color = "#2A9D8F"
        action = "AUTO-APPROVE (PROCEED TO INSTANT SETTLEMENT)"

    # Rule-based policy alerts (strictly distinct from SHAP feature attributions)
    policy_alerts = []
    if fe_df["empty_account_flag"].iloc[0] == 1:
        policy_alerts.append("Full Balance Liquidation Intent: Requested amount equals 100% of sender's starting balance.")
    if fe_df["orig_has_zero_balance"].iloc[0] == 1:
        policy_alerts.append("Zero Starting Balance: Sender account has $0.00 initial balance.")
    if cleaned["type"] in ["TRANSFER", "CASH_OUT"]:
        policy_alerts.append(f"High-Risk Channel: Transaction routed via {cleaned['type']} channel.")
    if cleaned["amount"] >= 200_000:
        policy_alerts.append(f"Large Monetary Volume: Amount exceeds $200,000 (${cleaned['amount']:,.2f}).")

    if not policy_alerts:
        policy_alerts.append("Transaction conforms to expected normal transactional baseline.")

    return {
        "mode": "pre_transaction",
        "is_fraud": is_fraud,
        "probability": prob,
        "threshold": th,
        "risk_category": risk_category,
        "risk_color": risk_color,
        "suggested_action": action,
        "policy_alerts": policy_alerts,
        "feature_summary": {
            "hour_of_day": int(fe_df["transaction_hour"].iloc[0]),
            "amount_log": round(float(fe_df["transaction_amount_log"].iloc[0]), 3),
            "amount_to_oldbalance_ratio": round(float(fe_df["amount_to_oldbalance_orig_ratio"].iloc[0]), 2),
            "merchant_dest": bool(fe_df["merchant_destination"].iloc[0] == 1),
        },
    }


def predict_post_transaction(
    model: Any,
    preprocessor: Any,
    step: int,
    trans_type: str,
    amount: float,
    oldbalanceOrg: float,
    newbalanceOrig: float,
    oldbalanceDest: float,
    newbalanceDest: float,
    nameDest: str = "C2000000000",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Score transaction using POST-TRANSACTION model.
    Utilizes post-clearing ledger balances and accounting discrepancies.
    """
    cleaned = validate_single_transaction_inputs(
        step=step,
        trans_type=trans_type,
        amount=amount,
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest,
        mode="post",
    )

    input_df = pd.DataFrame([{
        "step": cleaned["step"],
        "type": cleaned["type"],
        "amount": cleaned["amount"],
        "oldbalanceOrg": cleaned["oldbalanceOrg"],
        "newbalanceOrig": cleaned["newbalanceOrig"],
        "oldbalanceDest": cleaned["oldbalanceDest"],
        "newbalanceDest": cleaned["newbalanceDest"],
        "nameDest": str(nameDest),
    }])

    fe_df = engineer_post_transaction_features(input_df, drop_identifiers=True)

    _patch_preprocessor_compatibility(preprocessor)
    X_trans = preprocessor.transform(fe_df)

    prob = float(model.predict_proba(X_trans)[0, 1])
    th = threshold if threshold is not None else 0.50
    is_fraud = bool(prob >= th)

    if prob >= 0.80:
        risk_category = "CRITICAL"
        risk_color = "#E63946"
        action = "FREEZE ACCOUNTS & DISPATCH IMMEDIATE FRAUD INVESTIGATION ALERT"
    elif prob >= 0.50:
        risk_category = "HIGH"
        risk_color = "#F4A261"
        action = "HOLD TRANSACTION FOR SETTLEMENT RECONCILIATION & MANUAL REVIEW"
    elif prob >= 0.20:
        risk_category = "MODERATE"
        risk_color = "#E9C46A"
        action = "FLAG TRANSACTION FOR NEXT-DAY AUDIT TRAIL"
    else:
        risk_category = "MINIMAL"
        risk_color = "#2A9D8F"
        action = "SETTLEMENT RECONCILED WITH ZERO ANOMALIES"

    policy_alerts = []
    if fe_df["origin_balance_zero"].iloc[0] == 1:
        policy_alerts.append("Sender Account Liquidation: Sender balance depleted to exactly $0.00.")
    if abs(fe_df["balance_error_orig"].iloc[0]) > 0.01:
        policy_alerts.append(f"Sender Ledger Discrepancy: ${fe_df['balance_error_orig'].iloc[0]:,.2f} mismatch.")
    if abs(fe_df["balance_error_dest"].iloc[0]) > 0.01:
        policy_alerts.append(f"Receiver Ledger Discrepancy: ${fe_df['balance_error_dest'].iloc[0]:,.2f} mismatch.")
    if cleaned["type"] in ["TRANSFER", "CASH_OUT"]:
        policy_alerts.append(f"High-Risk Channel: {cleaned['type']}.")

    if not policy_alerts:
        policy_alerts.append("Transaction cleared ledger balance verification.")

    return {
        "mode": "post_transaction",
        "is_fraud": is_fraud,
        "probability": prob,
        "threshold": th,
        "risk_category": risk_category,
        "risk_color": risk_color,
        "suggested_action": action,
        "policy_alerts": policy_alerts,
        "ledger_metrics": {
            "origin_balance_diff": float(fe_df["origin_balance_difference"].iloc[0]),
            "dest_balance_diff": float(fe_df["destination_balance_difference"].iloc[0]),
            "balance_error_orig": float(fe_df["balance_error_orig"].iloc[0]),
            "balance_error_dest": float(fe_df["balance_error_dest"].iloc[0]),
            "hour_of_day": int(fe_df["transaction_hour"].iloc[0]),
        },
    }


# Backward-compatible wrapper for single prediction
predict_single_transaction = predict_post_transaction


def predict_batch_transactions(
    model: Any,
    preprocessor: Any,
    batch_df: pd.DataFrame,
    mode: str = "post",
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Process an uploaded batch DataFrame, validate schema, engineer features,
    and append fraud predictions and risk scores for either 'pre' or 'post' mode.
    """
    clean_df, _ = validate_batch_dataframe(batch_df, mode=mode)

    if mode.lower() == "pre":
        fe_df = engineer_pre_transaction_features(clean_df, drop_identifiers=True)
    else:
        fe_df = engineer_post_transaction_features(clean_df, drop_identifiers=True)

    _patch_preprocessor_compatibility(preprocessor)
    X_trans = preprocessor.transform(fe_df)

    probs = model.predict_proba(X_trans)[:, 1]
    preds = (probs >= threshold).astype(int)

    results_df = batch_df.copy()
    results_df["Fraud_Probability"] = np.round(probs, 4)
    results_df["Predicted_Fraud"] = preds
    results_df["Risk_Category"] = np.where(
        probs >= 0.80, "CRITICAL",
        np.where(probs >= 0.50, "HIGH",
        np.where(probs >= 0.20, "MODERATE", "MINIMAL"))
    )

    if "isFraud" in batch_df.columns:
        results_df["Actual_isFraud"] = batch_df["isFraud"]

    return results_df


def generate_sample_csv(mode: str = "post") -> str:
    """Generate a downloadable sample CSV for batch testing in Streamlit."""
    if mode.lower() == "pre":
        sample_data = pd.DataFrame([
            {"step": 1, "type": "PAYMENT", "amount": 9839.64, "nameOrig": "C1231006815", "oldbalanceOrg": 170136.0, "oldbalanceDest": 0.0, "nameDest": "M1979787155"},
            {"step": 1, "type": "TRANSFER", "amount": 181.0, "nameOrig": "C1305486145", "oldbalanceOrg": 181.0, "oldbalanceDest": 0.0, "nameDest": "C553264065"},
            {"step": 2, "type": "CASH_OUT", "amount": 181.0, "nameOrig": "C840083671", "oldbalanceOrg": 181.0, "oldbalanceDest": 21182.0, "nameDest": "C38997010"},
            {"step": 5, "type": "CASH_IN", "amount": 23557.0, "nameOrig": "C1002345", "oldbalanceOrg": 100000.0, "oldbalanceDest": 50000.0, "nameDest": "C2001234"},
            {"step": 10, "type": "TRANSFER", "amount": 500000.0, "nameOrig": "C999888", "oldbalanceOrg": 500000.0, "oldbalanceDest": 0.0, "nameDest": "C111222"},
        ])
    else:
        sample_data = pd.DataFrame([
            {"step": 1, "type": "PAYMENT", "amount": 9839.64, "nameOrig": "C1231006815", "oldbalanceOrg": 170136.0, "newbalanceOrig": 160296.36, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "nameDest": "M1979787155"},
            {"step": 1, "type": "TRANSFER", "amount": 181.0, "nameOrig": "C1305486145", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "nameDest": "C553264065"},
            {"step": 2, "type": "CASH_OUT", "amount": 181.0, "nameOrig": "C840083671", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0, "oldbalanceDest": 21182.0, "newbalanceDest": 21363.0, "nameDest": "C38997010"},
            {"step": 5, "type": "CASH_IN", "amount": 23557.0, "nameOrig": "C1002345", "oldbalanceOrg": 100000.0, "newbalanceOrig": 123557.0, "oldbalanceDest": 50000.0, "newbalanceDest": 26443.0, "nameDest": "C2001234"},
            {"step": 10, "type": "TRANSFER", "amount": 500000.0, "nameOrig": "C999888", "oldbalanceOrg": 500000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "nameDest": "C111222"},
        ])

    return sample_data.to_csv(index=False)
