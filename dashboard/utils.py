"""
Dashboard Utility Module.
Handles cached loading of model artifacts, feature engineering on user inputs,
batch prediction pipelines, and sample test data generators.
"""

import os
import sys
import io
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import FinancialFeatureEngineer, engineer_features


MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessing_pipeline.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Fraud_Analysis_Dataset.csv")
DATA_PATH_FALLBACK = os.path.join(PROJECT_ROOT, "Fraud_Analysis_Dataset.csv")
EVAL_PATH = os.path.join(PROJECT_ROOT, "reports", "evaluation_report.json")
FIN_PATH = os.path.join(PROJECT_ROOT, "reports", "financial_impact_analysis.json")
DATA_REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "data_quality_report.json")


def _patch_preprocessor_compatibility(preprocessor: Any) -> Any:
    """Defensively patch scikit-learn cross-version unpickling compatibility (e.g. SimpleImputer._fill_dtype)."""
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


def load_model_artifacts() -> Dict[str, Any]:
    """Load serialized model, preprocessor, and metadata with fallback error handling."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Model artifacts not found. Please run 'python run_project.py' to train and save models."
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = _patch_preprocessor_compatibility(joblib.load(PIPELINE_PATH))

    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "threshold": metadata.get("optimal_threshold", 0.50),
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


def load_evaluation_metrics() -> Dict[str, Any]:
    """Load evaluation report JSON."""
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_financial_metrics() -> Dict[str, Any]:
    """Load financial analysis report JSON."""
    if os.path.exists(FIN_PATH):
        with open(FIN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def predict_single_transaction(
    model: Any,
    preprocessor: Any,
    step: int,
    trans_type: str,
    amount: float,
    oldbalanceOrg: float,
    newbalanceOrig: float,
    oldbalanceDest: float,
    newbalanceDest: float,
    nameOrig: str = "C1000000000",
    nameDest: str = "C2000000000",
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Run live prediction and risk scoring for a single transaction.
    """
    input_data = pd.DataFrame([{
        "step": int(step),
        "type": str(trans_type),
        "amount": float(amount),
        "nameOrig": str(nameOrig),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "nameDest": str(nameDest),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest),
    }])

    # Engineer features
    fe_df = engineer_features(input_data, drop_identifiers=True)

    # Preprocess
    _patch_preprocessor_compatibility(preprocessor)
    X_trans = preprocessor.transform(fe_df)

    # Inference
    prob = float(model.predict_proba(X_trans)[0, 1])
    is_fraud = bool(prob >= threshold)

    # Determine risk tier
    if prob >= 0.80:
        risk_category = "CRITICAL"
        risk_color = "#E63946"
        action = "BLOCK TRANSACTION & DISPATCH FRAUD INVESTIGATION ALERT"
    elif prob >= 0.50:
        risk_category = "HIGH"
        risk_color = "#F4A261"
        action = "HOLD TRANSACTION FOR TWO-FACTOR RE-AUTHENTICATION & MANUAL REVIEW"
    elif prob >= 0.20:
        risk_category = "MODERATE"
        risk_color = "#E9C46A"
        action = "ALLOW TRANSACTION WITH PASSIVE STEP-UP MONITORING"
    else:
        risk_category = "MINIMAL"
        risk_color = "#2A9D8F"
        action = "AUTO-APPROVE TRANSACTION (STANDARD PROCESSING)"

    # Identify domain rule triggers
    reasons = []
    if fe_df["origin_balance_zero"].iloc[0] == 1:
        reasons.append("Sender account balance depleted to zero ($0.00).")
    if fe_df["empty_account_flag"].iloc[0] == 1:
        reasons.append("Exact 100% balance drainage (amount equals initial sender balance).")
    if abs(fe_df["balance_error_orig"].iloc[0]) > 0.01:
        reasons.append(f"Sender ledger imbalance detected: ${fe_df['balance_error_orig'].iloc[0]:,.2f} mismatch.")
    if abs(fe_df["balance_error_dest"].iloc[0]) > 0.01:
        reasons.append(f"Receiver ledger imbalance detected: ${fe_df['balance_error_dest'].iloc[0]:,.2f} mismatch.")
    if trans_type in ["TRANSFER", "CASH_OUT"]:
        reasons.append(f"Transaction type is high-risk channel ({trans_type}).")
    if amount > 200_000:
        reasons.append(f"Large monetary value transaction (${amount:,.2f}).")

    if not reasons:
        reasons.append("Transaction conforms to expected normal transactional behavior.")

    return {
        "is_fraud": is_fraud,
        "probability": prob,
        "risk_category": risk_category,
        "risk_color": risk_color,
        "suggested_action": action,
        "contributing_factors": reasons,
        "engineered_summary": {
            "origin_balance_diff": float(fe_df["origin_balance_difference"].iloc[0]),
            "dest_balance_diff": float(fe_df["destination_balance_difference"].iloc[0]),
            "balance_error_orig": float(fe_df["balance_error_orig"].iloc[0]),
            "balance_error_dest": float(fe_df["balance_error_dest"].iloc[0]),
            "hour_of_day": int(fe_df["transaction_hour"].iloc[0]),
        },
    }


def predict_batch_transactions(
    model: Any,
    preprocessor: Any,
    batch_df: pd.DataFrame,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Process an uploaded batch DataFrame, validate schema, engineer features,
    and append fraud predictions and risk scores.
    """
    required_cols = [
        "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest"
    ]
    missing = [c for c in required_cols if c not in batch_df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing}")

    df_clean = batch_df.copy()

    # Ensure optional identifiers exist for feature engineering
    if "nameOrig" not in df_clean.columns:
        df_clean["nameOrig"] = "C00000000"
    if "nameDest" not in df_clean.columns:
        df_clean["nameDest"] = "C99999999"

    # Save target if present for reporting, but drop for prediction
    actual_target = None
    if "isFraud" in df_clean.columns:
        actual_target = df_clean["isFraud"].copy()
        df_to_pred = df_clean.drop(columns=["isFraud"])
    else:
        df_to_pred = df_clean

    fe_df = engineer_features(df_to_pred, drop_identifiers=True)
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

    if actual_target is not None:
        results_df["Actual_isFraud"] = actual_target

    return results_df


def generate_sample_csv() -> str:
    """Generate a downloadable sample CSV for batch testing in Streamlit."""
    sample_data = pd.DataFrame([
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 9839.64,
            "nameOrig": "C1231006815",
            "oldbalanceOrg": 170136.0,
            "newbalanceOrig": 160296.36,
            "nameDest": "M1979787155",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        },
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "nameOrig": "C1305486145",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C553264065",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        },
        {
            "step": 1,
            "type": "CASH_OUT",
            "amount": 181.0,
            "nameOrig": "C840083671",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C38997010",
            "oldbalanceDest": 21182.0,
            "newbalanceDest": 0.0,
        },
        {
            "step": 2,
            "type": "CASH_IN",
            "amount": 15800.0,
            "nameOrig": "C190036674",
            "oldbalanceOrg": 50000.0,
            "newbalanceOrig": 65800.0,
            "nameDest": "C38997010",
            "oldbalanceDest": 100000.0,
            "newbalanceDest": 84200.0,
        },
        {
            "step": 3,
            "type": "TRANSFER",
            "amount": 2806.0,
            "nameOrig": "C1420196421",
            "oldbalanceOrg": 2806.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C972765878",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        },
    ])
    return sample_data.to_csv(index=False)

