"""
Unit tests for data validation, edge cases, and pre-transaction feature restrictions.
"""

import pytest
import numpy as np
import pandas as pd
import json
import joblib

from src.validation import (
    validate_single_transaction_inputs,
    validate_batch_dataframe,
    ValidationError,
)
from src.feature_engineering import (
    PreTransactionFeatureEngineer,
    PostTransactionFeatureEngineer,
    engineer_pre_transaction_features,
    engineer_post_transaction_features,
)
from dashboard.utils import (
    load_model_artifacts,
    predict_pre_transaction,
    predict_post_transaction,
    predict_batch_transactions,
)


def test_validation_negative_amount():
    with pytest.raises(ValidationError, match="cannot be negative"):
        validate_single_transaction_inputs(
            step=1,
            trans_type="TRANSFER",
            amount=-50.0,
            oldbalanceOrg=100.0,
            oldbalanceDest=0.0,
            mode="pre",
        )


def test_validation_negative_balance():
    with pytest.raises(ValidationError, match="cannot be negative"):
        validate_single_transaction_inputs(
            step=1,
            trans_type="CASH_OUT",
            amount=50.0,
            oldbalanceOrg=-10.0,
            oldbalanceDest=0.0,
            mode="pre",
        )


def test_validation_invalid_type():
    with pytest.raises(ValidationError, match="Invalid transaction type"):
        validate_single_transaction_inputs(
            step=1,
            trans_type="BITCOIN_TRANSFER",
            amount=50.0,
            oldbalanceOrg=100.0,
            oldbalanceDest=0.0,
            mode="pre",
        )


def test_validation_invalid_step():
    with pytest.raises(ValidationError, match="positive integer"):
        validate_single_transaction_inputs(
            step=0,
            trans_type="TRANSFER",
            amount=50.0,
            oldbalanceOrg=100.0,
            oldbalanceDest=0.0,
            mode="pre",
        )


def test_validation_nan_amount():
    with pytest.raises(ValidationError, match="must be a finite number"):
        validate_single_transaction_inputs(
            step=1,
            trans_type="TRANSFER",
            amount=np.nan,
            oldbalanceOrg=100.0,
            oldbalanceDest=0.0,
            mode="pre",
        )


def test_pre_transaction_feature_restrictions():
    """Verify that PreTransactionFeatureEngineer strictly excludes post-clearing ledger balances."""
    forbidden_features = [
        "newbalanceOrig",
        "newbalanceDest",
        "balance_error_orig",
        "balance_error_dest",
        "origin_balance_zero",
        "destination_balance_zero",
    ]

    sample_df = pd.DataFrame([{
        "step": 1,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C12345",
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C67890",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }])

    fe = PreTransactionFeatureEngineer()
    out = fe.transform(sample_df)

    for forbidden in forbidden_features:
        assert forbidden not in out.columns, f"Forbidden post-transaction feature '{forbidden}' leaked into pre-transaction features!"

    # Check allowed pre-transaction features exist
    allowed = ["transaction_hour", "transaction_day", "transaction_amount_log", "amount_to_oldbalance_orig_ratio", "merchant_destination", "empty_account_flag"]
    for feat in allowed:
        assert feat in out.columns, f"Expected pre-transaction feature '{feat}' missing!"


def test_post_transaction_feature_presence():
    """Verify that PostTransactionFeatureEngineer includes ledger settlement features."""
    sample_df = pd.DataFrame([{
        "step": 1,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C12345",
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C67890",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }])

    fe = PostTransactionFeatureEngineer()
    out = fe.transform(sample_df)

    expected = ["balance_error_orig", "balance_error_dest", "origin_balance_zero", "destination_balance_zero"]
    for feat in expected:
        assert feat in out.columns, f"Expected post-transaction feature '{feat}' missing!"


def test_dual_artifacts_and_metadata_parity():
    """Verify both pre and post artifacts load and match metadata feature counts."""
    pre_art = load_model_artifacts("pre")
    assert pre_art["mode"] == "pre"
    assert pre_art["metadata"]["system_mode"] == "pre_transaction"
    assert pre_art["metadata"]["input_features_count"] == 16
    assert 0.0 < pre_art["threshold"] < 1.0

    post_art = load_model_artifacts("post")
    assert post_art["mode"] == "post"
    assert post_art["metadata"]["system_mode"] == "post_transaction"
    assert post_art["metadata"]["input_features_count"] == 26
    assert 0.0 < post_art["threshold"] < 1.0


def test_pre_transaction_prediction_flow():
    """Verify standalone pre-transaction screening prediction function."""
    pre_art = load_model_artifacts("pre")
    res = predict_pre_transaction(
        model=pre_art["model"],
        preprocessor=pre_art["preprocessor"],
        step=10,
        trans_type="TRANSFER",
        amount=250000.0,
        oldbalanceOrg=250000.0,
        oldbalanceDest=0.0,
        nameDest="C99999",
        threshold=pre_art["threshold"],
    )

    assert "is_fraud" in res
    assert "probability" in res
    assert "risk_category" in res
    assert "policy_alerts" in res
    assert 0.0 <= res["probability"] <= 1.0


def test_post_transaction_prediction_flow():
    """Verify standalone post-transaction detection prediction function."""
    post_art = load_model_artifacts("post")
    res = predict_post_transaction(
        model=post_art["model"],
        preprocessor=post_art["preprocessor"],
        step=10,
        trans_type="TRANSFER",
        amount=250000.0,
        oldbalanceOrg=250000.0,
        newbalanceOrig=0.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
        threshold=post_art["threshold"],
    )

    assert "is_fraud" in res
    assert "probability" in res
    assert "risk_category" in res
    assert "policy_alerts" in res
    assert 0.0 <= res["probability"] <= 1.0


def test_batch_prediction_pre_and_post():
    """Verify batch prediction works across both pre and post modes."""
    pre_art = load_model_artifacts("pre")
    post_art = load_model_artifacts("post")

    pre_df = pd.DataFrame([
        {"step": 1, "type": "TRANSFER", "amount": 1000.0, "oldbalanceOrg": 1000.0, "oldbalanceDest": 0.0, "nameDest": "C123"},
        {"step": 2, "type": "PAYMENT", "amount": 50.0, "oldbalanceOrg": 500.0, "oldbalanceDest": 0.0, "nameDest": "M456"},
    ])
    pre_res = predict_batch_transactions(pre_art["model"], pre_art["preprocessor"], pre_df, mode="pre")
    assert len(pre_res) == 2
    assert "Fraud_Probability" in pre_res.columns

    post_df = pd.DataFrame([
        {"step": 1, "type": "TRANSFER", "amount": 1000.0, "oldbalanceOrg": 1000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "nameDest": "C123"},
        {"step": 2, "type": "PAYMENT", "amount": 50.0, "oldbalanceOrg": 500.0, "newbalanceOrig": 450.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "nameDest": "M456"},
    ])
    post_res = predict_batch_transactions(post_art["model"], post_art["preprocessor"], post_df, mode="post")
    assert len(post_res) == 2
    assert "Fraud_Probability" in post_res.columns
