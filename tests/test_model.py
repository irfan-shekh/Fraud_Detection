"""
Unit tests for model loading, prediction, threshold calibration, and serialization.
"""

import pytest
import os
import joblib
import numpy as np
import pandas as pd

from dashboard.utils import (
    load_model_artifacts,
    predict_single_transaction,
    predict_batch_transactions,
)


def test_model_artifacts_exist():
    assert os.path.exists("models/best_model.joblib")
    assert os.path.exists("models/preprocessing_pipeline.joblib")
    assert os.path.exists("models/model_metadata.json")


def test_model_loading():
    artifacts = load_model_artifacts()
    assert "model" in artifacts
    assert "preprocessor" in artifacts
    assert "threshold" in artifacts
    assert hasattr(artifacts["model"], "predict")


def test_single_transaction_prediction():
    artifacts = load_model_artifacts()
    model = artifacts["model"]
    prep = artifacts["preprocessor"]

    # Fraudulent drained transfer
    res = predict_single_transaction(
        model=model,
        preprocessor=prep,
        step=1,
        trans_type="TRANSFER",
        amount=181.0,
        oldbalanceOrg=181.0,
        newbalanceOrig=0.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
    )

    assert "is_fraud" in res
    assert "probability" in res
    assert "risk_category" in res
    assert 0.0 <= res["probability"] <= 1.0


def test_legitimate_prediction():
    artifacts = load_model_artifacts()
    model = artifacts["model"]
    prep = artifacts["preprocessor"]

    # Normal legitimate payment to merchant
    res = predict_single_transaction(
        model=model,
        preprocessor=prep,
        step=2,
        trans_type="PAYMENT",
        amount=1000.0,
        oldbalanceOrg=5000.0,
        newbalanceOrig=4000.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
        nameDest="M12345678",
    )

    assert res["is_fraud"] is False
    assert res["probability"] < 0.20
    assert res["risk_category"] == "MINIMAL"


def test_invalid_mode_handling():
    with pytest.raises(ValueError, match="Invalid model mode"):
        load_model_artifacts(mode="unsupported_mode")


def test_missing_artifact_handling():
    from unittest.mock import patch
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_model_artifacts(mode="pre")
