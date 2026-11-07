"""
Unit tests for dashboard components, batch predictions, and sample CSV generation.
"""

import pytest
import io
import pandas as pd
import numpy as np

from dashboard.utils import (
    load_model_artifacts,
    predict_batch_transactions,
    generate_sample_csv,
)


def test_sample_csv_generation():
    csv_str = generate_sample_csv()
    assert isinstance(csv_str, str)
    df = pd.read_csv(io.StringIO(csv_str))
    assert len(df) == 5
    assert "amount" in df.columns
    assert "type" in df.columns


def test_batch_prediction():
    artifacts = load_model_artifacts()
    model = artifacts["model"]
    prep = artifacts["preprocessor"]

    csv_str = generate_sample_csv()
    df = pd.read_csv(io.StringIO(csv_str))

    results_df = predict_batch_transactions(model, prep, df, threshold=0.50)
    assert len(results_df) == len(df)
    assert "Fraud_Probability" in results_df.columns
    assert "Predicted_Fraud" in results_df.columns
    assert "Risk_Category" in results_df.columns


def test_batch_prediction_missing_columns():
    artifacts = load_model_artifacts()
    model = artifacts["model"]
    prep = artifacts["preprocessor"]

    invalid_df = pd.DataFrame({"amount": [100.0], "type": ["PAYMENT"]})
    with pytest.raises(ValueError):
        predict_batch_transactions(model, prep, invalid_df)

