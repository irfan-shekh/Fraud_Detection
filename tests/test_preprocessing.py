"""
Unit tests for data loader, feature engineering, and preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import os
import joblib

from src.data_loader import load_raw_data, split_features_target
from src.feature_engineering import FinancialFeatureEngineer, engineer_features
from src.preprocessing import create_train_test_split, create_chronological_split


def test_data_loader():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 11142
    assert df.shape[1] == 10
    assert "isFraud" in df.columns


def test_split_features_target():
    df = load_raw_data()
    X, y = split_features_target(df)
    assert X.shape[0] == 11142
    assert "isFraud" not in X.columns
    assert y.shape[0] == 11142
    assert y.sum() == 1142


def test_feature_engineering():
    df = load_raw_data().head(10)
    fe_df = engineer_features(df, drop_identifiers=True)

    expected_cols = [
        "origin_balance_difference",
        "destination_balance_difference",
        "balance_error_orig",
        "balance_error_dest",
        "transaction_hour",
        "transaction_day",
        "transaction_amount_log",
        "origin_balance_zero",
        "destination_balance_zero",
        "merchant_destination",
        "empty_account_flag",
    ]
    for col in expected_cols:
        assert col in fe_df.columns

    # Verify identifiers are dropped
    assert "nameOrig" not in fe_df.columns
    assert "nameDest" not in fe_df.columns


def test_train_test_split():
    X_train, X_test, y_train, y_test = create_train_test_split(test_size=0.2, random_state=42)
    assert len(X_train) == 8913
    assert len(X_test) == 2229
    assert len(y_train) == 8913
    assert len(y_test) == 2229
    assert y_train.sum() == 914
    assert y_test.sum() == 228


def test_chronological_split():
    X_train, X_test, y_train, y_test = create_chronological_split(test_size=0.2)
    assert len(X_train) + len(X_test) == 11142
    assert X_train["step"].max() <= X_test["step"].min()

