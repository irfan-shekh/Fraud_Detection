"""
Feature Engineering Module for Pre-Transaction and Post-Transaction Detection.
Constructs domain-specific financial risk indicators and behavioral features
from transaction amounts, timestamps, and balance transitions while strictly
preventing target leakage in pre-transaction authorization workflows.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class PreTransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer strictly restricted to features available BEFORE
    transaction execution (pre-authorization screening).
    Strictly forbids: newbalanceOrig, newbalanceDest, balance_error_*, origin_balance_zero, etc.
    """

    def __init__(self, drop_identifiers: bool = True):
        self.drop_identifiers = drop_identifiers
        self.engineered_feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Transaction Hour (0-23)
        df["transaction_hour"] = df["step"] % 24

        # 2. Transaction Day (1, 2, 3...)
        df["transaction_day"] = ((df["step"] - 1) // 24) + 1

        # 3. Log-Transformed Amount
        df["transaction_amount_log"] = np.log1p(np.maximum(df["amount"], 0))

        # 4. Merchant Destination Flag (starts with 'M')
        if "nameDest" in df.columns:
            df["merchant_destination"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
        else:
            df["merchant_destination"] = 0

        # 5. Amount to Initial Sender Balance Ratio
        df["amount_to_oldbalance_orig_ratio"] = np.where(
            df["oldbalanceOrg"] > 0,
            np.clip(df["amount"] / df["oldbalanceOrg"], 0, 100),
            0.0,
        )

        # 6. Sender Initiates with Zero Balance
        df["orig_has_zero_balance"] = (df["oldbalanceOrg"] == 0).astype(int)

        # 7. Receiver Initiates with Zero Balance
        df["dest_has_zero_balance"] = (df["oldbalanceDest"] == 0).astype(int)

        # 8. Empty Account Intent Flag: Requested amount exactly equals starting balance
        df["empty_account_flag"] = (
            np.isclose(df["amount"], df["oldbalanceOrg"], atol=1e-2)
            & (df["oldbalanceOrg"] > 0)
        ).astype(int)

        # Drop post-transaction fields if present
        post_cols = [
            "newbalanceOrig", "newbalanceDest",
            "origin_balance_difference", "destination_balance_difference",
            "balance_error_orig", "balance_error_dest",
            "origin_balance_zero", "destination_balance_zero",
            "error_orig_zero", "error_dest_zero",
        ]
        df = df.drop(columns=[c for c in post_cols if c in df.columns])

        # Drop identifier columns
        if self.drop_identifiers:
            id_cols = [c for c in ["nameOrig", "nameDest"] if c in df.columns]
            df = df.drop(columns=id_cols)

        self.engineered_feature_names = list(df.columns)
        return df


class PostTransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer utilizing post-clearing ledger balances for post-processing
    audit, risk verification, and dispute resolution.
    """

    def __init__(self, drop_identifiers: bool = True):
        self.drop_identifiers = drop_identifiers
        self.engineered_feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # --- Pre-transaction features ---
        df["transaction_hour"] = df["step"] % 24
        df["transaction_day"] = ((df["step"] - 1) // 24) + 1
        df["transaction_amount_log"] = np.log1p(np.maximum(df["amount"], 0))

        if "nameDest" in df.columns:
            df["merchant_destination"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
        else:
            df["merchant_destination"] = 0

        df["amount_to_oldbalance_orig_ratio"] = np.where(
            df["oldbalanceOrg"] > 0,
            np.clip(df["amount"] / df["oldbalanceOrg"], 0, 100),
            0.0,
        )
        df["orig_has_zero_balance"] = (df["oldbalanceOrg"] == 0).astype(int)
        df["dest_has_zero_balance"] = (df["oldbalanceDest"] == 0).astype(int)
        df["empty_account_flag"] = (
            np.isclose(df["amount"], df["oldbalanceOrg"], atol=1e-2)
            & (df["oldbalanceOrg"] > 0)
        ).astype(int)

        # --- Post-transaction features ---
        # 1. Net balance deducted from sender
        df["origin_balance_difference"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

        # 2. Net balance added to receiver
        df["destination_balance_difference"] = df["newbalanceDest"] - df["oldbalanceDest"]

        # 3. Discrepancy in sender's ledger accounting
        df["balance_error_orig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]

        # 4. Discrepancy in receiver's ledger accounting
        df["balance_error_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

        # 5. Sender account was completely drained
        df["origin_balance_zero"] = (df["newbalanceOrig"] == 0).astype(int)

        # 6. Receiver account has zero post-transaction balance
        df["destination_balance_zero"] = (df["newbalanceDest"] == 0).astype(int)

        # 7. Balance Error Exact Zero Flags
        df["error_orig_zero"] = (np.abs(df["balance_error_orig"]) < 1e-2).astype(int)
        df["error_dest_zero"] = (np.abs(df["balance_error_dest"]) < 1e-2).astype(int)

        # Drop identifier columns
        if self.drop_identifiers:
            id_cols = [c for c in ["nameOrig", "nameDest"] if c in df.columns]
            df = df.drop(columns=id_cols)

        self.engineered_feature_names = list(df.columns)
        return df


# Compatibility alias
FinancialFeatureEngineer = PostTransactionFeatureEngineer


def engineer_pre_transaction_features(df: pd.DataFrame, drop_identifiers: bool = True) -> pd.DataFrame:
    """Convenience helper function for pre-transaction feature engineering."""
    fe = PreTransactionFeatureEngineer(drop_identifiers=drop_identifiers)
    return fe.transform(df)


def engineer_post_transaction_features(df: pd.DataFrame, drop_identifiers: bool = True) -> pd.DataFrame:
    """Convenience helper function for post-transaction feature engineering."""
    fe = PostTransactionFeatureEngineer(drop_identifiers=drop_identifiers)
    return fe.transform(df)


def engineer_features(df: pd.DataFrame, drop_identifiers: bool = True) -> pd.DataFrame:
    """Default feature engineering (defaults to post-transaction detection)."""
    return engineer_post_transaction_features(df, drop_identifiers=drop_identifiers)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath("."))
    from src.data_loader import load_raw_data
    raw_df = load_raw_data()
    pre_fe = engineer_pre_transaction_features(raw_df)
    post_fe = engineer_post_transaction_features(raw_df)
    print(f"[PRE-TRANSACTION] Shape: {pre_fe.shape}, Columns: {list(pre_fe.columns)}")
    print(f"[POST-TRANSACTION] Shape: {post_fe.shape}, Columns: {list(post_fe.columns)}")
