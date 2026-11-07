"""
Feature Engineering Module.
Constructs domain-specific financial risk indicators and behavioral features
from transaction amounts, timestamps, and balance transitions.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn compatible transformer for engineering
    financial transaction and behavioral features.
    """

    def __init__(self, drop_identifiers: bool = True):
        self.drop_identifiers = drop_identifiers
        self.engineered_feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Derive domain-specific financial features.

        Parameters
        ----------
        X : pd.DataFrame
            Transaction records.

        Returns
        -------
        pd.DataFrame
            Dataframe with added engineered features.
        """
        df = X.copy()

        # 1. Origin Balance Difference: Net balance deducted from sender
        df["origin_balance_difference"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

        # 2. Destination Balance Difference: Net balance added to receiver
        df["destination_balance_difference"] = df["newbalanceDest"] - df["oldbalanceDest"]

        # 3. Balance Error Origin: Discrepancy in sender's ledger accounting
        # Expected: oldbalanceOrg - amount == newbalanceOrig => balance_error_orig == 0
        df["balance_error_orig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]

        # 4. Balance Error Destination: Discrepancy in receiver's ledger accounting
        # Expected: oldbalanceDest + amount == newbalanceDest => balance_error_dest == 0
        df["balance_error_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

        # 5. Transaction Hour: Hour of the day (0-23)
        df["transaction_hour"] = df["step"] % 24

        # 6. Transaction Day: Day index (1, 2, 3...)
        df["transaction_day"] = ((df["step"] - 1) // 24) + 1

        # 7. Log-Transformed Amount: Normalizing skewed monetary amounts
        df["transaction_amount_log"] = np.log1p(np.maximum(df["amount"], 0))

        # 8. Origin Balance Zero: Sender account was completely drained
        df["origin_balance_zero"] = (df["newbalanceOrig"] == 0).astype(int)

        # 9. Destination Balance Zero: Receiver account has zero post-transaction balance
        df["destination_balance_zero"] = (df["newbalanceDest"] == 0).astype(int)

        # 10. Merchant Destination Flag: Destination is a merchant account (starts with 'M')
        if "nameDest" in df.columns:
            df["merchant_destination"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
        else:
            df["merchant_destination"] = 0

        # 11. Empty Account Flag: Transaction amount equals starting balance
        df["empty_account_flag"] = (
            np.isclose(df["amount"], df["oldbalanceOrg"], atol=1e-2)
            & (df["oldbalanceOrg"] > 0)
        ).astype(int)

        # 12. Amount to Old Balance Ratio
        df["amount_to_oldbalance_orig_ratio"] = np.where(
            df["oldbalanceOrg"] > 0,
            np.clip(df["amount"] / df["oldbalanceOrg"], 0, 100),
            0.0,
        )

        # 13. Balance Error Exact Zero Flags
        df["error_orig_zero"] = (np.abs(df["balance_error_orig"]) < 1e-2).astype(int)
        df["error_dest_zero"] = (np.abs(df["balance_error_dest"]) < 1e-2).astype(int)

        # Drop identifier columns to avoid target leakage and high cardinality memorization
        if self.drop_identifiers:
            id_cols = [c for c in ["nameOrig", "nameDest"] if c in df.columns]
            df = df.drop(columns=id_cols)

        self.engineered_feature_names = list(df.columns)
        return df


def engineer_features(df: pd.DataFrame, drop_identifiers: bool = True) -> pd.DataFrame:
    """Convenience helper function to apply feature engineering to a DataFrame."""
    fe = FinancialFeatureEngineer(drop_identifiers=drop_identifiers)
    return fe.transform(df)


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    raw_df = load_raw_data()
    featured_df = engineer_features(raw_df)
    print("[FEATURE ENGINEERING] Transformed shape:", featured_df.shape)
    print("Columns:", list(featured_df.columns))
    print(featured_df.head(3))

