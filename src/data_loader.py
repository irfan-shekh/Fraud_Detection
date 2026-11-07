"""
Data Loader Module for Fraud Detection Project.
Handles robust ingestion of BIA financial transaction dataset.
"""

import os
import pandas as pd
from typing import Tuple, Optional


def get_default_data_path() -> str:
    """Return the relative path to the raw dataset."""
    primary_path = os.path.join("data", "raw", "Fraud_Analysis_Dataset.csv")
    fallback_path = "Fraud_Analysis_Dataset.csv"
    if os.path.exists(primary_path):
        return primary_path
    elif os.path.exists(fallback_path):
        return fallback_path
    else:
        raise FileNotFoundError(
            f"Dataset not found at '{primary_path}' or '{fallback_path}'. "
            "Please ensure Fraud_Analysis_Dataset.csv is placed in the project directory."
        )


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw transaction dataset into a pandas DataFrame.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. If None, default discovery path is used.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with standard column names.
    """
    if filepath is None:
        filepath = get_default_data_path()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[DATA LOADER] Successfully loaded dataset from '{filepath}' with shape {df.shape}")
    return df


def split_features_target(
    df: pd.DataFrame, target_col: str = "isFraud"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction dataframe.
    target_col : str, default='isFraud'
        Target column name.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X (features) and y (target).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


if __name__ == "__main__":
    df = load_raw_data()
    X, y = split_features_target(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Class breakdown:\n", y.value_counts(normalize=True))

