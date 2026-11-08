"""
Data Preprocessing Module for Pre- and Post-Transaction Detection.
Builds leak-free ColumnTransformer and Pipeline for encoding and scaling,
performs Stratified and Chronological Train-Test splits, and persists artifacts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.data_loader import load_raw_data, split_features_target
from src.feature_engineering import (
    engineer_pre_transaction_features,
    engineer_post_transaction_features,
    engineer_features,
)


def get_feature_columns(df_features: pd.DataFrame) -> Tuple[list, list]:
    """Identify categorical and numerical column names."""
    categorical_cols = [c for c in df_features.columns if df_features[c].dtype == "object"]
    numerical_cols = [c for c in df_features.columns if c not in categorical_cols]
    return categorical_cols, numerical_cols


def build_preprocessor_pipeline(
    categorical_cols: list, numerical_cols: list
) -> ColumnTransformer:
    """
    Construct a ColumnTransformer with OneHotEncoder for categoricals
    and RobustScaler for skewed financial distributions.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="first"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def create_train_test_split(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    mode: str = "post",
    use_feature_engineering: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into Stratified Train and Test sets.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input raw dataframe.
    test_size : float, default=0.2
        Fraction for test set.
    random_state : int, default=42
        Reproducibility seed.
    mode : str, default='post'
        'pre' for pre-transaction feature set, 'post' for post-transaction feature set.
    use_feature_engineering : bool, default=True
        Whether to engineer domain features.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    if df is None:
        df = load_raw_data()

    if not use_feature_engineering:
        df_proc = df.drop(
            columns=[c for c in ["nameOrig", "nameDest"] if c in df.columns]
        )
    elif mode.lower() == "pre":
        df_proc = engineer_pre_transaction_features(df, drop_identifiers=True)
    else:
        df_proc = engineer_post_transaction_features(df, drop_identifiers=True)

    X, y = split_features_target(df_proc, target_col="isFraud")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(
        f"[SPLIT - {mode.upper()}] Stratified Split (80/20): Train={X_train.shape[0]} rows "
        f"({y_train.sum()} frauds, {X_train.shape[1]} features), Test={X_test.shape[0]} rows ({y_test.sum()} frauds)"
    )
    return X_train, X_test, y_train, y_test


def create_chronological_split(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    mode: str = "post",
    use_feature_engineering: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronological validation split based on transaction 'step' (hour).
    """
    if df is None:
        df = load_raw_data()

    if not use_feature_engineering:
        df_proc = df.drop(
            columns=[c for c in ["nameOrig", "nameDest"] if c in df.columns]
        )
    elif mode.lower() == "pre":
        df_proc = engineer_pre_transaction_features(df, drop_identifiers=True)
    else:
        df_proc = engineer_post_transaction_features(df, drop_identifiers=True)

    df_sorted = df_proc.sort_values(by="step").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train, y_train = split_features_target(train_df, "isFraud")
    X_test, y_test = split_features_target(test_df, "isFraud")

    print(
        f"[CHRONOLOGICAL SPLIT - {mode.upper()}] Train steps: {train_df['step'].min()} - {train_df['step'].max()} "
        f"({y_train.sum()} frauds) | Test steps: {test_df['step'].min()} - {test_df['step'].max()} "
        f"({y_test.sum()} frauds)"
    )
    return X_train, X_test, y_train, y_test


def prepare_and_save_data(output_dir: str = "data/processed") -> Dict[str, Any]:
    """
    Prepare and save both pre- and post-transaction split datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_df = load_raw_data()

    # Pre-transaction splits
    X_tr_pre, X_te_pre, y_tr_pre, y_te_pre = create_train_test_split(raw_df, mode="pre")
    train_pre = X_tr_pre.copy()
    train_pre["isFraud"] = y_tr_pre
    train_pre.to_csv(os.path.join(output_dir, "train_pre.csv"), index=False)
    test_pre = X_te_pre.copy()
    test_pre["isFraud"] = y_te_pre
    test_pre.to_csv(os.path.join(output_dir, "test_pre.csv"), index=False)

    # Post-transaction splits
    X_tr_post, X_te_post, y_tr_post, y_te_post = create_train_test_split(raw_df, mode="post")
    train_post = X_tr_post.copy()
    train_post["isFraud"] = y_tr_post
    train_post.to_csv(os.path.join(output_dir, "train_post.csv"), index=False)
    train_post.to_csv(os.path.join(output_dir, "train.csv"), index=False)  # Legacy compatibility
    test_post = X_te_post.copy()
    test_post["isFraud"] = y_te_post
    test_post.to_csv(os.path.join(output_dir, "test_post.csv"), index=False)
    test_post.to_csv(os.path.join(output_dir, "test.csv"), index=False)    # Legacy compatibility

    cat_pre, num_pre = get_feature_columns(X_tr_pre)
    cat_post, num_post = get_feature_columns(X_tr_post)

    prep_pre = build_preprocessor_pipeline(cat_pre, num_pre)
    prep_post = build_preprocessor_pipeline(cat_post, num_post)

    return {
        "pre": {
            "X_train": X_tr_pre, "X_test": X_te_pre,
            "y_train": y_tr_pre, "y_test": y_te_pre,
            "preprocessor": prep_pre,
            "categorical_cols": cat_pre, "numerical_cols": num_pre,
        },
        "post": {
            "X_train": X_tr_post, "X_test": X_te_post,
            "y_train": y_tr_post, "y_test": y_te_post,
            "preprocessor": prep_post,
            "categorical_cols": cat_post, "numerical_cols": num_post,
        },
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath("."))
    prepare_and_save_data()
