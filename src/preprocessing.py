"""
Data Preprocessing Module.
Builds leak-free ColumnTransformer and Pipeline for encoding and scaling,
performs Stratified and Chronological Train-Test splits, and persists artifacts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.data_loader import load_raw_data, split_features_target
from src.feature_engineering import FinancialFeatureEngineer, engineer_features


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

    Parameters
    ----------
    categorical_cols : list
        List of categorical column names.
    numerical_cols : list
        List of numerical column names.

    Returns
    -------
    ColumnTransformer
        Configured preprocessor transformer.
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
    use_feature_engineering : bool, default=True
        Whether to engineer features before returning splits.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    if df is None:
        df = load_raw_data()

    if use_feature_engineering:
        df_proc = engineer_features(df, drop_identifiers=True)
    else:
        df_proc = df.drop(
            columns=[c for c in ["nameOrig", "nameDest"] if c in df.columns]
        )

    X, y = split_features_target(df_proc, target_col="isFraud")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(
        f"[SPLIT] Stratified Split (80/20): Train={X_train.shape[0]} rows "
        f"({y_train.sum()} frauds), Test={X_test.shape[0]} rows ({y_test.sum()} frauds)"
    )
    return X_train, X_test, y_train, y_test


def create_chronological_split(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    use_feature_engineering: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronological validation split based on transaction 'step' (hour).
    Simulates real-world deployment where past data trains models on future transactions.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    if df is None:
        df = load_raw_data()

    if use_feature_engineering:
        df_proc = engineer_features(df, drop_identifiers=True)
    else:
        df_proc = df.drop(
            columns=[c for c in ["nameOrig", "nameDest"] if c in df.columns]
        )

    # Sort chronologically by step
    df_sorted = df_proc.sort_values(by="step").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train, y_train = split_features_target(train_df, "isFraud")
    X_test, y_test = split_features_target(test_df, "isFraud")

    print(
        f"[CHRONOLOGICAL SPLIT] Train steps: {train_df['step'].min()} - {train_df['step'].max()} "
        f"({y_train.sum()} frauds) | Test steps: {test_df['step'].min()} - {test_df['step'].max()} "
        f"({y_test.sum()} frauds)"
    )
    return X_train, X_test, y_train, y_test


def prepare_and_save_data(
    output_dir: str = "data/processed",
    pipeline_save_path: str = "models/preprocessing_pipeline.joblib",
) -> Dict[str, Any]:
    """
    Execute end-to-end preprocessing, fit leak-free pipeline on X_train only,
    transform both sets, and save data and pipeline artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(pipeline_save_path), exist_ok=True)

    X_train, X_test, y_train, y_test = create_train_test_split()

    cat_cols, num_cols = get_feature_columns(X_train)
    preprocessor = build_preprocessor_pipeline(cat_cols, num_cols)

    # Fit preprocessor strictly on training data
    preprocessor.fit(X_train)

    # Transform
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Save fitted pipeline
    joblib.dump(preprocessor, pipeline_save_path)
    print(f"[PREPROCESSING] Fitted preprocessor saved to '{pipeline_save_path}'.")

    # Save raw split csvs for transparency
    train_full = X_train.copy()
    train_full["isFraud"] = y_train
    train_full.to_csv(os.path.join(output_dir, "train.csv"), index=False)

    test_full = X_test.copy()
    test_full["isFraud"] = y_test
    test_full.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"[PREPROCESSING] Processed train and test sets saved in '{output_dir}'.")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols,
    }


if __name__ == "__main__":
    prepare_and_save_data()

