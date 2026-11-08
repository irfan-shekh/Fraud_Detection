"""
Input Validation and Integrity Verification Module.
Provides robust, reusable validation functions for single transaction inputs
and batch transaction DataFrames in both Pre-Transaction and Post-Transaction modes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional


VALID_TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

PRE_TRANSACTION_REQUIRED_COLUMNS = [
    "step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"
]

POST_TRANSACTION_REQUIRED_COLUMNS = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest"
]


class ValidationError(ValueError):
    """Custom exception raised when transaction data violates business integrity rules."""
    pass


def validate_single_transaction_inputs(
    step: Any,
    trans_type: Any,
    amount: Any,
    oldbalanceOrg: Any,
    oldbalanceDest: Any,
    newbalanceOrig: Optional[Any] = None,
    newbalanceDest: Optional[Any] = None,
    mode: str = "post",
) -> Dict[str, Any]:
    """
    Validate and coerce input fields for a single transaction.

    Parameters
    ----------
    step : Any
        Simulation time step (positive integer).
    trans_type : Any
        Transaction instrument type (e.g. 'TRANSFER', 'CASH_OUT').
    amount : Any
        Monetary amount in USD (non-negative float).
    oldbalanceOrg : Any
        Initial sender balance (non-negative float).
    oldbalanceDest : Any
        Initial recipient balance (non-negative float).
    newbalanceOrig : Any, optional
        Post-transaction sender balance (required for 'post' mode).
    newbalanceDest : Any, optional
        Post-transaction recipient balance (required for 'post' mode).
    mode : str, default='post'
        Validation mode: 'pre' (pre-transaction) or 'post' (post-transaction).

    Returns
    -------
    Dict[str, Any]
        Cleaned, coerced dictionary ready for feature engineering.

    Raises
    ------
    ValidationError
        If any input fails domain integrity or type constraints.
    """
    errors: List[str] = []

    # 1. Step validation
    try:
        clean_step = int(step)
        if clean_step < 1:
            errors.append(f"Transaction step must be a positive integer >= 1 (got {clean_step}).")
    except (ValueError, TypeError):
        errors.append(f"Invalid transaction step '{step}': must be an integer.")

    # 2. Transaction Type validation
    clean_type = str(trans_type).strip().upper()
    if clean_type not in VALID_TRANSACTION_TYPES:
        errors.append(
            f"Invalid transaction type '{clean_type}'. Must be one of: {', '.join(VALID_TRANSACTION_TYPES)}."
        )

    # 3. Amount validation
    try:
        clean_amount = float(amount)
        if np.isnan(clean_amount) or np.isinf(clean_amount):
            errors.append("Transaction amount must be a finite number.")
        elif clean_amount < 0:
            errors.append(f"Transaction amount cannot be negative (got ${clean_amount:,.2f}).")
    except (ValueError, TypeError):
        errors.append(f"Invalid amount '{amount}': must be a valid number.")

    # 4. Old Balance Origin
    try:
        clean_old_orig = float(oldbalanceOrg)
        if np.isnan(clean_old_orig) or np.isinf(clean_old_orig):
            errors.append("Sender original balance must be a finite number.")
        elif clean_old_orig < 0:
            errors.append(f"Sender original balance cannot be negative (got ${clean_old_orig:,.2f}).")
    except (ValueError, TypeError):
        errors.append(f"Invalid oldbalanceOrg '{oldbalanceOrg}': must be a valid number.")

    # 5. Old Balance Destination
    try:
        clean_old_dest = float(oldbalanceDest)
        if np.isnan(clean_old_dest) or np.isinf(clean_old_dest):
            errors.append("Receiver original balance must be a finite number.")
        elif clean_old_dest < 0:
            errors.append(f"Receiver original balance cannot be negative (got ${clean_old_dest:,.2f}).")
    except (ValueError, TypeError):
        errors.append(f"Invalid oldbalanceDest '{oldbalanceDest}': must be a valid number.")

    # 6. Post-transaction fields validation
    clean_new_orig = None
    clean_new_dest = None
    if mode.lower() == "post":
        if newbalanceOrig is None:
            errors.append("Sender new balance ('newbalanceOrig') is required for post-transaction detection.")
        else:
            try:
                clean_new_orig = float(newbalanceOrig)
                if np.isnan(clean_new_orig) or np.isinf(clean_new_orig):
                    errors.append("Sender new balance must be a finite number.")
                elif clean_new_orig < 0:
                    errors.append(f"Sender new balance cannot be negative (got ${clean_new_orig:,.2f}).")
            except (ValueError, TypeError):
                errors.append(f"Invalid newbalanceOrig '{newbalanceOrig}': must be a valid number.")

        if newbalanceDest is None:
            errors.append("Receiver new balance ('newbalanceDest') is required for post-transaction detection.")
        else:
            try:
                clean_new_dest = float(newbalanceDest)
                if np.isnan(clean_new_dest) or np.isinf(clean_new_dest):
                    errors.append("Receiver new balance must be a finite number.")
                elif clean_new_dest < 0:
                    errors.append(f"Receiver new balance cannot be negative (got ${clean_new_dest:,.2f}).")
            except (ValueError, TypeError):
                errors.append(f"Invalid newbalanceDest '{newbalanceDest}': must be a valid number.")

    if errors:
        raise ValidationError("; ".join(errors))

    cleaned = {
        "step": clean_step,
        "type": clean_type,
        "amount": clean_amount,
        "oldbalanceOrg": clean_old_orig,
        "oldbalanceDest": clean_old_dest,
    }
    if mode.lower() == "post":
        cleaned["newbalanceOrig"] = clean_new_orig
        cleaned["newbalanceDest"] = clean_new_dest

    return cleaned


def validate_batch_dataframe(
    df: pd.DataFrame, mode: str = "post"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate schema, types, and ranges for batch transaction CSV upload.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded batch DataFrame.
    mode : str, default='post'
        Validation mode: 'pre' or 'post'.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Validated cleaned DataFrame and list of non-fatal warnings.

    Raises
    ------
    ValidationError
        If missing required columns, completely empty, or invalid data types.
    """
    if df is None or len(df) == 0:
        raise ValidationError("Uploaded dataset is empty.")

    required_cols = (
        PRE_TRANSACTION_REQUIRED_COLUMNS
        if mode.lower() == "pre"
        else POST_TRANSACTION_REQUIRED_COLUMNS
    )

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Uploaded CSV is missing required {mode}-transaction columns: {missing}. "
            f"Expected columns: {required_cols}"
        )

    clean_df = df.copy()
    warnings: List[str] = []

    # Check for NaN / Inf
    for col in required_cols:
        if clean_df[col].isnull().any():
            nan_count = int(clean_df[col].isnull().sum())
            raise ValidationError(
                f"Column '{col}' contains {nan_count} missing (NaN) values. All transaction fields must be populated."
            )

    # Step validation
    if not np.issubdtype(clean_df["step"].dtype, np.integer):
        try:
            clean_df["step"] = clean_df["step"].astype(int)
        except Exception:
            raise ValidationError("Column 'step' must contain valid integer hours.")
    if (clean_df["step"] < 1).any():
        raise ValidationError("Column 'step' contains non-positive values (< 1).")

    # Transaction type validation
    clean_df["type"] = clean_df["type"].astype(str).str.strip().str.upper()
    invalid_types = clean_df[~clean_df["type"].isin(VALID_TRANSACTION_TYPES)]["type"].unique()
    if len(invalid_types) > 0:
        raise ValidationError(
            f"Column 'type' contains unrecognized transaction categories: {list(invalid_types)}. "
            f"Allowed types: {VALID_TRANSACTION_TYPES}"
        )

    # Monetary numeric checks
    numeric_cols = ["amount", "oldbalanceOrg", "oldbalanceDest"]
    if mode.lower() == "post":
        numeric_cols.extend(["newbalanceOrig", "newbalanceDest"])

    for col in numeric_cols:
        try:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="raise")
        except Exception:
            raise ValidationError(f"Column '{col}' contains non-numeric values.")

        if np.isinf(clean_df[col]).any():
            raise ValidationError(f"Column '{col}' contains infinite (Inf) values.")

        if (clean_df[col] < 0).any():
            neg_count = int((clean_df[col] < 0).sum())
            raise ValidationError(
                f"Column '{col}' contains {neg_count} negative values. Balances and amounts must be non-negative."
            )

    return clean_df, warnings

