"""
Phase 2 Data Leakage and Dataset Audit Script
Inspects duplicates, ID distributions, synthetic artifacts, correlations, and leakage.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_raw_data

def run_leakage_audit():
    print("=" * 80)
    print("   PHASE 2: DATA LEAKAGE & SYNTHETIC DATASET AUDIT")
    print("=" * 80)

    raw_df = load_raw_data()
    n_total = len(raw_df)
    n_fraud = int(raw_df["isFraud"].sum())
    n_legit = n_total - n_fraud

    print(f"\n1. DATASET OVERVIEW: Total={n_total}, Legit={n_legit} ({n_legit/n_total*100:.2f}%), Fraud={n_fraud} ({n_fraud/n_total*100:.2f}%)")

    # A. Duplicate checks
    exact_dups = int(raw_df.duplicated().sum())
    feature_dups_no_id = int(raw_df.drop(columns=["nameOrig", "nameDest"]).duplicated().sum())
    orig_unique = raw_df["nameOrig"].nunique()
    dest_unique = raw_df["nameDest"].nunique()

    print(f"   Exact Duplicates (all cols): {exact_dups}")
    print(f"   Feature Duplicates (ignoring IDs): {feature_dups_no_id}")
    print(f"   Unique nameOrig: {orig_unique} (out of {n_total})")
    print(f"   Unique nameDest: {dest_unique} (out of {n_total})")

    # Check cross-split duplicates
    train_post = pd.read_csv("data/processed/train_post.csv")
    test_post = pd.read_csv("data/processed/test_post.csv")
    
    # Check if any row in test matches a row in train
    cross_dups = pd.merge(train_post, test_post, how="inner")
    print(f"   Cross-Split Duplicates (train_post vs test_post): {len(cross_dups)}")

    train_pre = pd.read_csv("data/processed/train_pre.csv")
    test_pre = pd.read_csv("data/processed/test_pre.csv")
    cross_dups_pre = pd.merge(train_pre, test_pre, how="inner")
    print(f"   Cross-Split Duplicates (train_pre vs test_pre): {len(cross_dups_pre)}")

    # B. Synthetic Rule & Artifact Analysis
    # Rule 1: Channel specificity
    fraud_by_type = raw_df.groupby("type")["isFraud"].agg(["count", "sum"]).rename(columns={"count": "Total", "sum": "Frauds"})
    fraud_by_type["Fraud_Rate_%"] = (fraud_by_type["Frauds"] / fraud_by_type["Total"]) * 100
    print("\n2. FRAUD BY TRANSACTION TYPE:")
    print(fraud_by_type)

    # Rule 2: Account Liquidation (amount == oldbalanceOrg)
    exact_drain_all = ((raw_df["amount"] == raw_df["oldbalanceOrg"]) & (raw_df["oldbalanceOrg"] > 0))
    exact_drain_fraud = exact_drain_all[raw_df["isFraud"] == 1].sum()
    exact_drain_legit = exact_drain_all[raw_df["isFraud"] == 0].sum()
    print(f"\n3. ACCOUNT LIQUIDATION INTENT (amount == oldbalanceOrg > 0):")
    print(f"   In Fraud: {exact_drain_fraud} / {n_fraud} ({exact_drain_fraud/n_fraud*100:.2f}%)")
    print(f"   In Legit: {exact_drain_legit} / {n_legit} ({exact_drain_legit/n_legit*100:.2f}%)")

    # Rule 3: Zero Final Balance (newbalanceOrig == 0)
    zero_orig_all = (raw_df["newbalanceOrig"] == 0)
    zero_orig_fraud = zero_orig_all[raw_df["isFraud"] == 1].sum()
    zero_orig_legit = zero_orig_all[raw_df["isFraud"] == 0].sum()
    print(f"\n4. SENDER BALANCE ZERO AFTER TRANSACTION (newbalanceOrig == 0):")
    print(f"   In Fraud: {zero_orig_fraud} / {n_fraud} ({zero_orig_fraud/n_fraud*100:.2f}%)")
    print(f"   In Legit: {zero_orig_legit} / {n_legit} ({zero_orig_legit/n_legit*100:.2f}%)")

    # Rule 4: Ledger balance discrepancy
    # balance_error_orig = oldbalanceOrg - amount - newbalanceOrig
    raw_df["bal_err_orig"] = raw_df["oldbalanceOrg"] - raw_df["amount"] - raw_df["newbalanceOrig"]
    err_zero_fraud = (np.abs(raw_df[raw_df["isFraud"] == 1]["bal_err_orig"]) < 0.01).sum()
    err_zero_legit = (np.abs(raw_df[raw_df["isFraud"] == 0]["bal_err_orig"]) < 0.01).sum()
    print(f"\n5. SENDER BALANCE RECONCILIATION ERROR == 0:")
    print(f"   In Fraud: {err_zero_fraud} / {n_fraud} ({err_zero_fraud/n_fraud*100:.2f}%)")
    print(f"   In Legit: {err_zero_legit} / {n_legit} ({err_zero_legit/n_legit*100:.2f}%)")

    # Rule 5: Destination balance error
    # balance_error_dest = oldbalanceDest + amount - newbalanceDest
    raw_df["bal_err_dest"] = raw_df["oldbalanceDest"] + raw_df["amount"] - raw_df["newbalanceDest"]
    dest_zero_fraud = ((raw_df[raw_df["isFraud"] == 1]["oldbalanceDest"] == 0) & (raw_df[raw_df["isFraud"] == 1]["newbalanceDest"] == 0)).sum()
    print(f"\n6. DESTINATION NEVER CREDITED (oldbalanceDest == 0 and newbalanceDest == 0):")
    print(f"   In Fraud: {dest_zero_fraud} / {n_fraud} ({dest_zero_fraud/n_fraud*100:.2f}%)")

    # Rule 6: Temporal distribution artifact
    steps_with_legit = raw_df[raw_df["isFraud"] == 0]["step"].unique()
    steps_with_fraud = raw_df[raw_df["isFraud"] == 1]["step"].unique()
    print(f"\n7. TEMPORAL ARTIFACT:")
    print(f"   Steps with Legit transactions: {min(steps_with_legit)} to {max(steps_with_legit)} (Total: {len(steps_with_legit)})")
    print(f"   Steps with Fraud transactions: {min(steps_with_fraud)} to {max(steps_with_fraud)} (Total: {len(steps_with_fraud)})")
    print(f"   Transactions at step > 7: Total={len(raw_df[raw_df['step'] > 7])}, Frauds={raw_df[raw_df['step'] > 7]['isFraud'].sum()}, Legit={(raw_df[raw_df['step'] > 7]['isFraud'] == 0).sum()}")

    # Monetary statistics
    print("\n8. MONETARY VALUE CONTRAST:")
    f_amt = raw_df[raw_df["isFraud"] == 1]["amount"]
    l_amt = raw_df[raw_df["isFraud"] == 0]["amount"]
    print(f"   Fraud Amount: Mean=${f_amt.mean():,.2f}, Median=${f_amt.median():,.2f}, Min=${f_amt.min():,.2f}, Max=${f_amt.max():,.2f}")
    print(f"   Legit Amount: Mean=${l_amt.mean():,.2f}, Median=${l_amt.median():,.2f}, Min=${l_amt.min():,.2f}, Max=${l_amt.max():,.2f}")

    results = {
        "n_total": n_total,
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "exact_duplicates": exact_dups,
        "feature_duplicates_no_id": feature_dups_no_id,
        "cross_split_duplicates_post": len(cross_dups),
        "cross_split_duplicates_pre": len(cross_dups_pre),
        "unique_senders": orig_unique,
        "unique_receivers": dest_unique,
        "fraud_in_transfer_and_cashout_only": bool(raw_df[~raw_df["type"].isin(["TRANSFER", "CASH_OUT"])]["isFraud"].sum() == 0),
        "exact_drain_fraud_pct": float(exact_drain_fraud / n_fraud),
        "exact_drain_legit_pct": float(exact_drain_legit / n_legit),
        "zero_orig_balance_fraud_pct": float(zero_orig_fraud / n_fraud),
        "zero_orig_balance_legit_pct": float(zero_orig_legit / n_legit),
        "sender_err_zero_fraud_pct": float(err_zero_fraud / n_fraud),
        "dest_never_credited_fraud_pct": float(dest_zero_fraud / n_fraud),
        "step_legit_max": int(max(steps_with_legit)),
        "step_fraud_max": int(max(steps_with_fraud)),
        "post_step7_pure_fraud_count": int(raw_df[raw_df["step"] > 7]["isFraud"].sum()),
    }

    with open("unseen_data_audit/data/leakage_audit_summary.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n[SUCCESS] Audit metrics saved to unseen_data_audit/data/leakage_audit_summary.json")
    return results

if __name__ == "__main__":
    run_leakage_audit()

