"""
Comprehensive Unseen Data Generalization & Reliability Experiment Suite.
Executes Test A (Original Holdout), Test B (Independently Generated Unseen),
Test C (6 Distribution Shifts), Test D (Noise Robustness),
Threshold Sensitivity, and Permutation Feature Importance.
"""

import os
import sys

os.environ["MPLCONFIGDIR"] = os.path.abspath("unseen_data_audit")
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath("."))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance
from dashboard.utils import load_model_artifacts
from src.feature_engineering import (
    engineer_pre_transaction_features,
    engineer_post_transaction_features,
)

# Set random seeds
AUDIT_SEED = 2026
np.random.seed(AUDIT_SEED)

OUTPUT_DATA_DIR = "unseen_data_audit/data"
OUTPUT_FIG_DIR = "unseen_data_audit/figures"
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)


def evaluate_model_on_dataset(
    model: Any,
    preprocessor: Any,
    raw_df: pd.DataFrame,
    mode: str = "pre",
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Helper to preprocess raw transactions and score with given model."""
    if mode == "pre":
        fe_df = engineer_pre_transaction_features(raw_df, drop_identifiers=True)
    else:
        fe_df = engineer_post_transaction_features(raw_df, drop_identifiers=True)

    y_true = raw_df["isFraud"].values
    X_features = fe_df.drop(columns=["isFraud"]) if "isFraud" in fe_df.columns else fe_df

    # Transform
    X_trans = preprocessor.transform(X_features)
    probs = model.predict_proba(X_trans)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Metrics
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    
    # Specificity & Rates
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 1.0
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 1.0
    except Exception:
        pr_auc = 0.0

    # Financial Exposure
    amounts = raw_df["amount"].values
    total_fraud_val = float(amounts[y_true == 1].sum())
    flagged_fraud_val = float(amounts[(y_true == 1) & (preds == 1)].sum())
    capture_rate = (flagged_fraud_val / total_fraud_val) if total_fraud_val > 0 else 1.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "specificity": float(spec),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "total_transactions": len(raw_df),
        "total_frauds": int(y_true.sum()),
        "total_fraud_value_usd": total_fraud_val,
        "flagged_fraud_value_usd": flagged_fraud_val,
        "capital_capture_rate": float(capture_rate),
        "threshold": float(threshold),
        "probabilities": probs,
        "predictions": preds,
    }


def run_test_a_original_holdout(pre_artifacts, post_artifacts) -> Dict[str, Any]:
    print("\n--- TEST A: ORIGINAL HOLDOUT EVALUATION ---")
    raw_df = pd.read_csv("Fraud_Analysis_Dataset.csv")
    from src.preprocessing import create_train_test_split

    _, test_pre, _, y_te_pre = create_train_test_split(raw_df, mode="pre", test_size=0.2, random_state=42)
    test_pre_full = raw_df.loc[test_pre.index].copy()

    pre_res = evaluate_model_on_dataset(
        pre_artifacts["model"], pre_artifacts["preprocessor"],
        test_pre_full, mode="pre", threshold=pre_artifacts["threshold"]
    )
    post_res = evaluate_model_on_dataset(
        post_artifacts["model"], post_artifacts["preprocessor"],
        test_pre_full, mode="post", threshold=post_artifacts["threshold"]
    )

    print(f"Pre-Tx RF (th={pre_artifacts['threshold']:.2f}):  Prec={pre_res['precision']:.4f}, Rec={pre_res['recall']:.4f}, F1={pre_res['f1_score']:.4f}, PR-AUC={pre_res['pr_auc']:.4f}, TP={pre_res['true_positives']}, FP={pre_res['false_positives']}")
    print(f"Post-Tx XGB (th={post_artifacts['threshold']:.2f}): Prec={post_res['precision']:.4f}, Rec={post_res['recall']:.4f}, F1={post_res['f1_score']:.4f}, PR-AUC={post_res['pr_auc']:.4f}, TP={post_res['true_positives']}, FP={post_res['false_positives']}")

    return {"pre_transaction": pre_res, "post_transaction": post_res}


def generate_independent_unseen_dataset(n_samples: int = 12000, seed: int = 2026) -> pd.DataFrame:
    """
    Generates a completely new, independently modeled synthetic transaction dataset.
    DOES NOT reuse the deterministic templates of the original dataset.
    Includes:
    - Normal retail payments, transfers, cashouts, cashins, debits.
    - Frauds that DO NOT always drain 100% of balance (partial drains: 20% to 90%).
    - Legitimate customers who DO drain 100% of their balance (e.g. closing account, big purchase).
    - Multi-channel fraud (some in PAYMENT and DEBIT).
    - Realistic noise in ledger balances.
    """
    rng = np.random.default_rng(seed)
    records = []

    types = ["PAYMENT", "CASH_OUT", "TRANSFER", "CASH_IN", "DEBIT"]
    type_probs = [0.45, 0.25, 0.15, 0.12, 0.03]

    for i in range(n_samples):
        step = int(rng.integers(1, 150)) # Extended operational time
        t_type = rng.choice(types, p=type_probs)
        name_orig = f"C_UNSEEN_{i:07d}"
        
        # Decide if destination is Merchant or Customer
        is_merchant = (t_type == "PAYMENT") or (rng.random() < 0.15)
        name_dest = f"M_UNSEEN_{rng.integers(1000, 9999)}" if is_merchant else f"C_UNSEEN_{rng.integers(10000, 99999)}"

        # Initial balances (log-normal realistic banking balance)
        oldbalance_org = float(np.round(rng.lognormal(mean=9.5, sigma=1.8), 2)) # median ~$13k
        oldbalance_dest = float(np.round(rng.lognormal(mean=9.0, sigma=2.0), 2)) if not is_merchant else 0.0

        # Decide ground truth fraud status: 5% overall fraud rate
        # But allow fraud across multiple channels and varying behaviors
        is_fraud = 0
        if t_type in ["TRANSFER", "CASH_OUT"]:
            if rng.random() < 0.18: # 18% channel fraud rate
                is_fraud = 1
        elif t_type in ["PAYMENT", "DEBIT"]:
            if rng.random() < 0.02: # 2% novel card fraud
                is_fraud = 1

        if is_fraud == 1:
            fraud_archetype = rng.choice(["full_drain", "partial_drain", "overdraft_attempt", "random_bustout"], p=[0.40, 0.40, 0.10, 0.10])
            if fraud_archetype == "full_drain":
                # Classical full drain, but occasionally with small residual balance left
                amount = oldbalance_org if rng.random() < 0.7 else float(np.round(oldbalance_org * rng.uniform(0.95, 0.99), 2))
            elif fraud_archetype == "partial_drain":
                # Steal 20% to 75% of funds to evade naive full-drain threshold
                amount = float(np.round(oldbalance_org * rng.uniform(0.20, 0.75), 2))
            elif fraud_archetype == "overdraft_attempt":
                # Attempt to take more than balance
                amount = float(np.round(oldbalance_org * rng.uniform(1.05, 2.0), 2))
            else:
                # Flat round burst amount
                amount = float(rng.choice([50000.0, 100000.0, 250000.0, 500000.0, 1000000.0]))

            amount = max(amount, 100.0)

            # Post-clearing balances for fraud
            # Senders balance deduction
            newbalance_orig = max(0.0, float(np.round(oldbalance_org - amount, 2)))
            # Recipient may or may not be credited immediately (mule delay / cashout)
            if rng.random() < 0.5:
                newbalance_dest = oldbalance_dest # Zero credit anomaly
            else:
                newbalance_dest = float(np.round(oldbalance_dest + amount, 2))

        else:
            # Legitimate transactions
            legit_archetype = rng.choice(["typical_spend", "closing_account", "large_transfer", "salary_deposit"], p=[0.75, 0.05, 0.15, 0.05])
            if legit_archetype == "typical_spend":
                amount = float(np.round(rng.lognormal(mean=5.5, sigma=1.2), 2)) # ~$250 typical
                amount = min(amount, oldbalance_org * 0.5) if oldbalance_org > 10 else amount
            elif legit_archetype == "closing_account":
                # Legitimate customer completely zeros their balance! (Borderline challenge case)
                amount = oldbalance_org
            elif legit_archetype == "large_transfer":
                amount = float(np.round(oldbalance_org * rng.uniform(0.30, 0.85), 2))
            else:
                amount = float(np.round(rng.uniform(1000.0, 15000.0), 2))

            amount = max(amount, 1.0)
            if t_type == "CASH_IN":
                newbalance_orig = float(np.round(oldbalance_org + amount, 2))
                newbalance_dest = max(0.0, float(np.round(oldbalance_dest - amount, 2)))
            else:
                newbalance_orig = max(0.0, float(np.round(oldbalance_org - amount, 2)))
                newbalance_dest = float(np.round(oldbalance_dest + amount, 2)) if not is_merchant else 0.0

        records.append({
            "step": step,
            "type": t_type,
            "amount": amount,
            "nameOrig": name_orig,
            "oldbalanceOrg": oldbalance_org,
            "newbalanceOrig": newbalance_orig,
            "nameDest": name_dest,
            "oldbalanceDest": oldbalance_dest,
            "newbalanceDest": newbalance_dest,
            "isFraud": is_fraud,
        })

    df = pd.DataFrame(records)
    return df


def run_test_b_independent_unseen(pre_artifacts, post_artifacts) -> Tuple[Dict[str, Any], pd.DataFrame]:
    print("\n--- TEST B: INDEPENDENTLY GENERATED UNSEEN TRANSACTIONS (N=12,000) ---")
    df_unseen = generate_independent_unseen_dataset(n_samples=12000, seed=AUDIT_SEED)
    df_unseen.to_csv(f"{OUTPUT_DATA_DIR}/unseen_transactions_independent.csv", index=False)
    print(f"Generated {len(df_unseen)} transactions. Frauds={df_unseen['isFraud'].sum()} ({df_unseen['isFraud'].mean()*100:.2f}%)")

    pre_res = evaluate_model_on_dataset(
        pre_artifacts["model"], pre_artifacts["preprocessor"],
        df_unseen, mode="pre", threshold=pre_artifacts["threshold"]
    )
    post_res = evaluate_model_on_dataset(
        post_artifacts["model"], post_artifacts["preprocessor"],
        df_unseen, mode="post", threshold=post_artifacts["threshold"]
    )

    print(f"Pre-Tx RF on Unseen Data:  Prec={pre_res['precision']:.4f}, Rec={pre_res['recall']:.4f}, F1={pre_res['f1_score']:.4f}, PR-AUC={pre_res['pr_auc']:.4f}, TP={pre_res['true_positives']}, FP={pre_res['false_positives']}, FN={pre_res['false_negatives']}")
    print(f"Post-Tx XGB on Unseen Data: Prec={post_res['precision']:.4f}, Rec={post_res['recall']:.4f}, F1={post_res['f1_score']:.4f}, PR-AUC={post_res['pr_auc']:.4f}, TP={post_res['true_positives']}, FP={post_res['false_positives']}, FN={post_res['false_negatives']}")

    return {"pre_transaction": pre_res, "post_transaction": post_res}, df_unseen


def run_test_c_distribution_shifts(pre_artifacts, post_artifacts) -> Dict[str, Any]:
    print("\n--- TEST C: DISTRIBUTION SHIFT TESTING (6 SCENARIOS) ---")
    rng = np.random.default_rng(AUDIT_SEED + 10)
    scenarios = {}

    # 1. Normal Retail Distribution Shift (Small retail POS payments dominate)
    s1_df = generate_independent_unseen_dataset(n_samples=10000, seed=AUDIT_SEED + 1)
    s1_df["amount"] = np.clip(s1_df["amount"] * 0.5, 1.0, 50000.0)

    # 2. High Transaction Amount Distribution Shift (Corporate / High Net Worth, $500k - $5M)
    s2_df = generate_independent_unseen_dataset(n_samples=5000, seed=AUDIT_SEED + 2)
    s2_df["amount"] = s2_df["amount"] * rng.uniform(5.0, 20.0, size=len(s2_df))
    s2_df["oldbalanceOrg"] = s2_df["oldbalanceOrg"] * rng.uniform(5.0, 20.0, size=len(s2_df))
    s2_df["newbalanceOrig"] = np.maximum(0.0, s2_df["oldbalanceOrg"] - s2_df["amount"])

    # 3. Off-Hours / Nighttime Velocity Shift (step mod 24 in 0..5, odd hours)
    s3_df = generate_independent_unseen_dataset(n_samples=5000, seed=AUDIT_SEED + 3)
    s3_df["step"] = rng.choice([0, 1, 2, 3, 4, 24, 25, 26, 48, 49, 72, 73], size=len(s3_df))

    # 4. Low Fraud Prevalence Shift (Realistic banking environment: 0.2% fraud rate)
    s4_df = generate_independent_unseen_dataset(n_samples=20000, seed=AUDIT_SEED + 4)
    # Downsample fraud cases to exactly 0.2% (40 frauds out of 20,000)
    legit_subset = s4_df[s4_df["isFraud"] == 0].iloc[:19960]
    fraud_subset = s4_df[s4_df["isFraud"] == 1].iloc[:40]
    s4_df = pd.concat([legit_subset, fraud_subset]).sample(frac=1.0, random_state=AUDIT_SEED).reset_index(drop=True)

    # 5. Overlapping Characteristics Shift (Challenging borderline cases:
    # 50% of legit users drain 100% of balance, while 50% of frauds steal only 30-50% without zeroing balance)
    s5_df = generate_independent_unseen_dataset(n_samples=8000, seed=AUDIT_SEED + 5)
    for idx in s5_df.index:
        if s5_df.loc[idx, "isFraud"] == 0 and rng.random() < 0.40:
            # Legit zeros out!
            s5_df.loc[idx, "amount"] = s5_df.loc[idx, "oldbalanceOrg"]
            s5_df.loc[idx, "newbalanceOrig"] = 0.0
        elif s5_df.loc[idx, "isFraud"] == 1 and rng.random() < 0.50:
            # Fraud only takes 30%
            s5_df.loc[idx, "amount"] = float(np.round(s5_df.loc[idx, "oldbalanceOrg"] * 0.35, 2))
            s5_df.loc[idx, "newbalanceOrig"] = float(np.round(s5_df.loc[idx, "oldbalanceOrg"] * 0.65, 2))

    # 6. Novel Combinations Shift (Fraud attacks executed in PAYMENT and DEBIT channels targeting Merchants)
    s6_df = generate_independent_unseen_dataset(n_samples=5000, seed=AUDIT_SEED + 6)
    for idx in s6_df[s6_df["isFraud"] == 1].index:
        s6_df.loc[idx, "type"] = rng.choice(["PAYMENT", "DEBIT"])
        s6_df.loc[idx, "nameDest"] = f"M_FRAUD_{rng.integers(1000, 9999)}"

    shift_scenarios = [
        ("Scenario 1: Normal Retail Shift", s1_df),
        ("Scenario 2: High Amount Shift ($500K-$5M)", s2_df),
        ("Scenario 3: Off-Hours Nighttime Velocity", s3_df),
        ("Scenario 4: Ultra-Low Fraud Prevalence (0.2%)", s4_df),
        ("Scenario 5: Overlapping Ambiguous Profiles", s5_df),
        ("Scenario 6: Novel Payment/Debit Channel Fraud", s6_df),
    ]

    shift_results = {}
    for name, df_s in shift_scenarios:
        print(f"Evaluating {name} (N={len(df_s)}, Frauds={df_s['isFraud'].sum()})...")
        pre_res = evaluate_model_on_dataset(
            pre_artifacts["model"], pre_artifacts["preprocessor"],
            df_s, mode="pre", threshold=pre_artifacts["threshold"]
        )
        post_res = evaluate_model_on_dataset(
            post_artifacts["model"], post_artifacts["preprocessor"],
            df_s, mode="post", threshold=post_artifacts["threshold"]
        )
        shift_results[name] = {
            "pre_transaction": pre_res,
            "post_transaction": post_res,
            "sample_count": len(df_s),
            "fraud_count": int(df_s["isFraud"].sum()),
        }
        print(f"   Pre-Tx RF:  Prec={pre_res['precision']:.4f}, Rec={pre_res['recall']:.4f}, F1={pre_res['f1_score']:.4f}, PR-AUC={pre_res['pr_auc']:.4f}")
        print(f"   Post-Tx XGB: Prec={post_res['precision']:.4f}, Rec={post_res['recall']:.4f}, F1={post_res['f1_score']:.4f}, PR-AUC={post_res['pr_auc']:.4f}")

    return shift_results


def run_test_d_noise_robustness(pre_artifacts, post_artifacts, df_test: pd.DataFrame) -> Dict[str, Any]:
    print("\n--- TEST D: CONTROLLED NOISE AND ROBUSTNESS TESTING ---")
    noise_levels = [0.05, 0.10, 0.15, 0.20]
    num_cols = ["amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceOrig", "newbalanceDest"]
    rng = np.random.default_rng(AUDIT_SEED + 99)

    noise_results = {}
    for level in noise_levels:
        df_noisy = df_test.copy()
        for col in num_cols:
            if col in df_noisy.columns:
                noise = rng.normal(0, level, size=len(df_noisy))
                df_noisy[col] = np.maximum(0.0, df_noisy[col] * (1.0 + noise))

        pre_res = evaluate_model_on_dataset(
            pre_artifacts["model"], pre_artifacts["preprocessor"],
            df_noisy, mode="pre", threshold=pre_artifacts["threshold"]
        )
        post_res = evaluate_model_on_dataset(
            post_artifacts["model"], post_artifacts["preprocessor"],
            df_noisy, mode="post", threshold=post_artifacts["threshold"]
        )
        noise_results[f"{int(level*100)}%"] = {
            "level": level,
            "pre_transaction": pre_res,
            "post_transaction": post_res,
        }
        print(f"Noise {int(level*100)}%: Pre-Tx RF Rec={pre_res['recall']:.4f}, Prec={pre_res['precision']:.4f}, F1={pre_res['f1_score']:.4f} | Post-Tx XGB Rec={post_res['recall']:.4f}, Prec={post_res['precision']:.4f}, F1={post_res['f1_score']:.4f}")

    return noise_results


def run_threshold_sensitivity_analysis(pre_artifacts, post_artifacts, df_unseen: pd.DataFrame) -> Dict[str, Any]:
    print("\n--- PHASE 5: THRESHOLD SENSITIVITY ANALYSIS ---")
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    th_results = {}

    for th in thresholds:
        pre_res = evaluate_model_on_dataset(
            pre_artifacts["model"], pre_artifacts["preprocessor"],
            df_unseen, mode="pre", threshold=th
        )
        post_res = evaluate_model_on_dataset(
            post_artifacts["model"], post_artifacts["preprocessor"],
            df_unseen, mode="post", threshold=th
        )
        th_results[f"{th:.2f}"] = {
            "threshold": th,
            "pre_transaction": pre_res,
            "post_transaction": post_res,
        }
        print(f"Th={th:.2f}: Pre-Tx Prec={pre_res['precision']:.4f}, Rec={pre_res['recall']:.4f}, F1={pre_res['f1_score']:.4f} | Post-Tx Prec={post_res['precision']:.4f}, Rec={post_res['recall']:.4f}, F1={post_res['f1_score']:.4f}")

    return th_results


def run_permutation_feature_importance(pre_artifacts, df_unseen: pd.DataFrame) -> Dict[str, Any]:
    print("\n--- PHASE 6: PERMUTATION FEATURE IMPORTANCE ON UNSEEN DATA ---")
    fe_df = engineer_pre_transaction_features(df_unseen, drop_identifiers=True)
    y_true = df_unseen["isFraud"].values
    X_features = fe_df.drop(columns=["isFraud"]) if "isFraud" in fe_df.columns else fe_df

    X_trans = pre_artifacts["preprocessor"].transform(X_features)
    model = pre_artifacts["model"]

    # Use a representative sample of 2000 for fast, robust permutation importance
    sample_idx = np.random.choice(len(X_trans), size=min(2000, len(X_trans)), replace=False)
    X_sub = X_trans[sample_idx]
    y_sub = y_true[sample_idx]

    perm = permutation_importance(model, X_sub, y_sub, n_repeats=5, random_state=AUDIT_SEED, scoring="average_precision", n_jobs=1)

    # Feature names
    try:
        raw_names = pre_artifacts["preprocessor"].get_feature_names_out()
        clean_names = [n.split("__")[-1] for n in raw_names]
    except Exception:
        clean_names = [f"feat_{i}" for i in range(X_trans.shape[1])]

    df_perm = pd.DataFrame({
        "Feature": clean_names,
        "Importance_Mean": perm.importances_mean,
        "Importance_Std": perm.importances_std,
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

    print("Top 10 Most Important Features on Unseen Data (PR-AUC Permutation Impact):")
    print(df_perm.head(10).to_string(index=False))

    df_perm.to_csv(f"{OUTPUT_DATA_DIR}/permutation_importance_unseen.csv", index=False)
    return df_perm.to_dict(orient="records")


def plot_audit_visualizations(test_a, test_b, test_c, test_d, test_th, perm_imp):
    print("\n--- GENERATING COMPREHENSIVE AUDIT CHARTS ---")
    sns.set_theme(style="whitegrid")

    # Plot 1: Original Holdout vs Independently Generated Unseen Performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = ["Precision", "Recall", "F1_Score", "PR_AUC"]
    pre_a = [test_a["pre_transaction"][m.lower()] for m in metrics]
    pre_b = [test_b["pre_transaction"][m.lower()] for m in metrics]
    post_a = [test_a["post_transaction"][m.lower()] for m in metrics]
    post_b = [test_b["post_transaction"][m.lower()] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width/2, pre_a, width, label="Original Test Set", color="#1D3557")
    ax.bar(x + width/2, pre_b, width, label="Unseen Dataset (Test B)", color="#E63946")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title("Pre-Transaction Model (Random Forest)", fontsize=12, fontweight="bold")
    ax.legend()

    ax = axes[1]
    ax.bar(x - width/2, post_a, width, label="Original Test Set", color="#457B9D")
    ax.bar(x + width/2, post_b, width, label="Unseen Dataset (Test B)", color="#E76F51")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title("Post-Transaction Model (XGBoost)", fontsize=12, fontweight="bold")
    ax.legend()

    plt.suptitle("Generalization Comparison: Original Test Partition vs. Independent Unseen Data", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FIG_DIR}/01_original_vs_unseen_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Distribution Shift Performance
    fig, ax = plt.subplots(figsize=(12, 6))
    scenarios = list(test_c.keys())
    short_names = [s.split(":")[1].strip() if ":" in s else s for s in scenarios]
    f1_pre = [test_c[s]["pre_transaction"]["f1_score"] for s in scenarios]
    f1_post = [test_c[s]["post_transaction"]["f1_score"] for s in scenarios]
    rec_pre = [test_c[s]["pre_transaction"]["recall"] for s in scenarios]
    rec_post = [test_c[s]["post_transaction"]["recall"] for s in scenarios]

    x = np.arange(len(scenarios))
    ax.plot(x, f1_pre, marker="o", lw=2.5, color="#1D3557", label="Pre-Tx RF F1-Score")
    ax.plot(x, f1_post, marker="s", lw=2.5, color="#2A9D8F", label="Post-Tx XGB F1-Score")
    ax.plot(x, rec_pre, marker="^", ls="--", lw=2, color="#E63946", label="Pre-Tx RF Recall")
    ax.plot(x, rec_post, marker="d", ls="--", lw=2, color="#F4A261", label="Post-Tx XGB Recall")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Model Resilience Under 6 Distribution Shifts", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FIG_DIR}/02_distribution_shift_performance.png", dpi=300)
    plt.close()

    # Plot 3: Noise Robustness Degradation
    fig, ax = plt.subplots(figsize=(9, 5))
    noise_lvls = [int(k.replace("%", "")) for k in test_d.keys()]
    f1_pre_noise = [test_d[k]["pre_transaction"]["f1_score"] for k in test_d.keys()]
    f1_post_noise = [test_d[k]["post_transaction"]["f1_score"] for k in test_d.keys()]
    rec_pre_noise = [test_d[k]["pre_transaction"]["recall"] for k in test_d.keys()]
    rec_post_noise = [test_d[k]["post_transaction"]["recall"] for k in test_d.keys()]

    ax.plot(noise_lvls, f1_pre_noise, marker="o", lw=2.5, color="#1D3557", label="Pre-Tx RF F1")
    ax.plot(noise_lvls, f1_post_noise, marker="s", lw=2.5, color="#2A9D8F", label="Post-Tx XGB F1")
    ax.plot(noise_lvls, rec_pre_noise, marker="^", ls="--", color="#E63946", label="Pre-Tx RF Recall")
    ax.plot(noise_lvls, rec_post_noise, marker="d", ls="--", color="#F4A261", label="Post-Tx XGB Recall")

    ax.set_xlabel("Feature Perturbation Noise Level (%)", fontsize=11)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_title("Feature Noise Perturbation Impact on Performance", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FIG_DIR}/03_noise_robustness_degradation.png", dpi=300)
    plt.close()

    # Plot 4: Permutation Feature Importance on Unseen Data
    fig, ax = plt.subplots(figsize=(10, 6))
    df_p = pd.DataFrame(perm_imp).head(10).sort_values(by="Importance_Mean", ascending=True)
    ax.barh(df_p["Feature"], df_p["Importance_Mean"], xerr=df_p["Importance_Std"], color="#457B9D", capsize=4)
    ax.set_xlabel("Mean Decrease in PR-AUC (Permutation Importance)", fontsize=11)
    ax.set_title("Top 10 Feature Attributions on Independent Unseen Data", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FIG_DIR}/05_permutation_feature_importance.png", dpi=300)
    plt.close()

    print(f"[SUCCESS] All figures saved to {OUTPUT_FIG_DIR}/")


def main():
    print("=" * 80)
    print("   UNSEEN DATA GENERALIZATION & RELIABILITY AUDIT PIPELINE")
    print("=" * 80)

    with joblib.parallel_backend("threading"):
        pre_art = load_model_artifacts("pre")
        post_art = load_model_artifacts("post")

        # 1. Test A: Original Holdout
        test_a = run_test_a_original_holdout(pre_art, post_art)

        # 2. Test B: Independent Unseen Data
        test_b, df_unseen = run_test_b_independent_unseen(pre_art, post_art)

        # 3. Test C: Distribution Shifts
        test_c = run_test_c_distribution_shifts(pre_art, post_art)

        # 4. Test D: Noise Robustness
        test_d = run_test_d_noise_robustness(pre_art, post_art, df_unseen)

        # 5. Threshold Sensitivity
        test_th = run_threshold_sensitivity_analysis(pre_art, post_art, df_unseen)

        # 6. Permutation Importance
        perm_imp = run_permutation_feature_importance(pre_art, df_unseen)

        # 7. Generate Figures
        # Clean probabilities from serializable dictionary
        def strip_arrays(d):
            clean = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    clean[k] = strip_arrays(v)
                elif isinstance(v, np.ndarray):
                    clean[k] = v.tolist()
                elif k in ["probabilities", "predictions"]:
                    continue
                else:
                    clean[k] = v
            return clean

        plot_audit_visualizations(test_a, test_b, test_c, test_d, test_th, perm_imp)

        full_results = {
            "test_a_original_holdout": strip_arrays(test_a),
            "test_b_independent_unseen": strip_arrays(test_b),
            "test_c_distribution_shifts": strip_arrays(test_c),
            "test_d_noise_robustness": strip_arrays(test_d),
            "threshold_sensitivity": strip_arrays(test_th),
            "permutation_importance": perm_imp,
        }

        with open(f"{OUTPUT_DATA_DIR}/full_unseen_audit_results.json", "w") as f:
            json.dump(full_results, f, indent=4)

        print("\n" + "=" * 80)
        print("   UNSEEN AUDIT EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)


if __name__ == "__main__":
    main()
