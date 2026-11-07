"""
Financial Risk and Business Impact Analysis Module.
Quantifies monetary fraud prevention, false positive operational costs,
and evaluates configurable business scenarios (Low, Medium, High).
"""

import os
import json
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

sns.set_theme(style="whitegrid", palette="muted")


def compute_financial_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    investigation_cost_per_flag: float = 30.0,
    fraud_recovery_rate: float = 0.80,
    fp_friction_rate: float = 0.015,
) -> Dict[str, Any]:
    """
    Compute rigorous financial metrics comparing measured dataset values
    with configurable operational cost assumptions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amounts = np.asarray(amounts)

    # Boolean masks
    tp_mask = (y_true == 1) & (y_pred == 1)  # Detected Fraud
    fn_mask = (y_true == 1) & (y_pred == 0)  # Missed Fraud
    fp_mask = (y_true == 0) & (y_pred == 1)  # False Alarm (Legitimate Flagged)
    tn_mask = (y_true == 0) & (y_pred == 0)  # True Legitimate

    # Counts
    total_fraud_count = int((y_true == 1).sum())
    total_legit_count = int((y_true == 0).sum())
    tp_count = int(tp_mask.sum())
    fn_count = int(fn_mask.sum())
    fp_count = int(fp_mask.sum())
    tn_count = int(tn_mask.sum())

    # Dollar amounts directly measured from dataset
    total_fraud_amount = float(amounts[y_true == 1].sum())
    total_legit_amount = float(amounts[y_true == 0].sum())
    detected_fraud_amount = float(amounts[tp_mask].sum())
    missed_fraud_amount = float(amounts[fn_mask].sum())
    flagged_legit_amount = float(amounts[fp_mask].sum())

    # Rates
    detection_rate_pct = float(tp_count / max(total_fraud_count, 1) * 100)
    dollar_detection_rate_pct = float(detected_fraud_amount / max(total_fraud_amount, 1e-6) * 100)

    # Business impact calculations based on explicit stated assumptions
    gross_loss_prevented = detected_fraud_amount * fraud_recovery_rate
    loss_incurred_missed_fraud = missed_fraud_amount
    total_investigation_cost = (tp_count + fp_count) * investigation_cost_per_flag
    fp_friction_cost = flagged_legit_amount * fp_friction_rate
    total_operational_cost = total_investigation_cost + fp_friction_cost

    net_financial_benefit = gross_loss_prevented - total_operational_cost

    return {
        "measured_metrics": {
            "total_fraud_transactions": total_fraud_count,
            "total_legit_transactions": total_legit_count,
            "detected_fraud_count": tp_count,
            "missed_fraud_count": fn_count,
            "false_positive_count": fp_count,
            "true_negative_count": tn_count,
            "total_fraud_amount": total_fraud_amount,
            "total_legit_amount": total_legit_amount,
            "detected_fraud_amount": detected_fraud_amount,
            "missed_fraud_amount": missed_fraud_amount,
            "flagged_legit_amount": flagged_legit_amount,
            "transaction_detection_rate_pct": round(detection_rate_pct, 2),
            "dollar_detection_rate_pct": round(dollar_detection_rate_pct, 2),
        },
        "business_assumptions": {
            "investigation_cost_per_flag_usd": float(investigation_cost_per_flag),
            "fraud_recovery_rate": float(fraud_recovery_rate),
            "fp_friction_rate": float(fp_friction_rate),
        },
        "estimated_business_impact": {
            "gross_loss_prevented_usd": round(gross_loss_prevented, 2),
            "loss_incurred_missed_fraud_usd": round(loss_incurred_missed_fraud, 2),
            "total_investigation_cost_usd": round(total_investigation_cost, 2),
            "fp_friction_cost_usd": round(fp_friction_cost, 2),
            "total_operational_cost_usd": round(total_operational_cost, 2),
            "net_financial_benefit_usd": round(net_financial_benefit, 2),
        },
    }


def evaluate_cost_scenarios(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Simulate Low, Medium, and High enterprise cost scenarios.
    """
    scenarios = {
        "Low Cost / High Efficiency": {
            "investigation_cost_per_flag": 15.0,
            "fraud_recovery_rate": 0.90,
            "fp_friction_rate": 0.005,
        },
        "Medium (Base Case)": {
            "investigation_cost_per_flag": 30.0,
            "fraud_recovery_rate": 0.80,
            "fp_friction_rate": 0.015,
        },
        "High Cost / Conservative": {
            "investigation_cost_per_flag": 60.0,
            "fraud_recovery_rate": 0.70,
            "fp_friction_rate": 0.030,
        },
    }

    results = {}
    for name, params in scenarios.items():
        results[name] = compute_financial_impact(y_true, y_pred, amounts, **params)
    return results


def plot_financial_impact_visualizations(
    impact_data: Dict[str, Any],
    scenario_data: Dict[str, Dict[str, Any]],
    output_dir: str = "reports/figures",
) -> None:
    """Generate professional financial impact and scenario comparison charts."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Detected vs Missed Fraud Dollar Breakdown
    plt.figure(figsize=(8, 5))
    meas = impact_data["measured_metrics"]
    labels = ["Detected Fraud ($)", "Missed Fraud ($)", "Flagged Legit ($ FP)"]
    vals = [meas["detected_fraud_amount"], meas["missed_fraud_amount"], meas["flagged_legit_amount"]]
    colors = ["#2A9D8F", "#E63946", "#F4A261"]

    ax = sns.barplot(x=labels, y=vals, palette=colors)
    plt.title("Financial Value Breakdown by Model Classification Outcome", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Total Monetary Value ($ USD)", fontsize=11)
    for p in ax.patches:
        height = float(p.get_height())
        ax.annotate(
            f"${height:,.0f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            xytext=(0, 4),
            textcoords="offset points",
        )
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "24_financial_impact_breakdown.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[FINANCIAL] Breakdown chart saved to '{chart1_path}'.")

    # 2. Scenario Comparison (Net Financial Benefit)
    plt.figure(figsize=(9, 5))
    scen_names = list(scenario_data.keys())
    net_benefits = [scenario_data[s]["estimated_business_impact"]["net_financial_benefit_usd"] for s in scen_names]
    op_costs = [scenario_data[s]["estimated_business_impact"]["total_operational_cost_usd"] for s in scen_names]

    df_scen = pd.DataFrame({
        "Net Financial Benefit ($)": net_benefits,
        "Operational Cost ($)": op_costs,
    }, index=scen_names)

    ax = df_scen.plot(kind="bar", color=["#2A9D8F", "#E76F51"], figsize=(9, 5), edgecolor="black", linewidth=0.5)
    plt.title("Net Financial Savings & Operational Cost Across Enterprise Scenarios", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("USD ($)", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.legend(frameon=True)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "25_financial_scenarios.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[FINANCIAL] Scenario chart saved to '{chart2_path}'.")

