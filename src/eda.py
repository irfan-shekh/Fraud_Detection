"""
Exploratory Data Analysis (EDA) Module.
Generates comprehensive visual analytics and saves figures into reports/figures/.
"""

import os
import tempfile
# Configure matplotlib for non-interactive backend and writable cache
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional

from src.data_loader import load_raw_data

# Set visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def generate_all_eda_figures(
    df: Optional[pd.DataFrame] = None, output_dir: str = "reports/figures"
) -> None:
    """
    Generate all 15 required EDA visualizations and save them as PNGs.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Loaded raw transaction dataset.
    output_dir : str, default='reports/figures'
        Target directory to save figures.
    """
    if df is None:
        df = load_raw_data()

    os.makedirs(output_dir, exist_ok=True)
    print(f"[EDA] Generating exploratory visualizations in '{output_dir}'...")

    # Color constants
    fraud_color = "#E63946"     # Coral red
    legit_color = "#1D3557"     # Deep navy blue
    accent_palette = ["#457B9D", "#A8DADC", "#E63946", "#1D3557", "#2A9D8F"]

    # 1. Fraud vs Legitimate Transaction Distribution (Count)
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        x="isFraud", hue="isFraud", data=df, palette=[legit_color, fraud_color], legend=False
    )
    plt.title("1. Transaction Class Distribution (Count)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Transaction Class", fontsize=11)
    plt.ylabel("Number of Transactions", fontsize=11)
    plt.xticks([0, 1], ["Legitimate (0)", "Fraudulent (1)"])
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(
            f"{height:,}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            xytext=(0, 4),
            textcoords="offset points",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_distribution_count.png"), dpi=300)
    plt.close()

    # 2. Fraud Percentage Distribution (Pie Chart)
    plt.figure(figsize=(6, 6))
    counts = df["isFraud"].value_counts()
    plt.pie(
        counts,
        labels=["Legitimate", "Fraudulent"],
        autopct="%1.2f%%",
        startangle=140,
        colors=[legit_color, fraud_color],
        explode=(0, 0.1),
        textprops={"fontsize": 11, "fontweight": "bold", "color": "white"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    plt.title("2. Class Proportion Breakdown (%)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(["Legitimate (89.75%)", "Fraudulent (10.25%)"], loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_fraud_percentage_pie.png"), dpi=300)
    plt.close()

    # 3. Transaction Type Distribution
    plt.figure(figsize=(8, 5))
    type_counts = df["type"].value_counts()
    ax = sns.barplot(x=type_counts.index, y=type_counts.values, hue=type_counts.index, palette="Blues_r", legend=False)
    plt.title("3. Overall Transaction Type Volume", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Transaction Type", fontsize=11)
    plt.ylabel("Transaction Count", fontsize=11)
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(
            f"{height:,}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 4),
            textcoords="offset points",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_transaction_type_distribution.png"), dpi=300)
    plt.close()

    # 4. Fraud Distribution by Transaction Type
    plt.figure(figsize=(9, 5))
    ct = pd.crosstab(df["type"], df["isFraud"])
    ax = ct.plot(
        kind="bar",
        stacked=False,
        color=[legit_color, fraud_color],
        figsize=(9, 5),
        edgecolor="black",
        linewidth=0.5,
    )
    plt.title("4. Fraud vs Legitimate by Transaction Type", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Transaction Type", fontsize=11)
    plt.ylabel("Number of Transactions", fontsize=11)
    plt.legend(["Legitimate", "Fraudulent"], title="Class", frameon=True)
    plt.xticks(rotation=0)
    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(
                f"{height:,}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_fraud_by_transaction_type.png"), dpi=300)
    plt.close()

    # 5. Transaction Amount Distribution (Log Scale)
    plt.figure(figsize=(9, 5))
    sns.histplot(
        np.log10(df["amount"] + 1),
        kde=True,
        color="#2A9D8F",
        bins=50,
        edgecolor="white",
    )
    plt.title("5. Transaction Amount Distribution (Log10 Scale)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Log10(Amount in USD + 1)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_transaction_amount_distribution.png"), dpi=300)
    plt.close()

    # 6. Fraud Transaction Amount Distribution
    plt.figure(figsize=(9, 5))
    fraud_amounts = df[df["isFraud"] == 1]["amount"]
    sns.histplot(
        np.log10(fraud_amounts + 1),
        kde=True,
        color=fraud_color,
        bins=35,
        edgecolor="white",
    )
    plt.title("6. Fraudulent Transaction Amount Distribution (Log10 Scale)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Log10(Amount in USD + 1)", fontsize=11)
    plt.ylabel("Frequency (Fraud Records)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_fraud_amount_distribution.png"), dpi=300)
    plt.close()

    # 7. Legitimate Transaction Amount Distribution
    plt.figure(figsize=(9, 5))
    legit_amounts = df[df["isFraud"] == 0]["amount"]
    sns.histplot(
        np.log10(legit_amounts + 1),
        kde=True,
        color=legit_color,
        bins=50,
        edgecolor="white",
    )
    plt.title("7. Legitimate Transaction Amount Distribution (Log10 Scale)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Log10(Amount in USD + 1)", fontsize=11)
    plt.ylabel("Frequency (Legitimate Records)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_legitimate_amount_distribution.png"), dpi=300)
    plt.close()

    # 8. Transaction Amount Boxplots by Class
    plt.figure(figsize=(8, 5))
    df_plot = df.copy()
    df_plot["amount_log10"] = np.log10(df_plot["amount"] + 1)
    sns.boxplot(
        x="isFraud",
        y="amount_log10",
        hue="isFraud",
        data=df_plot,
        palette=[legit_color, fraud_color],
        width=0.4,
        legend=False,
    )
    plt.title("8. Log10 Transaction Amount Boxplot by Class", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Transaction Class", fontsize=11)
    plt.ylabel("Log10(Amount + 1)", fontsize=11)
    plt.xticks([0, 1], ["Legitimate", "Fraudulent"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_amount_boxplots.png"), dpi=300)
    plt.close()

    # 9. Correlation Heatmap
    plt.figure(figsize=(9, 7))
    num_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]
    corr_matrix = df[num_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("9. Pearson Correlation Heatmap (Raw Features)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_correlation_heatmap.png"), dpi=300)
    plt.close()

    # 10. Sender Balance Analysis (Pre vs Post Balance)
    plt.figure(figsize=(9, 5))
    plt.scatter(
        np.log10(df[df["isFraud"] == 0]["oldbalanceOrg"] + 1),
        np.log10(df[df["isFraud"] == 0]["newbalanceOrig"] + 1),
        alpha=0.3,
        color=legit_color,
        label="Legitimate",
        s=18,
    )
    plt.scatter(
        np.log10(df[df["isFraud"] == 1]["oldbalanceOrg"] + 1),
        np.log10(df[df["isFraud"] == 1]["newbalanceOrig"] + 1),
        alpha=0.7,
        color=fraud_color,
        label="Fraudulent",
        s=25,
        marker="x",
    )
    plt.title("10. Sender Old vs New Balance (Log10 Scale)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Log10(oldbalanceOrg + 1)", fontsize=11)
    plt.ylabel("Log10(newbalanceOrig + 1)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "10_sender_balance_analysis.png"), dpi=300)
    plt.close()

    # 11. Receiver Balance Analysis
    plt.figure(figsize=(9, 5))
    plt.scatter(
        np.log10(df[df["isFraud"] == 0]["oldbalanceDest"] + 1),
        np.log10(df[df["isFraud"] == 0]["newbalanceDest"] + 1),
        alpha=0.3,
        color=legit_color,
        label="Legitimate",
        s=18,
    )
    plt.scatter(
        np.log10(df[df["isFraud"] == 1]["oldbalanceDest"] + 1),
        np.log10(df[df["isFraud"] == 1]["newbalanceDest"] + 1),
        alpha=0.7,
        color=fraud_color,
        label="Fraudulent",
        s=25,
        marker="x",
    )
    plt.title("11. Receiver Old vs New Balance (Log10 Scale)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Log10(oldbalanceDest + 1)", fontsize=11)
    plt.ylabel("Log10(newbalanceDest + 1)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "11_receiver_balance_analysis.png"), dpi=300)
    plt.close()

    # 12. Fraud Transaction Patterns by Time (Step / Hour)
    plt.figure(figsize=(11, 5))
    step_fraud = df[df["isFraud"] == 1]["step"].value_counts().sort_index()
    step_legit = df[df["isFraud"] == 0]["step"].value_counts().sort_index()
    plt.plot(step_legit.index, step_legit.values, label="Legitimate Volume", color=legit_color, lw=2)
    plt.plot(step_fraud.index, step_fraud.values, label="Fraudulent Volume", color=fraud_color, lw=2.2, linestyle="--")
    plt.title("12. Transaction Velocity Across Steps (Hours 1–95)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Simulation Step (Hours)", fontsize=11)
    plt.ylabel("Transaction Count per Hour", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "12_fraud_patterns_by_time.png"), dpi=300)
    plt.close()

    # 13. Fraud Percentage by Transaction Type
    plt.figure(figsize=(8, 5))
    fraud_pct_by_type = (df.groupby("type")["isFraud"].mean() * 100).sort_values(ascending=False)
    ax = sns.barplot(x=fraud_pct_by_type.index, y=fraud_pct_by_type.values, hue=fraud_pct_by_type.index, palette="Reds_r", legend=False)
    plt.title("13. Fraud Incidence Rate (%) by Transaction Type", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Transaction Type", fontsize=11)
    plt.ylabel("Fraud Rate (%)", fontsize=11)
    for p in ax.patches:
        height = float(p.get_height())
        ax.annotate(
            f"{height:.2f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            xytext=(0, 4),
            textcoords="offset points",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "13_fraud_rate_by_type.png"), dpi=300)
    plt.close()

    # 14. Outlier Analysis Across Balance and Amount Features
    plt.figure(figsize=(10, 5))
    num_vars = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    df_melt = pd.melt(df_plot, id_vars=["isFraud"], value_vars=[v for v in num_vars if v in df_plot.columns])
    df_melt["log_value"] = np.log10(df_melt["value"] + 1)
    sns.boxplot(
        x="variable",
        y="log_value",
        hue="isFraud",
        data=df_melt,
        palette=[legit_color, fraud_color],
    )
    plt.title("14. Outlier Distribution Across Financial Variables (Log10)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Financial Feature", fontsize=11)
    plt.ylabel("Log10(Value + 1)", fontsize=11)
    plt.legend(["Legitimate", "Fraudulent"], title="Class", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "14_outlier_analysis.png"), dpi=300)
    plt.close()

    # 15. Fraud vs Non-Fraud Metric Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    comp_metrics = pd.DataFrame({
        "Mean Amount ($)": [legit_amounts.mean(), fraud_amounts.mean()],
        "Median Amount ($)": [legit_amounts.median(), fraud_amounts.median()],
        "Mean Sender Old Bal ($)": [
            df[df["isFraud"] == 0]["oldbalanceOrg"].mean(),
            df[df["isFraud"] == 1]["oldbalanceOrg"].mean(),
        ],
        "Mean Sender New Bal ($)": [
            df[df["isFraud"] == 0]["newbalanceOrig"].mean(),
            df[df["isFraud"] == 1]["newbalanceOrig"].mean(),
        ],
    }, index=["Legitimate", "Fraudulent"]).T

    ax = comp_metrics.plot(
        kind="bar",
        color=[legit_color, fraud_color],
        figsize=(10, 5),
        edgecolor="black",
        linewidth=0.5,
        logy=True,
    )
    plt.title("15. Mean & Median Financial Values Comparison (Log Scale)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Log10 Financial Value ($)", fontsize=11)
    plt.xlabel("Metric", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "15_fraud_vs_nonfraud_comparison.png"), dpi=300)
    plt.close()

    print(f"[EDA] All 15 figures successfully generated and saved to '{output_dir}'.")


if __name__ == "__main__":
    generate_all_eda_figures()
