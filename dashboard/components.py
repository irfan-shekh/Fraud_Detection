"""
Dashboard UI Components and Plotly Visualizations.
Houses custom styles, metric cards, and interactive charts for Streamlit.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def apply_custom_styles():
    """Apply modern financial analytics CSS styling."""
    st.markdown(
        """
        <style>
        /* Main background & typography */
        .main {
            background-color: #f8f9fa;
        }
        /* Top header branding */
        .brand-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.5px;
            margin-bottom: 0px;
        }
        .brand-subtitle {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 25px;
        }
        /* Metric card styling */
        .metric-card {
            background: #ffffff;
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.05);
            border-left: 4px solid #1D3557;
            margin-bottom: 12px;
        }
        .metric-label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 4px;
        }
        .metric-subtext {
            font-size: 0.78rem;
            color: #94a3b8;
            margin-top: 2px;
        }
        /* Risk Alert Box */
        .risk-banner {
            border-radius: 8px;
            padding: 16px 20px;
            margin: 15px 0px;
            font-weight: 600;
        }
        .risk-critical {
            background-color: #fee2e2;
            color: #991b1b;
            border-left: 5px solid #dc2626;
        }
        .risk-minimal {
            background-color: #ecfdf5;
            color: #065f46;
            border-left: 5px solid #059669;
        }
        .disclaimer-box {
            background-color: #f1f5f9;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.8rem;
            color: #64748b;
            border: 1px solid #e2e8f0;
            margin-top: 25px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str, value: str, subtext: str = "", border_color: str = "#1D3557"
):
    """Render a styled financial metric card."""
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color: {border_color};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_fraud_distribution_pie(df: pd.DataFrame) -> go.Figure:
    """Create interactive donut chart for fraud vs legit distribution."""
    counts = df["isFraud"].value_counts()
    labels = ["Legitimate", "Fraudulent"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=["#1D3557", "#E63946"]),
                textinfo="label+percent",
                hoverinfo="value+percent",
            )
        ]
    )
    fig.update_layout(
        title="Transaction Class Composition",
        margin=dict(t=40, b=10, l=10, r=10),
        height=320,
        showlegend=False,
    )
    return fig


def create_type_fraud_bar(df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart for transactions by type and fraud status."""
    ct = pd.crosstab(df["type"], df["isFraud"]).reset_index()
    ct.columns = ["type", "Legitimate", "Fraudulent"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Legitimate",
            x=ct["type"],
            y=ct["Legitimate"],
            marker_color="#1D3557",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Fraudulent",
            x=ct["type"],
            y=ct["Fraudulent"],
            marker_color="#E63946",
        )
    )

    fig.update_layout(
        title="Volume by Transaction Type & Class",
        barmode="group",
        xaxis_title="Transaction Channel",
        yaxis_title="Count",
        margin=dict(t=40, b=10, l=10, r=10),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_hourly_trend_line(df: pd.DataFrame) -> go.Figure:
    """Create time-series line chart of transaction velocity across steps."""
    step_counts = (
        df.groupby(["step", "isFraud"]).size().unstack(fill_value=0).reset_index()
    )
    if 0 not in step_counts:
        step_counts[0] = 0
    if 1 not in step_counts:
        step_counts[1] = 0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=step_counts["step"],
            y=step_counts[0],
            name="Legitimate",
            line=dict(color="#1D3557", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=step_counts["step"],
            y=step_counts[1],
            name="Fraudulent",
            line=dict(color="#E63946", width=2.5, dash="dot"),
        )
    )

    fig.update_layout(
        title="Hourly Transaction Velocity (Steps 1–95)",
        xaxis_title="Time Step (Hours)",
        yaxis_title="Hourly Volume",
        margin=dict(t=40, b=10, l=10, r=10),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_risk_gauge(probability: float) -> go.Figure:
    """Create a sleek speedometer gauge for live fraud probability."""
    prob_pct = probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={"suffix": "%", "font": {"size": 28, "color": "#0f172a"}},
            title={"text": "Fraud Probability", "font": {"size": 16, "color": "#475569"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "gray"},
                "bar": {"color": "#E63946" if prob_pct >= 50 else "#2A9D8F", "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 20], "color": "#ecfdf5"},
                    {"range": [20, 50], "color": "#fef9c3"},
                    {"range": [50, 80], "color": "#ffedd5"},
                    {"range": [80, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=240, margin=dict(t=30, b=10, l=20, r=20))
    return fig


def create_dynamic_financial_breakdown_chart(
    gross_prevented: float,
    total_review_cost: float,
    friction_cost: float,
    net_benefit: float,
    missed_fraud_amt: float,
) -> go.Figure:
    """Create an interactive Plotly bar chart dynamically reacting to simulator parameters."""
    categories = [
        "Gross Fraud Intercepted",
        "Operational Review Cost",
        "FP Customer Friction",
        "Net Financial Benefit",
        "Missed Fraud Loss",
    ]
    values = [
        gross_prevented,
        total_review_cost,
        friction_cost,
        net_benefit,
        missed_fraud_amt,
    ]
    colors = [
        "#2A9D8F",  # Teal green for gross prevented
        "#E76F51",  # Orange red for review cost
        "#F4A261",  # Orange for friction cost
        "#1B4332",  # Deep emerald for net benefit
        "#E63946",  # Red for missed loss
    ]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"${v:,.0f}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Live Financial Impact Breakdown ($ USD)",
        xaxis_title="Financial Component",
        yaxis_title="USD ($)",
        margin=dict(t=40, b=30, l=20, r=20),
        height=360,
    )
    return fig


def create_dynamic_financial_scenario_comparison_chart(
    custom_benefit: float,
    custom_cost: float,
    custom_recovery_pct: float,
) -> go.Figure:
    """Compare benchmark scenarios with the user's custom live slider configuration."""
    scenarios = [
        "High Efficiency (90% Recov)",
        "Base Case (80% Recov)",
        "Conservative (70% Recov)",
        f"⚡ Live Custom ({custom_recovery_pct:.0f}% Recov)",
    ]
    net_benefits = [
        211884067.54,
        188337610.03,
        164787747.53,
        custom_benefit,
    ]
    op_costs = [
        3405.00,
        6810.00,
        13620.00,
        custom_cost,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Net Financial Benefit ($)",
            x=scenarios,
            y=net_benefits,
            marker_color=["#52B788", "#2A9D8F", "#1D3557", "#0077B6"],
            text=[f"${v/1e6:,.1f}M" for v in net_benefits],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Operational Review Cost ($)",
            x=scenarios,
            y=op_costs,
            marker_color="#F4A261",
            text=[f"${v:,.0f}" for v in op_costs],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Enterprise Scenarios vs. Your Live Custom Configuration",
        barmode="group",
        xaxis_title="Scenario",
        yaxis_title="USD ($)",
        margin=dict(t=40, b=30, l=20, r=20),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

