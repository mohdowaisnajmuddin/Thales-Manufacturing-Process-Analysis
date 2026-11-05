
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
OUTPUT_DIR = ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DARK_TEMPLATE = "plotly_dark"

METRICS = [
    "Production_Speed_units_per_hr",
    "Error_Rate_%",
    "Quality_Control_Defect_Rate_%"
]


def load_data():
    df = pd.read_csv(DATA_PATH)

    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip()
        + " "
        + df["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    df = df.dropna(
        subset=["DateTime", "Machine_ID", "Operation_Mode"] + METRICS
    ).copy()

    df["Date"] = df["DateTime"].dt.normalize()

    print("\nDATA VALIDATION")
    print("-" * 50)
    print("Rows:", len(df))
    print("Machines:", df["Machine_ID"].nunique())
    print("Observation days:", df["Date"].nunique())
    print("Operation modes:", df["Operation_Mode"].nunique())

    return df


def calculate_daily_metrics(df):
    daily = (
        df.groupby("Date", as_index=False)
        .agg(
            Production_Mean=("Production_Speed_units_per_hr", "mean"),
            Error_Rate_Mean=("Error_Rate_%", "mean"),
            Defect_Rate_Mean=("Quality_Control_Defect_Rate_%", "mean"),
            Record_Count=("Machine_ID", "size")
        )
        .sort_values("Date")
    )

    return daily


def save_figure(fig, filename):
    output_path = OUTPUT_DIR / filename

    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True
    )

    print("Saved:", output_path)


def plot_daily_production(daily):
    fig = px.line(
        daily,
        x="Date",
        y="Production_Mean",
        markers=True,
        title="Daily Average Production Speed",
        labels={
            "Date": "Date",
            "Production_Mean": "Production Speed (units/hour)"
        },
        template=DARK_TEMPLATE
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Observation Date",
        yaxis_title="Production Speed (units/hour)"
    )

    save_figure(fig, "daily_production_speed.html")


def plot_daily_error_rate(daily):
    fig = px.line(
        daily,
        x="Date",
        y="Error_Rate_Mean",
        markers=True,
        title="Daily Average Error Rate",
        labels={
            "Date": "Date",
            "Error_Rate_Mean": "Error Rate (%)"
        },
        template=DARK_TEMPLATE
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Observation Date",
        yaxis_title="Error Rate (%)"
    )

    save_figure(fig, "daily_error_rate.html")


def plot_daily_defect_rate(daily):
    fig = px.line(
        daily,
        x="Date",
        y="Defect_Rate_Mean",
        markers=True,
        title="Daily Average Quality-Control Defect Rate",
        labels={
            "Date": "Date",
            "Defect_Rate_Mean": "Defect Rate (%)"
        },
        template=DARK_TEMPLATE
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Observation Date",
        yaxis_title="Defect Rate (%)"
    )

    save_figure(fig, "daily_defect_rate.html")


def plot_production_vs_defects(df):
    fig = px.scatter(
        df,
        x="Production_Speed_units_per_hr",
        y="Quality_Control_Defect_Rate_%",
        color="Operation_Mode",
        opacity=0.45,
        title="Production Speed vs Quality-Control Defect Rate",
        labels={
            "Production_Speed_units_per_hr": "Production Speed (units/hour)",
            "Quality_Control_Defect_Rate_%": "Defect Rate (%)",
            "Operation_Mode": "Operation Mode"
        },
        template=DARK_TEMPLATE
    )

    fig.update_layout(
        xaxis_title="Production Speed (units/hour)",
        yaxis_title="Defect Rate (%)",
        legend_title="Operation Mode"
    )

    save_figure(fig, "production_vs_defect_rate.html")


def plot_operation_mode_comparison(df):
    mode_summary = (
        df.groupby("Operation_Mode", as_index=False)
        .agg(
            Production_Mean=("Production_Speed_units_per_hr", "mean"),
            Error_Rate_Mean=("Error_Rate_%", "mean"),
            Defect_Rate_Mean=("Quality_Control_Defect_Rate_%", "mean")
        )
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Production Speed",
            "Error Rate",
            "Defect Rate"
        )
    )

    charts = [
        ("Production_Mean", "Production (units/hour)"),
        ("Error_Rate_Mean", "Error Rate (%)"),
        ("Defect_Rate_Mean", "Defect Rate (%)")
    ]

    for col_index, (metric, label) in enumerate(charts, start=1):
        fig.add_trace(
            go.Bar(
                x=mode_summary["Operation_Mode"],
                y=mode_summary[metric],
                name=label,
                showlegend=False
            ),
            row=1,
            col=col_index
        )

        fig.update_yaxes(
            title_text=label,
            row=1,
            col=col_index
        )

    fig.update_layout(
        title="Production and Quality Comparison by Operation Mode",
        template=DARK_TEMPLATE,
        height=550
    )

    save_figure(fig, "operation_mode_comparison.html")


def main():
    df = load_data()

    print("\nCALCULATING DAILY METRICS...")
    daily = calculate_daily_metrics(df)

    daily.to_csv(
        OUTPUT_DIR / "daily_metrics_for_visualization.csv",
        index=False
    )

    print("\nGENERATING CHARTS...")
    plot_daily_production(daily)
    plot_daily_error_rate(daily)
    plot_daily_defect_rate(daily)
    plot_production_vs_defects(df)
    plot_operation_mode_comparison(df)

    print("\nVISUALIZATION COMPLETE")
    print("-" * 50)
    print("Charts generated: 5")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
