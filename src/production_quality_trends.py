
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "Production_Speed_units_per_hr",
    "Error_Rate_%",
    "Quality_Control_Defect_Rate_%"
]


def load_data():
    df = pd.read_csv(DATA_PATH)

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip()
        + " " + df["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    df = df.dropna(subset=["DateTime"] + METRICS).copy()
    df["Date"] = df["DateTime"].dt.normalize()

    print("\nDATA VALIDATION")
    print("-" * 50)
    print("Valid records:", len(df))
    print("Observation dates:", df["Date"].nunique())
    print("Machines:", df["Machine_ID"].nunique())

    return df


def calculate_daily_metrics(df):
    daily = (
        df.groupby("Date", as_index=False)
        .agg(
            Production_Mean=("Production_Speed_units_per_hr", "mean"),
            Production_Std=("Production_Speed_units_per_hr", "std"),
            Error_Rate_Mean=("Error_Rate_%", "mean"),
            Error_Rate_Std=("Error_Rate_%", "std"),
            Defect_Rate_Mean=("Quality_Control_Defect_Rate_%", "mean"),
            Defect_Rate_Std=("Quality_Control_Defect_Rate_%", "std"),
            Record_Count=("Machine_ID", "size")
        )
        .sort_values("Date")
    )

    return daily


def analyze_trend(daily, column):
    values = daily[column].dropna()
    positions = np.arange(len(values))

    if len(values) < 3 or values.nunique() < 2:
        return None

    result = linregress(positions, values)

    return {
        "Metric": column,
        "Observation_Days": len(values),
        "Mean": values.mean(),
        "Standard_Deviation": values.std(),
        "Minimum_Daily_Mean": values.min(),
        "Maximum_Daily_Mean": values.max(),
        "Trend_Per_Day": result.slope,
        "Trend_P_Value": result.pvalue,
        "R_Squared": result.rvalue ** 2
    }


def main():
    df = load_data()

    print("\nCALCULATING DAILY PRODUCTION AND QUALITY...")
    daily = calculate_daily_metrics(df)

    daily_path = OUTPUT_DIR / "daily_production_quality_trends.csv"
    daily.to_csv(daily_path, index=False)

    trend_columns = [
        "Production_Mean",
        "Error_Rate_Mean",
        "Defect_Rate_Mean"
    ]

    results = []

    print("\nANALYZING TEMPORAL TRENDS...")
    for column in trend_columns:
        result = analyze_trend(daily, column)
        if result:
            results.append(result)

    trends = pd.DataFrame(results)

    trend_path = OUTPUT_DIR / "production_quality_trend_statistics.csv"
    trends.to_csv(trend_path, index=False)

    print("\nDAILY METRIC SUMMARY")
    print("-" * 70)
    print(daily[trend_columns].describe().round(4).to_string())

    print("\nTEMPORAL TREND RESULTS")
    print("-" * 70)
    print(trends.round(6).to_string(index=False))

    print("\nLIMITATIONS")
    print("-" * 50)
    print("Linear trends describe association with observation order.")
    print("They do not establish operational causes.")
    print("Ordinary regression p-values may be optimistic because")
    print("daily observations can be temporally dependent.")
    print("A trend over this short dataset is not a long-term forecast.")

    print("\nFILES SAVED:")
    print(daily_path)
    print(trend_path)


if __name__ == "__main__":
    main()

