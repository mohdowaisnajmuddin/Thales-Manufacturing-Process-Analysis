
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPEED = "Production_Speed_units_per_hr"

FEATURES = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Predictive_Maintenance_Score",
    "Error_Rate_%"
]


def distribution_audit(df):
    print("\n" + "=" * 65)
    print("DISTRIBUTION AUDIT")
    print("=" * 65)

    records = []

    columns = [SPEED] + FEATURES

    for column in columns:
        values = pd.to_numeric(
            df[column], errors="coerce"
        ).dropna()

        counts, _ = np.histogram(
            values,
            bins=10
        )

        expected = len(values) / 10

        chi_square = np.sum(
            (counts - expected) ** 2 / expected
        )

        records.append({
            "feature": column,
            "count": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "q25": values.quantile(0.25),
            "median": values.median(),
            "q75": values.quantile(0.75),
            "max": values.max(),
            "10_bin_chi_square": chi_square,
            "bin_count_min": counts.min(),
            "bin_count_max": counts.max()
        })

    result = pd.DataFrame(records)

    print(result.round(4).to_string(index=False))

    result.to_csv(
        REPORT_DIR / "data_distribution_audit.csv",
        index=False
    )


def temporal_autocorrelation_audit(df):
    print("\n" + "=" * 65)
    print("WITHIN-MACHINE TEMPORAL AUTOCORRELATION")
    print("=" * 65)

    columns = [SPEED] + FEATURES

    records = []

    for machine, group in df.groupby("Machine_ID"):
        group = group.sort_values("Combined_Timestamp")

        for column in columns:
            series = group[column].dropna()

            if len(series) < 10:
                continue

            records.append({
                "Machine_ID": machine,
                "feature": column,
                "observations": len(series),
                "lag1_autocorrelation": series.autocorr(lag=1),
                "lag2_autocorrelation": series.autocorr(lag=2),
                "lag5_autocorrelation": series.autocorr(lag=5)
            })

    result = pd.DataFrame(records)

    summary = (
        result.groupby("feature")[
            [
                "lag1_autocorrelation",
                "lag2_autocorrelation",
                "lag5_autocorrelation"
            ]
        ]
        .agg(["mean", "median", "std", "min", "max"])
    )

    print(summary.round(4).to_string())

    result.to_csv(
        REPORT_DIR / "machine_temporal_autocorrelation.csv",
        index=False
    )

    summary.to_csv(
        REPORT_DIR / "temporal_autocorrelation_summary.csv"
    )


def within_machine_relationship_audit(df):
    print("\n" + "=" * 65)
    print("WITHIN-MACHINE SENSOR RELATIONSHIPS")
    print("=" * 65)

    records = []

    for feature in FEATURES:
        temp = df[["Machine_ID", SPEED, feature]].copy()

        # Remove machine-level averages to focus on
        # variation within each individual machine.
        temp["speed_centered"] = (
            temp[SPEED]
            - temp.groupby("Machine_ID")[SPEED].transform("mean")
        )

        temp["feature_centered"] = (
            temp[feature]
            - temp.groupby("Machine_ID")[feature].transform("mean")
        )

        correlation = temp[
            ["speed_centered", "feature_centered"]
        ].corr().iloc[0, 1]

        records.append({
            "feature": feature,
            "within_machine_correlation": correlation
        })

    result = pd.DataFrame(records)

    print(
        result.sort_values(
            "within_machine_correlation",
            key=lambda x: x.abs(),
            ascending=False
        ).round(5).to_string(index=False)
    )

    result.to_csv(
        REPORT_DIR / "within_machine_sensor_relationships.csv",
        index=False
    )


def main():
    df = pd.read_csv(RAW_FILE)

    df["Combined_Timestamp"] = pd.to_datetime(
        df["Date"].astype(str)
        + " "
        + df["Timestamp"].astype(str),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    required = [
        "Machine_ID",
        "Combined_Timestamp",
        SPEED
    ] + FEATURES

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}"
        )

    for column in [SPEED] + FEATURES:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(
        subset=required
    ).copy()

    df = df.sort_values(
        ["Machine_ID", "Combined_Timestamp"]
    )

    print("Rows audited:", len(df))
    print("Machines:", df["Machine_ID"].nunique())
    print("Date range:",
          df["Combined_Timestamp"].min(),
          "to",
          df["Combined_Timestamp"].max())

    distribution_audit(df)
    temporal_autocorrelation_audit(df)
    within_machine_relationship_audit(df)

    print("\nAll audit reports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
