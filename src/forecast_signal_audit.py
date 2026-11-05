
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [30, 60, 120]

TARGET = "Future_Production_Speed"

FEATURES = [
    "Current_Production_Speed_units_per_hr",
    "Current_Temperature_C",
    "Current_Vibration_Hz",
    "Current_Power_Consumption_kW",
    "Current_Network_Latency_ms",
    "Current_Packet_Loss_%",
    "Current_Quality_Control_Defect_Rate_%",
    "Current_Predictive_Maintenance_Score",
    "Current_Error_Rate_%"
]


def audit_horizon(horizon):
    path = DATA_DIR / f"forecast_dataset_{horizon}min.csv"

    print(f"\n{'=' * 65}")
    print(f"{horizon}-MINUTE SIGNAL AUDIT")
    print(f"{'=' * 65}")

    df = pd.read_csv(path)

    print("\nDataset shape:", df.shape)
    print("\nAvailable columns:")
    print(df.columns.tolist())

    required_columns = [TARGET] + FEATURES + ["Machine_ID"]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print("\nMISSING COLUMNS:", missing)
        return None, None

    for column in [TARGET] + FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    ).copy()

    print("\nRows after numeric validation:", len(df))

    print("\nCURRENT VS FUTURE SPEED")
    print(
        df[
            [
                "Current_Production_Speed_units_per_hr",
                TARGET
            ]
        ].describe().round(3).to_string()
    )

    # Overall correlations
    records = []

    for feature in FEATURES:
        valid = df[[feature, TARGET]].dropna()

        records.append({
            "horizon_minutes": horizon,
            "feature": feature,
            "pearson_correlation": valid[feature].corr(
                valid[TARGET],
                method="pearson"
            ),
            "spearman_correlation": valid[feature].corr(
                valid[TARGET],
                method="spearman"
            ),
            "records": len(valid)
        })

    correlation_df = pd.DataFrame(records)

    print("\nFEATURE CORRELATION WITH FUTURE SPEED")
    print(
        correlation_df.assign(
            abs_pearson=correlation_df[
                "pearson_correlation"
            ].abs()
        ).sort_values(
            "abs_pearson",
            ascending=False
        ).drop(
            columns="abs_pearson"
        ).round(4).to_string(index=False)
    )

    # Correlations within individual machines
    machine_records = []

    for machine, group in df.groupby("Machine_ID"):
        if len(group) < 10:
            continue

        machine_records.append({
            "horizon_minutes": horizon,
            "Machine_ID": machine,
            "records": len(group),
            "current_future_pearson": group[
                "Current_Production_Speed_units_per_hr"
            ].corr(group[TARGET], method="pearson"),
            "current_future_spearman": group[
                "Current_Production_Speed_units_per_hr"
            ].corr(group[TARGET], method="spearman")
        })

    machine_df = pd.DataFrame(machine_records)

    print("\nCURRENT-FUTURE SPEED CORRELATION BY MACHINE")
    print(
        machine_df[
            [
                "current_future_pearson",
                "current_future_spearman"
            ]
        ].describe().round(4).to_string()
    )

    # Difference between current and future speed
    print("\nSOURCE-TARGET SPEED DIFFERENCE")

    difference = (
        df[TARGET]
        - df["Current_Production_Speed_units_per_hr"]
    )

    print(difference.describe().round(3).to_string())

    # Save results
    correlation_df.to_csv(
        REPORT_DIR /
        f"forecast_signal_correlations_{horizon}min.csv",
        index=False
    )

    machine_df.to_csv(
        REPORT_DIR /
        f"forecast_machine_correlations_{horizon}min.csv",
        index=False
    )

    return correlation_df, machine_df


def main():
    all_correlations = []

    for horizon in HORIZONS:
        correlations, machines = audit_horizon(horizon)

        if correlations is not None:
            all_correlations.append(correlations)

    if all_correlations:
        combined = pd.concat(
            all_correlations,
            ignore_index=True
        )

        combined.to_csv(
            REPORT_DIR / "forecast_signal_audit_all_horizons.csv",
            index=False
        )

        print("\nCOMBINED FEATURE CORRELATIONS")
        print(
            combined.sort_values(
                ["horizon_minutes", "pearson_correlation"],
                key=lambda x: (
                    x.abs()
                    if x.name == "pearson_correlation"
                    else x
                ),
                ascending=False
            ).round(4).to_string(index=False)
        )

        print("\nAudit reports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
