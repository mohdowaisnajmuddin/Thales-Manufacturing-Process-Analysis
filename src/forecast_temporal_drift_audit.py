
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
    "Current_Temperature_C",
    "Current_Vibration_Hz",
    "Current_Power_Consumption_kW",
    "Current_Network_Latency_ms",
    "Current_Packet_Loss_%",
    "Current_Quality_Control_Defect_Rate_%",
    "Current_Production_Speed_units_per_hr",
    "Current_Predictive_Maintenance_Score",
    "Current_Error_Rate_%"
]


def load_data(horizon):
    path = DATA_DIR / f"forecast_dataset_{horizon}min.csv"
    df = pd.read_csv(path)

    df["Source_Timestamp"] = pd.to_datetime(
        df["Source_Timestamp"], errors="coerce"
    )

    for column in [TARGET] + FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["Source_Timestamp", "Machine_ID", TARGET]
    ).copy()

    return df



def assign_periods(df):
    # Normalize timestamp columns before comparing them.
    timestamp_columns = ["Source_Timestamp", "Target_Timestamp"]

    for column in timestamp_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    # Remove rows with invalid timestamps.
    df = df.dropna(
        subset=timestamp_columns
    ).copy()

    # Build chronological boundaries from source timestamps.
    unique_times = np.sort(
        df["Source_Timestamp"].unique()
    )

    if len(unique_times) < 3:
        raise ValueError(
            "Not enough unique timestamps to create chronological splits."
        )

    train_end = unique_times[int(len(unique_times) * 0.70)]
    validation_end = unique_times[int(len(unique_times) * 0.85)]

    train = df[
        (df["Source_Timestamp"] < train_end)
        & (df["Target_Timestamp"] < train_end)
    ].copy()

    validation = df[
        (df["Source_Timestamp"] >= train_end)
        & (df["Source_Timestamp"] < validation_end)
        & (df["Target_Timestamp"] < validation_end)
    ].copy()

    test = df[
        df["Source_Timestamp"] >= validation_end
    ].copy()

    train["Period"] = "Train"
    validation["Period"] = "Validation"
    test["Period"] = "Test"

    result = pd.concat(
        [train, validation, test],
        ignore_index=True
    )

    print("\nSplit record counts:")
    print(result["Period"].value_counts())

    return result



def main():
    all_summaries = []
    all_correlations = []
    all_machine_summaries = []

    for horizon in HORIZONS:
        print("\n" + "=" * 65)
        print(f"{horizon}-MINUTE TEMPORAL DRIFT AUDIT")
        print("=" * 65)

        df = assign_periods(load_data(horizon))

        # Overall target and feature distribution by period.
        summary = df.groupby("Period").agg(
            records=(TARGET, "size"),
            target_mean=(TARGET, "mean"),
            target_std=(TARGET, "std"),
            target_median=(TARGET, "median"),
            target_min=(TARGET, "min"),
            target_max=(TARGET, "max")
        ).reset_index()

        summary.insert(0, "horizon_minutes", horizon)
        all_summaries.append(summary)

        print("\nTarget distribution:")
        print(summary.round(4).to_string(index=False))

        # Compare feature-target correlations by period.
        for period, group in df.groupby("Period"):
            for feature in FEATURES:
                valid = group[[feature, TARGET]].dropna()

                pearson = valid[feature].corr(
                    valid[TARGET], method="pearson"
                )

                spearman = valid[feature].corr(
                    valid[TARGET], method="spearman"
                )

                all_correlations.append({
                    "horizon_minutes": horizon,
                    "period": period,
                    "feature": feature,
                    "records": len(valid),
                    "pearson_correlation": pearson,
                    "spearman_correlation": spearman
                })

        # Machine-level target changes.
        machine_summary = df.groupby(
            ["Period", "Machine_ID"]
        ).agg(
            records=(TARGET, "size"),
            production_mean=(TARGET, "mean"),
            production_std=(TARGET, "std")
        ).reset_index()

        machine_summary.insert(0, "horizon_minutes", horizon)
        all_machine_summaries.append(machine_summary)

        print("\nAverage production speed by period:")
        print(
            machine_summary.groupby("Period")[
                "production_mean"
            ].agg(["mean", "std", "min", "max"])
            .round(4)
            .to_string()
        )

    summary_df = pd.concat(all_summaries, ignore_index=True)
    correlation_df = pd.DataFrame(all_correlations)
    machine_df = pd.concat(all_machine_summaries, ignore_index=True)

    summary_df.to_csv(
        REPORT_DIR / "forecast_temporal_target_summary.csv",
        index=False
    )

    correlation_df.to_csv(
        REPORT_DIR / "forecast_temporal_correlations.csv",
        index=False
    )

    machine_df.to_csv(
        REPORT_DIR / "forecast_temporal_machine_summary.csv",
        index=False
    )

    print("\nSaved:")
    print(REPORT_DIR / "forecast_temporal_target_summary.csv")
    print(REPORT_DIR / "forecast_temporal_correlations.csv")
    print(REPORT_DIR / "forecast_temporal_machine_summary.csv")


if __name__ == "__main__":
    main()
