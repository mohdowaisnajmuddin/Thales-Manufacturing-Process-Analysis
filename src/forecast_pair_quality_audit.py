
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [30, 60, 120]
TOLERANCE_MINUTES = 15


def audit_dataset(horizon):
    path = DATA_DIR / f"forecast_dataset_{horizon}min.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        "Machine_ID",
        "Source_Timestamp",
        "Target_Timestamp",
        "Actual_Lead_Minutes",
        "Forecast_Horizon_Minutes",
        "Future_Production_Speed"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{horizon} min dataset missing: {missing}")

    df["Source_Timestamp"] = pd.to_datetime(
        df["Source_Timestamp"], errors="coerce"
    )
    df["Target_Timestamp"] = pd.to_datetime(
        df["Target_Timestamp"], errors="coerce"
    )
    df["Actual_Lead_Minutes"] = pd.to_numeric(
        df["Actual_Lead_Minutes"], errors="coerce"
    )

    invalid = df[required].isna().any(axis=1)
    invalid_count = int(invalid.sum())
    df = df.loc[~invalid].copy()

    df["Lead_Deviation_Minutes"] = (
        df["Actual_Lead_Minutes"] - horizon
    ).abs()

    # A target timestamp must not be reused within a machine.
    target_usage = (
        df.groupby(["Machine_ID", "Target_Timestamp"])
        .size()
    )

    reused_targets = target_usage[target_usage > 1]

    # Source timestamps should be unique within a machine.
    source_usage = (
        df.groupby(["Machine_ID", "Source_Timestamp"])
        .size()
    )

    repeated_sources = source_usage[source_usage > 1]

    future_order_violations = (
        df["Target_Timestamp"] <= df["Source_Timestamp"]
    ).sum()

    tolerance_violations = (
        df["Lead_Deviation_Minutes"] > TOLERANCE_MINUTES
    ).sum()

    machine_counts = df.groupby("Machine_ID").size()

    summary = {
        "horizon_minutes": horizon,
        "valid_records": len(df),
        "invalid_records_removed": invalid_count,
        "machines": df["Machine_ID"].nunique(),
        "unique_machine_target_pairs": (
            df[["Machine_ID", "Target_Timestamp"]]
            .drop_duplicates()
            .shape[0]
        ),
        "reused_target_groups": len(reused_targets),
        "repeated_source_groups": len(repeated_sources),
        "future_order_violations": int(future_order_violations),
        "tolerance_violations": int(tolerance_violations),
        "minimum_lead_minutes": df["Actual_Lead_Minutes"].min(),
        "median_lead_minutes": df["Actual_Lead_Minutes"].median(),
        "maximum_lead_minutes": df["Actual_Lead_Minutes"].max(),
        "minimum_pairs_per_machine": machine_counts.min(),
        "maximum_pairs_per_machine": machine_counts.max(),
        "mean_pairs_per_machine": machine_counts.mean()
    }

    machine_summary = (
        df.groupby("Machine_ID")
        .agg(
            records=("Machine_ID", "size"),
            unique_targets=("Target_Timestamp", "nunique"),
            median_lead=("Actual_Lead_Minutes", "median"),
            minimum_lead=("Actual_Lead_Minutes", "min"),
            maximum_lead=("Actual_Lead_Minutes", "max")
        )
        .reset_index()
    )

    print(f"\n{'=' * 65}")
    print(f"{horizon}-MINUTE FINAL DATASET AUDIT")
    print(f"{'=' * 65}")

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary, machine_summary


def main():
    summaries = []
    machine_reports = []

    for horizon in HORIZONS:
        summary, machine_summary = audit_dataset(horizon)

        summaries.append(summary)

        machine_summary["horizon_minutes"] = horizon
        machine_reports.append(machine_summary)

    summary_df = pd.DataFrame(summaries)
    machine_df = pd.concat(machine_reports, ignore_index=True)

    summary_df.to_csv(
        REPORT_DIR / "forecast_final_dataset_quality.csv",
        index=False
    )

    machine_df.to_csv(
        REPORT_DIR / "forecast_final_dataset_machine_quality.csv",
        index=False
    )

    print("\nFINAL COMPARISON")
    print(summary_df.round(3).to_string(index=False))

    print("\nSaved:")
    print(REPORT_DIR / "forecast_final_dataset_quality.csv")
    print(REPORT_DIR / "forecast_final_dataset_machine_quality.csv")


if __name__ == "__main__":
    main()
