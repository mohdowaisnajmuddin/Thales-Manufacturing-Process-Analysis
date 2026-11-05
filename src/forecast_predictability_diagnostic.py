
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    ROOT / "data" / "raw"
    / "Thales_Group_Manufacturing.csv"
)

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPEED = "Production_Speed_units_per_hr"

HORIZONS = [30, 60, 120]

FEATURES = [
    SPEED,
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Predictive_Maintenance_Score",
    "Error_Rate_%"
]


def load_raw_data():
    df = pd.read_csv(RAW_FILE)

    required = [
        "Machine_ID",
        "Date",
        "Timestamp"
    ] + FEATURES

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["Combined_Timestamp"] = pd.to_datetime(
        df["Date"].astype(str)
        + " "
        + df["Timestamp"].astype(str),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Machine_ID",
            "Combined_Timestamp",
            SPEED
        ]
    ).copy()

    df = df.sort_values(
        ["Machine_ID", "Combined_Timestamp"]
    )

    return df


def audit_time_gaps(df):
    print("\n" + "=" * 70)
    print("1. MACHINE OBSERVATION TIME-GAP AUDIT")
    print("=" * 70)

    records = []

    for machine, group in df.groupby("Machine_ID"):

        timestamps = (
            group["Combined_Timestamp"]
            .sort_values()
        )

        gaps = timestamps.diff().dt.total_seconds() / 60

        gaps = gaps.dropna()

        if gaps.empty:
            continue

        records.append({
            "Machine_ID": machine,
            "observations": len(group),
            "median_gap_minutes": gaps.median(),
            "mean_gap_minutes": gaps.mean(),
            "std_gap_minutes": gaps.std(),
            "minimum_gap_minutes": gaps.min(),
            "maximum_gap_minutes": gaps.max(),
            "gap_30min_or_less_pct": (
                gaps.le(30).mean() * 100
            ),
            "gap_15_to_45min_pct": (
                gaps.between(15, 45).mean() * 100
            )
        })

    result = pd.DataFrame(records)

    print(
        result.round(3).to_string(index=False)
    )

    result.to_csv(
        REPORT_DIR / "machine_observation_time_gaps.csv",
        index=False
    )

    summary = result.drop(
        columns="Machine_ID"
    ).describe()

    print("\nAcross-machine summary:")
    print(summary.round(3).to_string())


def audit_time_based_autocorrelation(df):
    print("\n" + "=" * 70)
    print("2. TIME-BASED PRODUCTION SPEED AUTOCORRELATION")
    print("=" * 70)

    records = []

    for machine, group in df.groupby("Machine_ID"):

        group = group[
            ["Combined_Timestamp", SPEED]
        ].dropna().copy()

        group = group.sort_values(
            "Combined_Timestamp"
        )

        # Remove duplicate timestamps.
        group = group.drop_duplicates(
            subset="Combined_Timestamp",
            keep=False
        )

        group = group.set_index(
            "Combined_Timestamp"
        )

        if len(group) < 10:
            continue

        # Use nearest observation within 5 minutes
        # of each requested time interval.
        for horizon in HORIZONS:

            shifted = group[SPEED].shift(
                freq=pd.Timedelta(minutes=horizon)
            )

            aligned = pd.merge_asof(
                group.reset_index().sort_values(
                    "Combined_Timestamp"
                ),
                shifted.rename(
                    "Future_Aligned_Speed"
                ).reset_index().sort_values(
                    "Combined_Timestamp"
                ),
                on="Combined_Timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(minutes=5)
            )

            valid = aligned[
                [SPEED, "Future_Aligned_Speed"]
            ].dropna()

            if len(valid) < 10:
                correlation = np.nan
            else:
                correlation = valid[
                    SPEED
                ].corr(
                    valid["Future_Aligned_Speed"]
                )

            records.append({
                "Machine_ID": machine,
                "horizon_minutes": horizon,
                "aligned_pairs": len(valid),
                "pearson_correlation": correlation
            })

    result = pd.DataFrame(records)

    print("\nMachine-level correlations:")
    print(
        result.round(4).to_string(index=False)
    )

    summary = (
        result.groupby("horizon_minutes")[
            "pearson_correlation"
        ]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ])
    )

    print("\nAcross-machine correlation summary:")
    print(summary.round(4).to_string())

    result.to_csv(
        REPORT_DIR / "time_based_autocorrelation.csv",
        index=False
    )

    summary.to_csv(
        REPORT_DIR / "time_based_autocorrelation_summary.csv"
    )


def audit_machine_forecast_relationships():
    print("\n" + "=" * 70)
    print("3. FORECAST DATASET MACHINE-LEVEL RELATIONSHIPS")
    print("=" * 70)

    records = []

    for horizon in HORIZONS:

        path = (
            PROCESSED_DIR
            / f"forecast_dataset_{horizon}min.csv"
        )

        if not path.exists():
            print("Missing:", path)
            continue

        df = pd.read_csv(path)

        required = [
            "Machine_ID",
            "Current_Production_Speed_units_per_hr",
            "Future_Production_Speed"
        ]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            print(
                f"{horizon} minutes missing columns:",
                missing
            )
            continue

        for col in required[1:]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=required
        )

        for machine, group in df.groupby("Machine_ID"):

            if len(group) < 20:
                continue

            current = group[
                "Current_Production_Speed_units_per_hr"
            ]

            future = group[
                "Future_Production_Speed"
            ]

            records.append({
                "horizon_minutes": horizon,
                "Machine_ID": machine,
                "records": len(group),
                "pearson": current.corr(future),
                "spearman": current.corr(
                    future,
                    method="spearman"
                )
            })

    result = pd.DataFrame(records)

    if result.empty:
        print("No valid machine-level forecast records.")
        return

    print("\nMachine-level correlations:")
    print(
        result.round(4).to_string(index=False)
    )

    summary = (
        result.groupby("horizon_minutes")
        .agg(
            machines=("Machine_ID", "nunique"),
            mean_pearson=("pearson", "mean"),
            median_pearson=("pearson", "median"),
            positive_machines=(
                "pearson",
                lambda x: (x > 0).sum()
            ),
            negative_machines=(
                "pearson",
                lambda x: (x < 0).sum()
            )
        )
    )

    print("\nSummary:")
    print(summary.round(4).to_string())

    result.to_csv(
        REPORT_DIR / "machine_forecast_relationships.csv",
        index=False
    )

    summary.to_csv(
        REPORT_DIR / "machine_forecast_relationship_summary.csv"
    )


def main():
    print("LOADING RAW DATA")

    df = load_raw_data()

    print("Rows:", len(df))
    print("Machines:", df["Machine_ID"].nunique())
    print(
        "Date range:",
        df["Combined_Timestamp"].min(),
        "to",
        df["Combined_Timestamp"].max()
    )

    audit_time_gaps(df)

    audit_time_based_autocorrelation(df)

    audit_machine_forecast_relationships()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETED")
    print("Reports saved in:", REPORT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
