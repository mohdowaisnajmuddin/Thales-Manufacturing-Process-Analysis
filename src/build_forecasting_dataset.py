
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/raw/Thales_Group_Manufacturing.csv"
OUT = ROOT / "data/processed"
REPORTS = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

HORIZONS = [30, 60, 120]
TOLERANCE_MINUTES = 15

TARGET_COLUMN = "Production_Speed_units_per_hr"

FEATURE_COLUMNS = [
    "Operation_Mode",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%"
]

# Historical features are calculated only from observations
# at or before the source timestamp.
HISTORY_COLUMNS = [
    "Production_Speed_units_per_hr",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Error_Rate_%",
    "Quality_Control_Defect_Rate_%"
]

LAG_COLUMNS = [
    "Production_Speed_units_per_hr",
    "Temperature_C",
    "Vibration_Hz",
    "Error_Rate_%"
]

ROLLING_WINDOWS = [30, 60, 120]


# --------------------------------------------------
# LOAD AND CLEAN DATA
# --------------------------------------------------

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Raw dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    required = [
        "Date",
        "Timestamp",
        "Machine_ID",
        TARGET_COLUMN
    ] + FEATURE_COLUMNS

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Combined_Timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip()
        + " "
        + df["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    df = df.dropna(
        subset=["Machine_ID", "Combined_Timestamp"]
    ).copy()

    # Convert sensor/process variables to numeric.
    numeric_columns = [
        column for column in FEATURE_COLUMNS
        if column != "Operation_Mode"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Exclude ambiguous duplicate machine/timestamp groups.
    duplicate_mask = df.duplicated(
        subset=["Machine_ID", "Combined_Timestamp"],
        keep=False
    )

    duplicate_rows = df[duplicate_mask].copy()
    clean = df[~duplicate_mask].copy()

    duplicate_rows.to_csv(
        REPORTS / "forecast_excluded_duplicate_rows.csv",
        index=False
    )

    print("Duplicate machine/timestamp rows excluded:", len(duplicate_rows))

    return clean


# --------------------------------------------------
# CREATE HISTORICAL FEATURES
# --------------------------------------------------

def add_historical_features(group):
    group = group.sort_values(
        "Combined_Timestamp"
    ).copy()

    group = group.set_index("Combined_Timestamp")

    # Previous observations.
    for column in LAG_COLUMNS:
        group[f"Previous_1_{column}"] = group[column].shift(1)
        group[f"Previous_2_{column}"] = group[column].shift(2)

    # Time since previous observation.
    timestamps = group.index.to_series()

    group["Minutes_Since_Previous_Observation"] = (
        timestamps.diff().dt.total_seconds() / 60
    )

    # Time-based rolling features.
    # closed="both" includes the current source observation
    # and earlier observations, never future observations.
    for window in ROLLING_WINDOWS:
        rolling = group[HISTORY_COLUMNS].rolling(
            window=f"{window}min",
            min_periods=1,
            closed="both"
        )

        for column in HISTORY_COLUMNS:
            group[f"Rolling_{window}min_Mean_{column}"] = (
                rolling[column].mean()
            )

            group[f"Rolling_{window}min_Std_{column}"] = (
                rolling[column].std()
            )

        # Number of observations available in this time window.
        group[f"Rolling_{window}min_Observation_Count"] = (
            group[TARGET_COLUMN]
            .rolling(
                window=f"{window}min",
                min_periods=1,
                closed="both"
            )
            .count()
        )

    group = group.reset_index()

    return group


def prepare_machine_groups(df):
    machine_groups = {}

    for machine_id, original_group in df.groupby("Machine_ID"):
        group = add_historical_features(original_group)

        machine_groups[machine_id] = group

    return machine_groups


# --------------------------------------------------
# BUILD FUTURE TARGET PAIRS
# --------------------------------------------------

def build_pairs(machine_groups, horizon):
    all_pairs = []

    horizon_ns = int(horizon * 60 * 1e9)
    tolerance_ns = int(TOLERANCE_MINUTES * 60 * 1e9)

    for machine_id, group in machine_groups.items():

        times = (
            group["Combined_Timestamp"]
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
        )

        candidates = []

        for source_idx, source_ns in enumerate(times):

            desired_ns = source_ns + horizon_ns

            lower = desired_ns - tolerance_ns
            upper = desired_ns + tolerance_ns

            start = np.searchsorted(
                times, lower, side="left"
            )

            end = np.searchsorted(
                times, upper, side="right"
            )

            if start >= end:
                continue

            future_indices = np.arange(start, end)

            # Future target must be strictly after the source.
            future_indices = future_indices[
                times[future_indices] > source_ns
            ]

            if len(future_indices) == 0:
                continue

            differences = np.abs(
                times[future_indices] - desired_ns
            )

            best_position = np.argmin(differences)

            target_idx = int(
                future_indices[best_position]
            )

            target_ns = times[target_idx]

            candidates.append({
                "source_idx": source_idx,
                "target_idx": target_idx,
                "source_ns": source_ns,
                "target_ns": target_ns,
                "deviation_ns": abs(
                    target_ns - desired_ns
                )
            })

        if not candidates:
            continue

        candidate_df = pd.DataFrame(candidates)

        # One source observation per future target timestamp.
        candidate_df = candidate_df.sort_values(
            ["deviation_ns", "source_ns"]
        )

        candidate_df = candidate_df.drop_duplicates(
            subset=["target_ns"],
            keep="first"
        )

        for _, pair in candidate_df.iterrows():

            source = group.iloc[int(pair["source_idx"])]
            target = group.iloc[int(pair["target_idx"])]

            record = {
                "Machine_ID": machine_id,
                "Source_Timestamp": source["Combined_Timestamp"],
                "Target_Timestamp": target["Combined_Timestamp"],
                "Actual_Lead_Minutes": (
                    target["Combined_Timestamp"]
                    - source["Combined_Timestamp"]
                ).total_seconds() / 60,

                "Forecast_Horizon_Minutes": horizon,

                "Future_Production_Speed": target[TARGET_COLUMN]
            }

            # Current source-time features.
            for column in FEATURE_COLUMNS:
                record[f"Current_{column}"] = source[column]

            # Historical features calculated at source time.
            for column in group.columns:
                if (
                    column.startswith("Previous_")
                    or column.startswith("Rolling_")
                    or column == "Minutes_Since_Previous_Observation"
                ):
                    record[f"History_{column}"] = source[column]

            all_pairs.append(record)

    return pd.DataFrame(all_pairs)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    df = load_data()

    print("\nCLEAN SOURCE DATA")
    print("Rows:", len(df))
    print("Machines:", df["Machine_ID"].nunique())

    machine_groups = prepare_machine_groups(df)

    summaries = []

    for horizon in HORIZONS:

        pairs = build_pairs(
            machine_groups,
            horizon
        )

        if pairs.empty:
            print(f"No pairs created for {horizon}-minute horizon.")
            continue

        output_path = (
            OUT / f"forecast_dataset_{horizon}min.csv"
        )

        pairs.to_csv(
            output_path,
            index=False
        )

        machine_counts = pairs.groupby(
            "Machine_ID"
        ).size()

        summary = {
            "horizon_minutes": horizon,
            "unique_pairs": len(pairs),
            "unique_machines": pairs["Machine_ID"].nunique(),

            "median_actual_lead_minutes": (
                pairs["Actual_Lead_Minutes"].median()
            ),

            "mean_actual_lead_minutes": (
                pairs["Actual_Lead_Minutes"].mean()
            ),

            "minimum_pairs_per_machine": machine_counts.min(),
            "maximum_pairs_per_machine": machine_counts.max(),

            "historical_feature_count": sum(
                column.startswith("History_")
                for column in pairs.columns
            )
        }

        summaries.append(summary)

        print(f"\n{horizon}-MINUTE DATASET")
        print("Pairs:", len(pairs))
        print("Machines:", pairs["Machine_ID"].nunique())
        print(
            "Median actual lead:",
            round(pairs["Actual_Lead_Minutes"].median(), 2)
        )
        print(
            "Historical features:",
            summary["historical_feature_count"]
        )
        print("Saved:", output_path)

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        REPORTS / "forecast_dataset_build_summary.csv",
        index=False
    )

    print("\nFINAL DATASET SUMMARY")
    print(summary_df.round(2).to_string(index=False))

    print("\nReports saved in:", REPORTS)


if __name__ == "__main__":
    main()
