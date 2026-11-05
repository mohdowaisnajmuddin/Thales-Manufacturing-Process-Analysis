
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
OUT = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading manufacturing dataset...")
    df = pd.read_csv(DATA)

    required = ["Date", "Timestamp", "Machine_ID"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse the date and time fields.
    df["Combined_Timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip()
        + " "
        + df["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    invalid = df[df["Combined_Timestamp"].isna()].copy()

    if not invalid.empty:
        invalid.to_csv(
            OUT / "invalid_manufacturing_timestamps.csv",
            index=False
        )

    valid = df.dropna(
        subset=["Combined_Timestamp", "Machine_ID"]
    ).copy()

    valid = valid.sort_values(
        ["Machine_ID", "Combined_Timestamp"]
    )

    # Identify repeated machine/timestamp combinations.
    duplicate_mask = valid.duplicated(
        subset=["Machine_ID", "Combined_Timestamp"],
        keep=False
    )

    duplicates = valid[duplicate_mask].copy()

    duplicates.to_csv(
        OUT / "time_duplicate_machine_timestamps.csv",
        index=False
    )

    # Calculate time differences within each machine.
    valid["Interval_Minutes"] = (
        valid.groupby("Machine_ID")["Combined_Timestamp"]
        .diff()
        .dt.total_seconds() / 60
    )

    valid["Interval_Type"] = "First observation"

    valid.loc[
        valid["Interval_Minutes"] == 0,
        "Interval_Type"
    ] = "Duplicate timestamp"

    valid.loc[
        valid["Interval_Minutes"] > 0,
        "Interval_Type"
    ] = "Positive interval"

    valid.loc[
        valid["Interval_Minutes"] < 0,
        "Interval_Type"
    ] = "Negative interval"

    # Keep only positive intervals for cadence statistics.
    positive = valid[
        valid["Interval_Minutes"] > 0
    ].copy()

    # Overall interval distribution.
    interval_summary = positive["Interval_Minutes"].describe(
        percentiles=[0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )

    interval_summary.to_csv(
        OUT / "time_interval_overall_summary.csv",
        header=["minutes"]
    )

    # Most frequently occurring intervals.
    common_intervals = (
        positive["Interval_Minutes"]
        .value_counts()
        .rename_axis("interval_minutes")
        .reset_index(name="record_count")
    )

    common_intervals["percentage"] = (
        common_intervals["record_count"]
        / len(positive) * 100
    )

    common_intervals.head(30).to_csv(
        OUT / "time_interval_common_values.csv",
        index=False
    )

    # Per-machine interval statistics.
    machine_rows = []

    for machine_id, group in valid.groupby("Machine_ID"):
        machine_positive = group.loc[
            group["Interval_Minutes"] > 0,
            "Interval_Minutes"
        ]

        machine_median = machine_positive.median()

        if pd.notna(machine_median) and machine_median > 0:
            unusually_long = (
                machine_positive > 2 * machine_median
            ).sum()
        else:
            unusually_long = 0

        machine_rows.append({
            "Machine_ID": machine_id,
            "total_rows": len(group),
            "unique_timestamps": group[
                "Combined_Timestamp"
            ].nunique(),
            "duplicate_timestamp_rows": int(
                group.duplicated(
                    subset=["Combined_Timestamp"],
                    keep=False
                ).sum()
            ),
            "positive_intervals": len(machine_positive),
            "median_interval_minutes": machine_median,
            "mean_interval_minutes": machine_positive.mean(),
            "p90_interval_minutes": machine_positive.quantile(0.90),
            "p95_interval_minutes": machine_positive.quantile(0.95),
            "maximum_interval_minutes": machine_positive.max(),
            "intervals_over_2x_median": int(unusually_long)
        })

    machine_summary = pd.DataFrame(machine_rows)

    machine_summary.to_csv(
        OUT / "time_interval_by_machine.csv",
        index=False
    )

    # Flag long gaps using a machine-specific diagnostic threshold.
    # This is not a claim that the data collection system requires
    # a fixed sampling frequency.
    median_by_machine = (
        positive.groupby("Machine_ID")["Interval_Minutes"]
        .median()
        .rename("machine_median_interval")
    )

    positive = positive.join(
        median_by_machine,
        on="Machine_ID"
    )

    long_gaps = positive[
        positive["Interval_Minutes"]
        > 2 * positive["machine_median_interval"]
    ].copy()

    long_gaps.to_csv(
        OUT / "time_interval_long_gaps.csv",
        index=False
    )

    # Dataset-level timestamp coverage.
    coverage = valid.groupby("Machine_ID")[
        "Combined_Timestamp"
    ].agg(
        first_observation="min",
        last_observation="max",
        unique_timestamps="nunique"
    )

    coverage["coverage_days"] = (
        coverage["last_observation"]
        - coverage["first_observation"]
    ).dt.total_seconds() / 86400

    coverage.to_csv(
        OUT / "time_interval_machine_coverage.csv"
    )

    # Print a concise audit.
    print("\nTIMESTAMP INTERVAL AUDIT")
    print("Original rows:", len(df))
    print("Valid timestamp rows:", len(valid))
    print("Invalid timestamp rows:", len(invalid))
    print("Machines:", valid["Machine_ID"].nunique())
    print("Duplicate machine/timestamp rows:", len(duplicates))
    print("Positive time intervals:", len(positive))
    print("Long-gap flags:", len(long_gaps))

    print("\nOVERALL POSITIVE INTERVALS (MINUTES)")
    print(interval_summary.round(3))

    print("\nMOST COMMON INTERVALS")
    print(common_intervals.head(15).to_string(index=False))

    print("\nPER-MACHINE INTERVAL SUMMARY")
    print(machine_summary.round(3).to_string(index=False))

    print("\nReports saved to:", OUT)


if __name__ == "__main__":
    main()
