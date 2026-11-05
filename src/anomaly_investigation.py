
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "machine_anomaly_results.csv"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, parse_dates=["Combined_Timestamp"])

FEATURES = [
    "Production_Speed_units_per_hr",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Predictive_Maintenance_Score",
]

OUTCOMES = [
    "Quality_Control_Defect_Rate_%",
    "Error_Rate_%",
]

METHOD = "Isolation_Forest_Anomaly"

# 1. ANOMALY RATE BY MACHINE AND MODE

machine_mode = (
    df.groupby(["Machine_ID", "Operation_Mode"])
    .agg(
        Records=(METHOD, "size"),
        Anomalies=(METHOD, "sum"),
        Anomaly_Rate=(METHOD, "mean"),
    )
    .reset_index()
)

machine_mode["Anomaly_Rate_Pct"] = (
    machine_mode["Anomaly_Rate"] * 100
)

machine_mode.to_csv(
    REPORTS / "anomaly_machine_mode_rates.csv",
    index=False,
)

# 2. ANOMALY RATE BY DAY

daily = (
    df.groupby(pd.Grouper(key="Combined_Timestamp", freq="D"))
    .agg(
        Records=(METHOD, "size"),
        Anomalies=(METHOD, "sum"),
        Anomaly_Rate=(METHOD, "mean"),
    )
    .reset_index()
)

daily["Anomaly_Rate_Pct"] = daily["Anomaly_Rate"] * 100

daily.to_csv(
    REPORTS / "anomaly_daily_investigation.csv",
    index=False,
)

# 3. FEATURE DISTRIBUTIONS: FLAGGED VS NORMAL

feature_rows = []

for feature in FEATURES:
    flagged = df.loc[df[METHOD], feature].dropna()
    normal = df.loc[~df[METHOD], feature].dropna()

    feature_rows.append({
        "Feature": feature,
        "Flagged_Count": len(flagged),
        "Normal_Count": len(normal),
        "Flagged_Mean": flagged.mean(),
        "Normal_Mean": normal.mean(),
        "Mean_Difference": flagged.mean() - normal.mean(),
        "Flagged_Median": flagged.median(),
        "Normal_Median": normal.median(),
        "Flagged_Std": flagged.std(),
        "Normal_Std": normal.std(),
    })

feature_comparison = pd.DataFrame(feature_rows)

feature_comparison.to_csv(
    REPORTS / "anomaly_feature_comparison.csv",
    index=False,
)

# 4. QUALITY COMPARISON WITHIN MACHINE AND MODE

quality_rows = []

evaluation = df[df["Reference_Period"] == "Evaluation"].copy()

for (machine, mode), group in evaluation.groupby(
    ["Machine_ID", "Operation_Mode"]
):

    flagged = group[group[METHOD]]
    normal = group[~group[METHOD]]

    for outcome in OUTCOMES:

        flagged_values = flagged[outcome].dropna()
        normal_values = normal[outcome].dropna()

        if len(flagged_values) == 0 or len(normal_values) == 0:
            continue

        quality_rows.append({
            "Machine_ID": machine,
            "Operation_Mode": mode,
            "Outcome": outcome,
            "Flagged_Count": len(flagged_values),
            "Normal_Count": len(normal_values),
            "Flagged_Mean": flagged_values.mean(),
            "Normal_Mean": normal_values.mean(),
            "Mean_Difference": (
                flagged_values.mean() - normal_values.mean()
            ),
        })

quality_comparison = pd.DataFrame(quality_rows)

quality_comparison.to_csv(
    REPORTS / "anomaly_quality_within_machine_mode.csv",
    index=False,
)

# 5. SUMMARY

print("\n" + "=" * 65)
print("ANOMALY INVESTIGATION")
print("=" * 65)

print("\nTop machine-mode combinations:")
print(
    machine_mode.sort_values(
        "Anomaly_Rate_Pct",
        ascending=False,
    )
    .head(15)
    .round(3)
    .to_string(index=False)
)

print("\nHighest daily anomaly rates:")
print(
    daily.sort_values(
        "Anomaly_Rate_Pct",
        ascending=False,
    )
    .head(15)
    .round(3)
    .to_string(index=False)
)

print("\nFlagged vs normal feature comparison:")
print(
    feature_comparison.round(4).to_string(index=False)
)

print("\nWithin-machine/mode quality comparison:")
print(
    quality_comparison.groupby("Outcome")[
        ["Flagged_Count", "Normal_Count", "Mean_Difference"]
    ]
    .agg(["count", "mean", "median"])
    .round(4)
    .to_string()
)

print("\nSaved reports:")
print("anomaly_machine_mode_rates.csv")
print("anomaly_daily_investigation.csv")
print("anomaly_feature_comparison.csv")
print("anomaly_quality_within_machine_mode.csv")

print("\nInvestigation completed.")


# ---------------------------------------------------------
# ADDITIONAL VALIDATION
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("ADDITIONAL ANOMALY VALIDATION")
print("=" * 65)

# 1. Overall anomaly rate by operation mode

mode_rates = (
    df.groupby("Operation_Mode")[METHOD]
    .agg(
        Records="count",
        Anomalies="sum",
        Anomaly_Rate="mean",
    )
    .reset_index()
)

mode_rates["Anomaly_Rate_Pct"] = (
    mode_rates["Anomaly_Rate"] * 100
)

print("\nAnomaly rate by operation mode:")
print(mode_rates.round(3).to_string(index=False))

mode_rates.to_csv(
    REPORTS / "anomaly_rates_by_operation_mode.csv",
    index=False,
)

# 2. Investigate unusual daily record counts

daily_coverage = (
    df.groupby(
        df["Combined_Timestamp"].dt.date
    )
    .agg(
        Records=("Machine_ID", "size"),
        Machines=("Machine_ID", "nunique"),
        Anomalies=(METHOD, "sum"),
    )
    .reset_index()
)

daily_coverage["Anomaly_Rate_Pct"] = (
    daily_coverage["Anomalies"]
    / daily_coverage["Records"]
    * 100
)

print("\nDaily record coverage:")
print(
    daily_coverage.sort_values(
        "Records",
        ascending=False,
    ).head(15).round(3).to_string(index=False)
)

daily_coverage.to_csv(
    REPORTS / "anomaly_daily_coverage.csv",
    index=False,
)

# 3. Weighted and group-level quality comparison

evaluation = df[
    df["Reference_Period"] == "Evaluation"
].copy()

quality_results = []

for outcome in OUTCOMES:

    total_flagged_sum = 0.0
    total_normal_sum = 0.0
    total_flagged_count = 0
    total_normal_count = 0

    positive_groups = 0
    negative_groups = 0
    equal_groups = 0
    valid_groups = 0

    group_differences = []

    for _, group in evaluation.groupby(
        ["Machine_ID", "Operation_Mode"]
    ):

        flagged = group.loc[
            group[METHOD], outcome
        ].dropna()

        normal = group.loc[
            ~group[METHOD], outcome
        ].dropna()

        if len(flagged) == 0 or len(normal) == 0:
            continue

        difference = flagged.mean() - normal.mean()

        group_differences.append(difference)
        valid_groups += 1

        if difference > 0:
            positive_groups += 1
        elif difference < 0:
            negative_groups += 1
        else:
            equal_groups += 1

        total_flagged_sum += flagged.sum()
        total_normal_sum += normal.sum()

        total_flagged_count += len(flagged)
        total_normal_count += len(normal)

    weighted_flagged_mean = (
        total_flagged_sum / total_flagged_count
        if total_flagged_count else np.nan
    )

    weighted_normal_mean = (
        total_normal_sum / total_normal_count
        if total_normal_count else np.nan
    )

    quality_results.append({
        "Outcome": outcome,
        "Valid_Groups": valid_groups,
        "Positive_Groups": positive_groups,
        "Negative_Groups": negative_groups,
        "Equal_Groups": equal_groups,
        "Weighted_Flagged_Mean": weighted_flagged_mean,
        "Weighted_Normal_Mean": weighted_normal_mean,
        "Weighted_Mean_Difference": (
            weighted_flagged_mean - weighted_normal_mean
        ),
        "Unweighted_Group_Mean_Difference": (
            np.mean(group_differences)
            if group_differences else np.nan
        ),
    })

quality_summary = pd.DataFrame(quality_results)

print("\nWeighted quality comparison:")
print(quality_summary.round(4).to_string(index=False))

quality_summary.to_csv(
    REPORTS / "anomaly_quality_weighted_validation.csv",
    index=False,
)

print("\nAdditional validation completed.")


# ---------------------------------------------------------
# INVESTIGATE MARCH 1 RECORD-COUNT ANOMALY
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("MARCH 1 DATA-COVERAGE INVESTIGATION")
print("=" * 65)

march_1 = df[
    df["Combined_Timestamp"].dt.date
    == pd.Timestamp("2025-03-01").date()
].copy()

print("\nTotal March 1 records:", len(march_1))
print("Machines:", march_1["Machine_ID"].nunique())

# Records per machine
machine_counts = (
    march_1.groupby("Machine_ID")
    .agg(
        Records=("Combined_Timestamp", "size"),
        Unique_Timestamps=("Combined_Timestamp", "nunique"),
    )
    .reset_index()
)

machine_counts["Duplicate_Timestamps"] = (
    machine_counts["Records"]
    - machine_counts["Unique_Timestamps"]
)

print("\nRecords per machine:")
print(machine_counts.describe().round(2).to_string())

print("\nMachines with highest record counts:")
print(
    machine_counts.sort_values(
        "Records", ascending=False
    ).head(15).to_string(index=False)
)

# Exact duplicate machine/timestamp combinations
duplicate_rows = march_1.duplicated(
    subset=["Machine_ID", "Combined_Timestamp"],
    keep=False,
)

print("\nRows belonging to duplicate machine/timestamp groups:",
      int(duplicate_rows.sum()))

# Timestamp interval distribution per machine
intervals = (
    march_1.sort_values(
        ["Machine_ID", "Combined_Timestamp"]
    )
    .groupby("Machine_ID")["Combined_Timestamp"]
    .diff()
    .dt.total_seconds()
    .div(60)
    .dropna()
)

print("\nMinutes between consecutive records:")
print(intervals.describe(
    percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
).round(2).to_string())

# Compare record counts for adjacent dates
nearby = df[
    df["Combined_Timestamp"].dt.date.isin([
        pd.Timestamp("2025-02-28").date(),
        pd.Timestamp("2025-03-01").date(),
        pd.Timestamp("2025-03-02").date(),
    ])
]

nearby_summary = (
    nearby.groupby(nearby["Combined_Timestamp"].dt.date)
    .agg(
        Records=("Machine_ID", "size"),
        Machines=("Machine_ID", "nunique"),
        Unique_Timestamps=("Combined_Timestamp", "nunique"),
    )
    .reset_index()
)

print("\nAdjacent-day comparison:")
print(nearby_summary.to_string(index=False))

machine_counts.to_csv(
    REPORTS / "march_1_machine_coverage.csv",
    index=False,
)

nearby_summary.to_csv(
    REPORTS / "march_1_adjacent_day_coverage.csv",
    index=False,
)

print("\nMarch 1 investigation completed.")


# ---------------------------------------------------------
# COMPARE MACHINE-LEVEL COVERAGE: FEB 28, MAR 1, MAR 2
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("THREE-DAY MACHINE-LEVEL COVERAGE COMPARISON")
print("=" * 65)

comparison_dates = [
    pd.Timestamp("2025-02-28").date(),
    pd.Timestamp("2025-03-01").date(),
    pd.Timestamp("2025-03-02").date(),
]

comparison = df[
    df["Combined_Timestamp"].dt.date.isin(comparison_dates)
].copy()

comparison["Date"] = comparison["Combined_Timestamp"].dt.date

# Per-machine counts and unique timestamps
coverage = (
    comparison.groupby(["Date", "Machine_ID"])
    .agg(
        Records=("Combined_Timestamp", "size"),
        Unique_Timestamps=("Combined_Timestamp", "nunique"),
        Anomalies=(METHOD, "sum"),
    )
    .reset_index()
)

coverage["Duplicate_Rows"] = (
    coverage["Records"] - coverage["Unique_Timestamps"]
)

coverage["Anomaly_Rate_Pct"] = (
    coverage["Anomalies"] / coverage["Records"] * 100
)

print("\nDaily machine-level record summary:")
print(
    coverage.groupby("Date")[
        ["Records", "Unique_Timestamps", "Duplicate_Rows",
         "Anomalies", "Anomaly_Rate_Pct"]
    ].agg(["mean", "median", "min", "max"]).round(2).to_string()
)

# Per-machine interval comparison
interval_rows = []

for (date, machine), group in comparison.groupby(
    ["Date", "Machine_ID"]
):
    timestamps = (
        group["Combined_Timestamp"]
        .drop_duplicates()
        .sort_values()
    )

    gaps = (
        timestamps.diff()
        .dt.total_seconds()
        .div(60)
        .dropna()
    )

    interval_rows.append({
        "Date": date,
        "Machine_ID": machine,
        "Unique_Timestamps": len(timestamps),
        "Median_Gap_Minutes": gaps.median() if len(gaps) else np.nan,
        "Mean_Gap_Minutes": gaps.mean() if len(gaps) else np.nan,
        "Min_Gap_Minutes": gaps.min() if len(gaps) else np.nan,
        "Max_Gap_Minutes": gaps.max() if len(gaps) else np.nan,
    })

interval_comparison = pd.DataFrame(interval_rows)

print("\nMachine-level interval summary:")
print(
    interval_comparison.groupby("Date")[
        ["Unique_Timestamps", "Median_Gap_Minutes",
         "Mean_Gap_Minutes", "Min_Gap_Minutes", "Max_Gap_Minutes"]
    ].agg(["mean", "median", "min", "max"]).round(2).to_string()
)

# Identify machines with unusually large count changes on March 1
wide_counts = coverage.pivot(
    index="Machine_ID",
    columns="Date",
    values="Records",
).fillna(0)

wide_counts["March1_vs_Feb28"] = (
    wide_counts[comparison_dates[1]]
    - wide_counts[comparison_dates[0]]
)

wide_counts["March1_vs_March2"] = (
    wide_counts[comparison_dates[1]]
    - wide_counts[comparison_dates[2]]
)

print("\nLargest March 1 record-count increases:")
print(
    wide_counts.sort_values(
        "March1_vs_Feb28",
        ascending=False,
    ).head(15).round(2).to_string()
)

coverage.to_csv(
    REPORTS / "three_day_machine_coverage.csv",
    index=False,
)

interval_comparison.to_csv(
    REPORTS / "three_day_machine_intervals.csv",
    index=False,
)

wide_counts.to_csv(
    REPORTS / "march_1_machine_count_changes.csv",
)

print("\nThree-day comparison completed.")
