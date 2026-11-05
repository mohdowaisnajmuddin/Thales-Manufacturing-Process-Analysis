
"""Statistical analysis of manufacturing metrics across operation modes.

Raw-row Kruskal-Wallis results are retained as diagnostics only because
observations are repeated within machines.

The primary inferential analysis aggregates observations by machine and
operation mode, then uses the Friedman test for repeated-measures comparisons.

Limitations:
- Machine-level aggregation does not model temporal autocorrelation.
- Friedman assumes independent blocks (machines).
- Only machines observed in all three operation modes are included.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kruskal


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
OUT = ROOT / "reports"

METRICS = [
    "Production_Speed_units_per_hr",
    "Quality_Control_Defect_Rate_%",
    "Error_Rate_%",
    "Temperature_C",
    "Vibration_Hz",
    "Network_Latency_ms",
    "Packet_Loss_%",
]

MODES = ["Active", "Idle", "Maintenance"]


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)

    valid = np.isfinite(p_values)

    if not valid.any():
        return adjusted

    valid_indices = np.where(valid)[0]
    valid_p = p_values[valid]
    n = len(valid_p)

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    ranked_adjusted = sorted_p * n / np.arange(1, n + 1)

    # Enforce monotonicity, working from largest rank backwards.
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0, 1)

    adjusted_valid = np.empty(n)
    adjusted_valid[order] = ranked_adjusted
    adjusted[valid_indices] = adjusted_valid

    return adjusted


def calculate_epsilon_squared(h_statistic, sample_size, group_count):
    """Approximate Kruskal-Wallis epsilon-squared effect size."""
    denominator = sample_size - group_count

    if denominator <= 0:
        return np.nan

    return max(
        0.0,
        (h_statistic - group_count + 1) / denominator
    )


def calculate_kendalls_w(chi_square, block_count, group_count):
    """Calculate Kendall's W for a Friedman test."""
    denominator = block_count * (group_count - 1)

    if denominator <= 0:
        return np.nan

    return chi_square / denominator


def run_raw_row_kruskal(df):
    """Run original row-level tests for diagnostic comparison only."""
    results = []

    for metric in METRICS:
        groups = [
            group[metric].dropna().to_numpy()
            for mode, group in df.groupby("Operation_Mode")
            if group[metric].notna().any()
        ]

        if len(groups) < 2:
            continue

        try:
            statistic, p_value = kruskal(*groups)
        except ValueError as exc:
            print(f"Skipping raw Kruskal-Wallis for {metric}: {exc}")
            continue

        sample_size = sum(len(group) for group in groups)

        results.append({
            "metric": metric,
            "test": "Kruskal-Wallis",
            "statistic": statistic,
            "p_value": p_value,
            "group_count": len(groups),
            "sample_size": sample_size,
            "epsilon_squared": calculate_epsilon_squared(
                statistic,
                sample_size,
                len(groups)
            ),
            "inferential_status": "Diagnostic only: repeated machine observations",
            "interpretation_note": (
                "Raw-row p-value may be invalid because observations "
                "within machines may not be independent."
            ),
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df["bh_adjusted_p_value"] = benjamini_hochberg(
            results_df["p_value"]
        )

    return results_df


def run_machine_blocked_friedman(df):
    """Run Friedman tests using machines as repeated-measures blocks."""
    results = []
    aggregation_rows = []

    # One median measurement per machine and operation mode.
    machine_mode = (
        df.groupby(["Machine_ID", "Operation_Mode"])[METRICS]
        .median()
        .reset_index()
    )

    for metric in METRICS:
        pivot = machine_mode.pivot(
            index="Machine_ID",
            columns="Operation_Mode",
            values=metric
        )

        # Require all three operation modes for each machine.
        missing_modes = [mode for mode in MODES if mode not in pivot.columns]

        if missing_modes:
            print(
                f"Skipping {metric}: missing operation modes "
                f"{missing_modes}"
            )
            continue

        complete = pivot[MODES].dropna()
        block_count = len(complete)

        aggregation_rows.append({
            "metric": metric,
            "machine_count_total": df["Machine_ID"].nunique(),
            "machine_count_complete": block_count,
            "aggregation": "Median per machine per operation mode",
        })

        if block_count < 3:
            print(
                f"Skipping Friedman test for {metric}: "
                f"only {block_count} complete machines."
            )
            continue

        try:
            statistic, p_value = friedmanchisquare(
                complete["Active"].to_numpy(),
                complete["Idle"].to_numpy(),
                complete["Maintenance"].to_numpy(),
            )
        except ValueError as exc:
            print(f"Skipping Friedman test for {metric}: {exc}")
            continue

        kendalls_w = calculate_kendalls_w(
            statistic,
            block_count,
            len(MODES)
        )

        results.append({
            "metric": metric,
            "test": "Friedman",
            "statistic": statistic,
            "p_value": p_value,
            "group_count": len(MODES),
            "complete_machine_blocks": block_count,
            "kendalls_w": kendalls_w,
            "interpretation_note": (
                "Machine-blocked comparison; observational association, "
                "not causal. Temporal dependence is not explicitly modeled."
            ),
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df["bh_adjusted_p_value"] = benjamini_hochberg(
            results_df["p_value"]
        )

        results_df["significant_after_bh_0_05"] = (
            results_df["bh_adjusted_p_value"] < 0.05
        )

    aggregation_df = pd.DataFrame(aggregation_rows)

    return results_df, aggregation_df


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\n"
            "Check that the CSV is inside data/raw/."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)

    required_columns = ["Machine_ID", "Operation_Mode"] + METRICS
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    df["Machine_ID"] = df["Machine_ID"].astype(str).str.strip()
    df["Operation_Mode"] = (
        df["Operation_Mode"].astype(str).str.strip()
    )

    df[METRICS] = df[METRICS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    print("Dataset rows:", len(df))
    print("Unique machines:", df["Machine_ID"].nunique())
    print("Operation modes:", sorted(df["Operation_Mode"].dropna().unique()))

    unexpected_modes = sorted(
        set(df["Operation_Mode"].dropna()) - set(MODES)
    )

    if unexpected_modes:
        raise ValueError(
            f"Unexpected operation modes: {unexpected_modes}"
        )

    # 1. Original raw-row tests: diagnostics only.
    raw_results = run_raw_row_kruskal(df)

    raw_path = OUT / "operation_mode_kruskal_diagnostic.csv"
    raw_results.to_csv(raw_path, index=False)

    # 2. Machine-blocked repeated-measures tests.
    friedman_results, aggregation_summary = (
        run_machine_blocked_friedman(df)
    )

    friedman_path = OUT / "operation_mode_friedman_tests.csv"
    friedman_results.to_csv(friedman_path, index=False)

    aggregation_path = OUT / "machine_mode_aggregation_summary.csv"
    aggregation_summary.to_csv(aggregation_path, index=False)

    print("\nSaved:")
    print(raw_path)
    print(friedman_path)
    print(aggregation_path)

    print("\nMachine-blocked Friedman results:")
    if friedman_results.empty:
        print("No valid Friedman tests were produced.")
    else:
        print(
            friedman_results[
                [
                    "metric",
                    "statistic",
                    "p_value",
                    "bh_adjusted_p_value",
                    "kendalls_w",
                    "complete_machine_blocks",
                ]
            ].to_string(index=False)
        )

    print(
        "\nImportant: machine-level aggregation addresses repeated "
        "measurements within machines for this comparison, but does not "
        "explicitly model temporal autocorrelation or prove causality."
    )


if __name__ == "__main__":
    main()
