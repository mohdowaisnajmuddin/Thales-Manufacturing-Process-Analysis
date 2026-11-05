
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "Production_Speed_units_per_hr",
    "Error_Rate_%",
    "Quality_Control_Defect_Rate_%"
]

EXPECTED_MODES = ["Active", "Idle", "Maintenance"]


def load_data():
    df = pd.read_csv(DATA_PATH)

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df.dropna(subset=["Machine_ID", "Operation_Mode"] + METRICS)

    print("\nDATA VALIDATION")
    print("-" * 60)
    print("Rows:", len(df))
    print("Machines:", df["Machine_ID"].nunique())
    print("Modes:", sorted(df["Operation_Mode"].unique()))

    return df


def calculate_machine_mode_means(df):
    return (
        df.groupby(
            ["Machine_ID", "Operation_Mode"],
            observed=True
        )[METRICS]
        .mean()
        .reset_index()
    )


def run_friedman_tests(machine_means):
    results = []

    for metric in METRICS:
        pivot = machine_means.pivot(
            index="Machine_ID",
            columns="Operation_Mode",
            values=metric
        )

        # Retain only machines with all three operation modes.
        pivot = pivot.reindex(columns=EXPECTED_MODES).dropna()

        if len(pivot) < 2:
            print(f"Insufficient complete machine blocks for {metric}")
            continue

        statistic, p_value = friedmanchisquare(
            pivot["Active"],
            pivot["Idle"],
            pivot["Maintenance"]
        )

        number_of_machines = len(pivot)
        number_of_modes = len(EXPECTED_MODES)

        # Kendall's W effect size for the Friedman test.
        kendall_w = statistic / (
            number_of_machines * (number_of_modes - 1)
        )

        results.append({
            "Metric": metric,
            "Machines_Compared": number_of_machines,
            "Friedman_Statistic": statistic,
            "P_Value": p_value,
            "Kendalls_W": kendall_w,
            "Active_Mean": pivot["Active"].mean(),
            "Idle_Mean": pivot["Idle"].mean(),
            "Maintenance_Mean": pivot["Maintenance"].mean()
        })

    return pd.DataFrame(results)


def main():
    df = load_data()

    print("\nCALCULATING MACHINE-MODE MEANS...")
    machine_means = calculate_machine_mode_means(df)

    machine_means.to_csv(
        OUTPUT_DIR / "machine_mode_means_for_tests.csv",
        index=False
    )

    print("\nRUNNING FRIEDMAN TESTS...")
    results = run_friedman_tests(machine_means)

    if results.empty:
        raise ValueError("No statistical tests could be completed.")

    # Correct the three metric-level p-values together.
    results["Adjusted_P_Value"] = multipletests(
        results["P_Value"],
        method="fdr_bh"
    )[1]

    results["Significant_After_Correction"] = (
        results["Adjusted_P_Value"] < 0.05
    )

    results.to_csv(
        OUTPUT_DIR / "operation_mode_statistical_tests.csv",
        index=False
    )

    print("\nSTATISTICAL TEST RESULTS")
    print("-" * 80)
    print(results.round(6).to_string(index=False))

    print("\nINTERPRETATION GUIDE")
    print("-" * 60)
    print("Friedman test: compares mode rankings across machines.")
    print("Kendall's W: effect size; values closer to zero indicate")
    print("weaker agreement in mode-related differences.")
    print("Adjusted p-value: accounts for the three metric tests.")
    print("Statistical significance does not establish causation.")

    print("\nFILES SAVED:")
    print(OUTPUT_DIR / "machine_mode_means_for_tests.csv")
    print(OUTPUT_DIR / "operation_mode_statistical_tests.csv")


if __name__ == "__main__":
    main()
