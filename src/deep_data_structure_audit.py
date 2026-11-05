
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    ROOT / "data" / "raw"
    / "Thales_Group_Manufacturing.csv"
)

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPEED = "Production_Speed_units_per_hr"

NUMERIC_COLUMNS = [
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


def load_data():
    df = pd.read_csv(RAW_FILE)

    required = [
        "Machine_ID",
        "Operation_Mode"
    ] + NUMERIC_COLUMNS

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df.dropna(
        subset=["Machine_ID", "Operation_Mode", SPEED]
    ).copy()


def value_structure_audit(df):
    print("\n" + "=" * 70)
    print("1. VALUE STRUCTURE AND ROUNDING AUDIT")
    print("=" * 70)

    records = []

    for col in NUMERIC_COLUMNS:

        values = df[col].dropna()

        rounded_0 = values.round(0)
        rounded_1 = values.round(1)
        rounded_2 = values.round(2)

        records.append({
            "feature": col,
            "unique_values": values.nunique(),
            "integer_percentage": (
                np.isclose(
                    values,
                    rounded_0
                ).mean() * 100
            ),
            "one_decimal_percentage": (
                np.isclose(
                    values,
                    rounded_1
                ).mean() * 100
            ),
            "two_decimal_percentage": (
                np.isclose(
                    values,
                    rounded_2
                ).mean() * 100
            ),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max()
        })

    result = pd.DataFrame(records)

    print(result.round(4).to_string(index=False))

    result.to_csv(
        REPORT_DIR / "deep_value_structure_audit.csv",
        index=False
    )


def machine_distribution_audit(df):
    print("\n" + "=" * 70)
    print("2. MACHINE-LEVEL PRODUCTION DISTRIBUTIONS")
    print("=" * 70)

    result = (
        df.groupby("Machine_ID")[SPEED]
        .agg([
            "count",
            "mean",
            "std",
            "min",
            "median",
            "max",
            "nunique"
        ])
        .reset_index()
    )

    result["coefficient_of_variation_pct"] = (
        result["std"] / result["mean"].abs()
    ) * 100

    print(result.round(4).to_string(index=False))

    print("\nAcross-machine summary:")
    print(
        result.drop(columns="Machine_ID")
        .describe()
        .round(4)
        .to_string()
    )

    result.to_csv(
        REPORT_DIR / "machine_production_distributions.csv",
        index=False
    )


def operation_mode_audit(df):
    print("\n" + "=" * 70)
    print("3. OPERATION MODE PRODUCTION COMPARISON")
    print("=" * 70)

    result = (
        df.groupby("Operation_Mode")[NUMERIC_COLUMNS]
        .agg(["count", "mean", "std", "median"])
    )

    print(result.round(4).to_string())

    machine_mode = (
        df.groupby(
            ["Machine_ID", "Operation_Mode"]
        )[NUMERIC_COLUMNS]
        .mean()
        .reset_index()
    )

    machine_mode.to_csv(
        REPORT_DIR / "deep_machine_mode_means.csv",
        index=False
    )

    mode_summary = (
        df.groupby("Operation_Mode")[NUMERIC_COLUMNS]
        .mean()
    )

    mode_summary.to_csv(
        REPORT_DIR / "deep_operation_mode_means.csv"
    )


def sensor_relationship_audit(df):
    print("\n" + "=" * 70)
    print("4. SENSOR RELATIONSHIP AUDIT")
    print("=" * 70)

    pearson = df[NUMERIC_COLUMNS].corr(
        method="pearson"
    )

    spearman = df[NUMERIC_COLUMNS].corr(
        method="spearman"
    )

    print("\nPearson correlation:")
    print(pearson.round(3).to_string())

    print("\nSpearman correlation:")
    print(spearman.round(3).to_string())

    pearson.to_csv(
        REPORT_DIR / "deep_sensor_pearson.csv"
    )

    spearman.to_csv(
        REPORT_DIR / "deep_sensor_spearman.csv"
    )


def machine_sensor_relationship_audit(df):
    print("\n" + "=" * 70)
    print("5. MACHINE-SPECIFIC SENSOR RELATIONSHIPS")
    print("=" * 70)

    records = []

    sensor_columns = [
        col for col in NUMERIC_COLUMNS
        if col != SPEED
    ]

    for machine, group in df.groupby("Machine_ID"):

        for feature in sensor_columns:

            valid = group[
                [SPEED, feature]
            ].dropna()

            if len(valid) < 30:
                continue

            records.append({
                "Machine_ID": machine,
                "feature": feature,
                "records": len(valid),
                "pearson": valid[SPEED].corr(
                    valid[feature]
                ),
                "spearman": valid[SPEED].corr(
                    valid[feature],
                    method="spearman"
                )
            })

    result = pd.DataFrame(records)

    summary = (
        result.groupby("feature")
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

    print("\nMachine-level relationship summary:")
    print(summary.round(4).to_string())

    result.to_csv(
        REPORT_DIR / "deep_machine_sensor_relationships.csv",
        index=False
    )

    summary.to_csv(
        REPORT_DIR / "deep_machine_sensor_summary.csv"
    )


def main():
    df = load_data()

    print("Rows:", len(df))
    print("Machines:", df["Machine_ID"].nunique())
    print(
        "Operation modes:",
        df["Operation_Mode"].nunique()
    )

    value_structure_audit(df)
    machine_distribution_audit(df)
    operation_mode_audit(df)
    sensor_relationship_audit(df)
    machine_sensor_relationship_audit(df)

    print("\nAll deep structure audits completed.")
    print("Reports:", REPORT_DIR)


if __name__ == "__main__":
    main()
