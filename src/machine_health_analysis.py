
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SENSOR_COLUMNS = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
]

REQUIRED_COLUMNS = [
    "Machine_ID",
    *SENSOR_COLUMNS,
]


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df["Machine_ID"] = (
        df["Machine_ID"].astype("string").str.strip()
    )

    df = df.dropna(subset=["Machine_ID"])

    print("\nDATASET INFORMATION")
    print("-" * 50)
    print(f"Rows: {len(df):,}")
    print(f"Machines: {df['Machine_ID'].nunique()}")
    print(f"Sensor columns: {len(SENSOR_COLUMNS)}")

    return df


# --------------------------------------------------
# 3. MACHINE-WISE DESCRIPTIVE STATISTICS
# --------------------------------------------------

def calculate_machine_statistics(df):

    results = []

    for machine_id, group in df.groupby("Machine_ID"):

        for sensor in SENSOR_COLUMNS:

            values = group[sensor].dropna()

            if values.empty:
                continue

            mean_value = values.mean()
            std_value = values.std()

            cv = (
                (std_value / abs(mean_value)) * 100
                if pd.notna(std_value)
                and mean_value != 0
                else np.nan
            )

            results.append({
                "Machine_ID": machine_id,
                "Sensor": sensor,
                "Count": len(values),
                "Mean": mean_value,
                "Median": values.median(),
                "Std_Dev": std_value,
                "Minimum": values.min(),
                "Q1": values.quantile(0.25),
                "Q3": values.quantile(0.75),
                "Maximum": values.max(),
                "CV_Percent": cv,
            })

    return pd.DataFrame(results)


# --------------------------------------------------
# 4. MACHINE-SPECIFIC IQR OUTLIER DETECTION
# --------------------------------------------------

def detect_machine_outliers(df):

    results = []

    for machine_id, group in df.groupby("Machine_ID"):

        for sensor in SENSOR_COLUMNS:

            values = group[sensor].dropna()

            if len(values) < 4:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            mask = (
                group[sensor].notna()
                & (
                    (group[sensor] < lower_bound)
                    | (group[sensor] > upper_bound)
                )
            )

            outliers = group.loc[mask].copy()

            for _, row in outliers.iterrows():

                results.append({
                    "Machine_ID": machine_id,
                    "Sensor": sensor,
                    "Observed_Value": row[sensor],
                    "Lower_Bound": lower_bound,
                    "Upper_Bound": upper_bound,
                    "Deviation_Type": (
                        "Below lower bound"
                        if row[sensor] < lower_bound
                        else "Above upper bound"
                    ),
                })

    return pd.DataFrame(results)


# --------------------------------------------------
# 5. OUTLIER SUMMARY
# --------------------------------------------------

def calculate_outlier_summary(
    statistics_df,
    outliers_df
):

    summary = statistics_df[
        ["Machine_ID", "Sensor", "Count"]
    ].copy()

    if outliers_df.empty:

        summary["Outlier_Count"] = 0

    else:

        counts = (
            outliers_df.groupby(
                ["Machine_ID", "Sensor"]
            )
            .size()
            .reset_index(name="Outlier_Count")
        )

        summary = summary.merge(
            counts,
            on=["Machine_ID", "Sensor"],
            how="left"
        )

        summary["Outlier_Count"] = (
            summary["Outlier_Count"].fillna(0).astype(int)
        )

    summary["Outlier_Percent"] = (
        summary["Outlier_Count"]
        / summary["Count"]
        * 100
    )

    return summary


# --------------------------------------------------
# 6. MAIN EXECUTION
# --------------------------------------------------

def main():

    df = load_data()

    print("\nCALCULATING MACHINE STATISTICS...")

    statistics_df = calculate_machine_statistics(df)

    print("DETECTING MACHINE-SPECIFIC OUTLIERS...")

    outliers_df = detect_machine_outliers(df)

    outlier_summary = calculate_outlier_summary(
        statistics_df,
        outliers_df
    )

    # Export results
    statistics_df.to_csv(
        OUTPUT_DIR / "machine_sensor_statistics.csv",
        index=False
    )

    outliers_df.to_csv(
        OUTPUT_DIR / "machine_sensor_outliers.csv",
        index=False
    )

    outlier_summary.to_csv(
        OUTPUT_DIR / "machine_sensor_outlier_summary.csv",
        index=False
    )

    # Print summary
    print("\nANALYSIS COMPLETE")
    print("-" * 50)

    print(
        f"Machine-sensor combinations: {len(statistics_df)}"
    )

    print(
        f"Detected statistical outliers: {len(outliers_df)}"
    )

    print("\nOUTLIERS BY SENSOR")

    if not outliers_df.empty:
        print(
            outliers_df.groupby("Sensor").size()
        )
    else:
        print("No statistical outliers detected.")

    print("\nHIGHEST VARIABILITY BY SENSOR")

    for sensor in SENSOR_COLUMNS:

        sensor_data = statistics_df[
            statistics_df["Sensor"] == sensor
        ]

        print(f"\n{sensor}")

        print(
            sensor_data.nlargest(
                5, "CV_Percent"
            )[
                ["Machine_ID", "Mean", "Std_Dev", "CV_Percent"]
            ].to_string(index=False)
        )

    print("\nFILES SAVED TO:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
