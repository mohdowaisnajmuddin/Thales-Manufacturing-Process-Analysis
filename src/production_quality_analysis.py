
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {
    "Production_Speed_units_per_hr": "production",
    "Error_Rate_%": "high",
    "Quality_Control_Defect_Rate_%": "high",
}

GROUP_COLUMNS = [
    "Machine_ID",
    "Operation_Mode",
]


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_data():

    df = pd.read_csv(DATA_PATH)

    required = [
        *GROUP_COLUMNS,
        *METRICS.keys(),
        "Date",
        "Timestamp",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for metric in METRICS:
        df[metric] = pd.to_numeric(
            df[metric], errors="coerce"
        )

    df["Combined_Timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip()
        + " "
        + df["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.dropna(
        subset=GROUP_COLUMNS
    )

    print("\nDATA VALIDATION")
    print("-" * 50)
    print("Rows:", len(df))
    print("Machines:", df["Machine_ID"].nunique())
    print("Operation modes:", df["Operation_Mode"].nunique())

    return df


# --------------------------------------------------
# 3. MACHINE AND MODE SUMMARY
# --------------------------------------------------

def calculate_group_summary(df):

    results = []

    for (machine, mode), group in df.groupby(
        GROUP_COLUMNS,
        observed=True
    ):

        for metric in METRICS:

            values = group[metric].dropna()

            if values.empty:
                continue

            mean_value = values.mean()
            std_value = values.std()

            cv = (
                std_value / abs(mean_value) * 100
                if pd.notna(std_value)
                and mean_value != 0
                else np.nan
            )

            results.append({
                "Machine_ID": machine,
                "Operation_Mode": mode,
                "Metric": metric,
                "Count": len(values),
                "Mean": mean_value,
                "Median": values.median(),
                "Std_Dev": std_value,
                "Minimum": values.min(),
                "Maximum": values.max(),
                "CV_Percent": cv,
            })

    return pd.DataFrame(results)


# --------------------------------------------------
# 4. MODE-LEVEL SUMMARY
# --------------------------------------------------

def calculate_mode_summary(df):

    results = []

    for mode, group in df.groupby(
        "Operation_Mode",
        observed=True
    ):

        for metric in METRICS:

            values = group[metric].dropna()

            if values.empty:
                continue

            results.append({
                "Operation_Mode": mode,
                "Metric": metric,
                "Count": len(values),
                "Mean": values.mean(),
                "Median": values.median(),
                "Std_Dev": values.std(),
                "Minimum": values.min(),
                "Maximum": values.max(),
            })

    return pd.DataFrame(results)


# --------------------------------------------------
# 5. MACHINE-SPECIFIC IQR INVESTIGATION
# --------------------------------------------------

def detect_unusual_observations(df):

    results = []

    for machine, group in df.groupby(
        "Machine_ID",
        observed=True
    ):

        for metric, direction in METRICS.items():

            values = group[metric].dropna()

            if len(values) < 4:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            if direction == "high":

                mask = (
                    group[metric].notna()
                    & (group[metric] > upper)
                )

                reason = "Above machine-specific upper IQR bound"

            else:

                mask = (
                    group[metric].notna()
                    & (group[metric] < lower)
                )

                reason = "Below machine-specific lower IQR bound"

            unusual = group.loc[mask]

            for _, row in unusual.iterrows():

                results.append({
                    "Machine_ID": machine,
                    "Operation_Mode": row["Operation_Mode"],
                    "Timestamp": row["Combined_Timestamp"],
                    "Metric": metric,
                    "Observed_Value": row[metric],
                    "Lower_Bound": lower,
                    "Upper_Bound": upper,
                    "Detection_Reason": reason,
                })

    return pd.DataFrame(results)


# --------------------------------------------------
# 6. DAILY PRODUCTION AND QUALITY SUMMARY
# --------------------------------------------------

def calculate_daily_summary(df):

    valid = df.dropna(
        subset=["Combined_Timestamp"]
    ).copy()

    valid["Date"] = (
        valid["Combined_Timestamp"].dt.normalize()
    )

    daily = (
        valid.groupby(
            ["Date", "Machine_ID", "Operation_Mode"],
            as_index=False,
            observed=True
        )
        .agg(
            Production_Mean=(
                "Production_Speed_units_per_hr", "mean"
            ),
            Production_Std=(
                "Production_Speed_units_per_hr", "std"
            ),
            Error_Rate_Mean=("Error_Rate_%", "mean"),
            Defect_Rate_Mean=(
                "Quality_Control_Defect_Rate_%", "mean"
            ),
            Reading_Count=("Machine_ID", "size"),
        )
    )

    return daily.sort_values(
        ["Date", "Machine_ID"]
    )


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():

    df = load_data()

    print("\nCALCULATING MACHINE-MODE SUMMARIES...")

    group_summary = calculate_group_summary(df)

    print("\nCALCULATING OPERATION-MODE SUMMARIES...")

    mode_summary = calculate_mode_summary(df)

    print("\nDETECTING UNUSUAL OBSERVATIONS...")

    unusual = detect_unusual_observations(df)

    print("\nCALCULATING DAILY PRODUCTION AND QUALITY...")

    daily = calculate_daily_summary(df)

    # Export
    outputs = {
        "machine_mode_production_quality.csv": group_summary,
        "operation_mode_production_quality.csv": mode_summary,
        "production_quality_unusual_observations.csv": unusual,
        "daily_production_quality.csv": daily,
    }

    for filename, result in outputs.items():

        result.to_csv(
            OUTPUT_DIR / filename,
            index=False
        )

    # Results
    print("\nANALYSIS COMPLETE")
    print("-" * 50)

    print("Machine-mode metric combinations:", len(group_summary))
    print("Unusual observations:", len(unusual))
    print("Daily machine-mode records:", len(daily))

    print("\nUNUSUAL OBSERVATIONS BY METRIC")

    if unusual.empty:
        print("No unusual observations detected.")

    else:
        print(
            unusual.groupby("Metric").size()
        )

        print("\nMOST FREQUENTLY FLAGGED MACHINES")

        print(
            unusual.groupby(
                ["Machine_ID", "Metric"]
            ).size()
            .reset_index(name="Flag_Count")
            .sort_values(
                "Flag_Count",
                ascending=False
            )
            .head(10)
            .to_string(index=False)
        )

    print("\nOPERATION MODE SUMMARY")

    print(
        mode_summary.to_string(index=False)
    )

    print("\nFILES SAVED TO:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
