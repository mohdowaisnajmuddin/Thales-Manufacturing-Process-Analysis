
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
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


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_data():

    df = pd.read_csv(DATA_PATH)

    required = [
        "Machine_ID",
        "Date",
        "Timestamp",
        *SENSOR_COLUMNS,
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Combined_Timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip()
        + " "
        + df["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )

    for sensor in SENSOR_COLUMNS:
        df[sensor] = pd.to_numeric(
            df[sensor], errors="coerce"
        )

    df["Machine_ID"] = (
        df["Machine_ID"].astype("string").str.strip()
    )

    df = df.dropna(
        subset=["Machine_ID", "Combined_Timestamp"]
    )

    print("\nTIMESTAMP VALIDATION")
    print("-" * 50)
    print(f"Valid timestamp rows: {len(df):,}")
    print(
        "Unique dates:",
        df["Combined_Timestamp"].dt.date.nunique()
    )
    print(
        "Machines:",
        df["Machine_ID"].nunique()
    )

    return df


# --------------------------------------------------
# 3. CREATE DAILY SENSOR AVERAGES
# --------------------------------------------------

def calculate_daily_averages(df):

    daily = (
        df.groupby(
            [
                "Machine_ID",
                pd.Grouper(
                    key="Combined_Timestamp",
                    freq="D"
                ),
            ]
        )[SENSOR_COLUMNS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Flatten multi-level column names
    daily.columns = [
        "_".join(
            str(part) for part in col if str(part) != ""
        )
        if isinstance(col, tuple)
        else col
        for col in daily.columns
    ]

    daily = daily.rename(
        columns={
            "Combined_Timestamp_": "Date",
            "Machine_ID_": "Machine_ID",
        }
    )

    daily = daily.sort_values(
        ["Machine_ID", "Date"]
    )

    return daily


# --------------------------------------------------
# 4. CALCULATE TEMPORAL STABILITY METRICS
# --------------------------------------------------



def calculate_daily_averages(df):

    working = df.copy()

    # Ensure the timestamp is a valid datetime column.
    working["Combined_Timestamp"] = pd.to_datetime(
        working["Combined_Timestamp"],
        errors="coerce"
    )

    working = working.dropna(
        subset=["Machine_ID", "Combined_Timestamp"]
    )

    # Extract calendar date.
    working["Date"] = (
        working["Combined_Timestamp"].dt.normalize()
    )

    # Use named aggregation to avoid MultiIndex columns.
    daily = working.groupby(
        ["Machine_ID", "Date"],
        as_index=False
    ).agg(
        Temperature_C_mean=("Temperature_C", "mean"),
        Temperature_C_std=("Temperature_C", "std"),
        Temperature_C_count=("Temperature_C", "count"),

        Vibration_Hz_mean=("Vibration_Hz", "mean"),
        Vibration_Hz_std=("Vibration_Hz", "std"),
        Vibration_Hz_count=("Vibration_Hz", "count"),

        Power_Consumption_kW_mean=(
            "Power_Consumption_kW", "mean"
        ),
        Power_Consumption_kW_std=(
            "Power_Consumption_kW", "std"
        ),
        Power_Consumption_kW_count=(
            "Power_Consumption_kW", "count"
        ),
    )

    daily["Date"] = pd.to_datetime(daily["Date"])

    daily = daily.sort_values(
        by=["Machine_ID", "Date"]
    ).reset_index(drop=True)

    # Validate the resulting structure.
    required_output_columns = [
        "Machine_ID",
        "Date",
        "Temperature_C_mean",
        "Temperature_C_std",
        "Temperature_C_count",
        "Vibration_Hz_mean",
        "Vibration_Hz_std",
        "Vibration_Hz_count",
        "Power_Consumption_kW_mean",
        "Power_Consumption_kW_std",
        "Power_Consumption_kW_count",
    ]

    missing = [
        col for col in required_output_columns
        if col not in daily.columns
    ]

    if missing:
        raise ValueError(
            f"Daily aggregation is missing columns: {missing}"
        )

    print("\nDAILY AGGREGATION VALIDATION")
    print("-" * 50)
    print("Daily records:", len(daily))
    print("Output columns:", daily.columns.tolist())

    return daily



# --------------------------------------------------
# 5. MAIN EXECUTION
# --------------------------------------------------

def calculate_stability(daily):

    results = []

    for machine_id, group in daily.groupby("Machine_ID"):

        group = group.sort_values("Date").reset_index(drop=True)

        for sensor in SENSOR_COLUMNS:

            mean_col = f"{sensor}_mean"
            std_col = f"{sensor}_std"
            count_col = f"{sensor}_count"

            valid = group.dropna(subset=[mean_col]).copy()

            if valid.empty:
                continue

            values = valid[mean_col].to_numpy(dtype=float)

            overall_mean = np.mean(values)

            daily_std = (
                np.std(values, ddof=1)
                if len(values) > 1
                else np.nan
            )

            daily_cv = (
                daily_std / abs(overall_mean) * 100
                if pd.notna(daily_std) and overall_mean != 0
                else np.nan
            )

            quarter_size = max(
                1,
                int(np.ceil(len(values) * 0.25))
            )

            first_mean = np.mean(values[:quarter_size])
            last_mean = np.mean(values[-quarter_size:])

            change_percent = (
                (last_mean - first_mean)
                / abs(first_mean) * 100
                if first_mean != 0
                else np.nan
            )

            elapsed_days = (
                (valid["Date"] - valid["Date"].min())
                .dt.total_seconds()
                .to_numpy(dtype=float) / 86400
            )

            if len(values) >= 2 and np.unique(elapsed_days).size >= 2:
                trend_slope = np.polyfit(
                    elapsed_days, values, 1
                )[0]
            else:
                trend_slope = np.nan

            autocorrelation = (
                pd.Series(values).autocorr(lag=1)
                if len(values) >= 3 and np.std(values) > 0
                else np.nan
            )

            results.append({
                "Machine_ID": machine_id,
                "Sensor": sensor,
                "Valid_Days": len(values),
                "Total_Readings": int(valid[count_col].sum()),
                "Average_Daily_Mean": overall_mean,
                "Std_Dev_Daily_Mean": daily_std,
                "CV_Daily_Mean_Percent": daily_cv,
                "First_Quarter_Mean": first_mean,
                "Last_Quarter_Mean": last_mean,
                "First_to_Last_Change_Percent": change_percent,
                "Trend_Per_Day": trend_slope,
                "Lag1_Autocorrelation": autocorrelation,
                "Average_Within_Day_Std_Dev": valid[std_col].mean(),
            })

    return pd.DataFrame(results)

def main():

    df = load_data()

    print("\nCALCULATING DAILY SENSOR AVERAGES...")

    daily = calculate_daily_averages(df)

    print("\nCALCULATING TEMPORAL STABILITY...")

    stability = calculate_stability(daily)

    daily_path = (
        OUTPUT_DIR / "machine_sensor_daily_averages.csv"
    )

    stability_path = (
        OUTPUT_DIR / "machine_sensor_stability.csv"
    )

    daily.to_csv(daily_path, index=False)
    stability.to_csv(stability_path, index=False)

    print("\nANALYSIS COMPLETE")
    print("-" * 50)

    print("Daily machine records:", len(daily))
    print("Machine-sensor combinations:", len(stability))

    print("\nHIGHEST DAILY VARIABILITY")

    for sensor in SENSOR_COLUMNS:

        subset = stability[
            stability["Sensor"] == sensor
        ]

        print(f"\n{sensor}")

        print(
            subset.nlargest(
                5,
                "CV_Daily_Mean_Percent"
            )[
                [
                    "Machine_ID",
                    "Valid_Days",
                    "CV_Daily_Mean_Percent",
                    "First_to_Last_Change_Percent",
                    "Trend_Per_Day",
                ]
            ].to_string(index=False)
        )

    print("\nLARGEST ABSOLUTE TEMPORAL TRENDS")

    for sensor in SENSOR_COLUMNS:

        subset = stability[
            stability["Sensor"] == sensor
        ].copy()

        subset["Absolute_Trend"] = (
            subset["Trend_Per_Day"].abs()
        )

        print(f"\n{sensor}")

        print(
            subset.nlargest(
                5, "Absolute_Trend"
            )[
                [
                    "Machine_ID",
                    "Trend_Per_Day",
                    "First_to_Last_Change_Percent",
                ]
            ].to_string(index=False)
        )

    print("\nFILES SAVED:")
    print(daily_path)
    print(stability_path)


if __name__ == "__main__":
    main()
