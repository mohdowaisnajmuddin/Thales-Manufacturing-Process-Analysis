from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
REPORT_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GROUP_COLUMNS = ["Machine_ID", "Operation_Mode"]

DETECTION_FEATURES = [
    "Production_Speed_units_per_hr",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Predictive_Maintenance_Score",
]

OUTCOME_COLUMNS = [
    "Quality_Control_Defect_Rate_%",
    "Error_Rate_%",
]

REFERENCE_FRACTION = 0.70

# Analyst-selected screening thresholds.
# These are NOT verified engineering or safety limits.
ROBUST_Z_THRESHOLD = 3.5
ISOLATION_CONTAMINATION = 0.02

RANDOM_STATE = 42
MIN_REFERENCE_ROWS = 100


# ---------------------------------------------------------
# LOAD AND PREPARE DATA
# ---------------------------------------------------------

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = (
        GROUP_COLUMNS
        + DETECTION_FEATURES
        + OUTCOME_COLUMNS
        + ["Date", "Timestamp"]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["Combined_Timestamp"] = pd.to_datetime(
        df["Date"].astype(str)
        + " "
        + df["Timestamp"].astype(str),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )

    for column in DETECTION_FEATURES + OUTCOME_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=GROUP_COLUMNS + ["Combined_Timestamp"]
    ).copy()

    df = df.sort_values(
        ["Machine_ID", "Operation_Mode", "Combined_Timestamp"]
    ).reset_index(drop=True)

    print("\nDATASET")
    print("Rows:", len(df))
    print("Machines:", df["Machine_ID"].nunique())
    print("Operation modes:", df["Operation_Mode"].nunique())
    print(
        "Date range:",
        df["Combined_Timestamp"].min(),
        "to",
        df["Combined_Timestamp"].max(),
    )

    return df


# ---------------------------------------------------------
# CREATE CHRONOLOGICAL REFERENCE GROUPS
# ---------------------------------------------------------

def assign_reference_periods(df):

    df = df.copy()

    df["Reference_Period"] = ""

    for _, group in df.groupby(
        GROUP_COLUMNS,
        sort=False,
    ):

        indices = group.index

        split_position = int(
            len(group) * REFERENCE_FRACTION
        )

        if split_position < MIN_REFERENCE_ROWS:
            raise ValueError(
                "Insufficient reference observations for group "
                f"{group[GROUP_COLUMNS].iloc[0].to_dict()}"
            )

        df.loc[
            indices[:split_position],
            "Reference_Period",
        ] = "Reference"

        df.loc[
            indices[split_position:],
            "Reference_Period",
        ] = "Evaluation"

    return df


# ---------------------------------------------------------
# BUILD MACHINE-SPECIFIC NORMAL PROFILES
# ---------------------------------------------------------

def build_normal_profiles(df):

    profile_rows = []
    profile_lookup = {}

    for group_key, group in df.groupby(
        GROUP_COLUMNS,
        sort=True,
    ):

        machine_id, operation_mode = group_key

        reference = group[
            group["Reference_Period"] == "Reference"
        ]

        for feature in DETECTION_FEATURES:

            values = reference[feature].dropna()

            if len(values) < MIN_REFERENCE_ROWS:
                continue

            median = values.median()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            # Fallback for constant or nearly constant features.
            if iqr <= 1e-12:
                scale = values.std()

                if pd.isna(scale) or scale <= 1e-12:
                    scale = 1.0
            else:
                scale = iqr / 1.349

            profile_lookup[
                (machine_id, operation_mode, feature)
            ] = {
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "scale": scale,
                "reference_count": len(values),
            }

            profile_rows.append({
                "Machine_ID": machine_id,
                "Operation_Mode": operation_mode,
                "Feature": feature,
                "Reference_Count": len(values),
                "Reference_Median": median,
                "Reference_Q1": q1,
                "Reference_Q3": q3,
                "Reference_IQR": iqr,
                "Robust_Scale": scale,
            })

    profiles = pd.DataFrame(profile_rows)

    profiles.to_csv(
        REPORT_DIR / "machine_normal_profiles.csv",
        index=False,
    )

    return profiles, profile_lookup


# ---------------------------------------------------------
# ROBUST STATISTICAL ANOMALY DETECTION
# ---------------------------------------------------------

def calculate_robust_anomalies(df, profile_lookup):

    df = df.copy()

    robust_scores = pd.DataFrame(
        index=df.index,
        columns=DETECTION_FEATURES,
        dtype=float,
    )

    for index, row in df.iterrows():

        machine_id = row["Machine_ID"]
        operation_mode = row["Operation_Mode"]

        for feature in DETECTION_FEATURES:

            value = row[feature]

            profile = profile_lookup.get(
                (machine_id, operation_mode, feature)
            )

            if profile is None or pd.isna(value):
                continue

            score = (
                value - profile["median"]
            ) / profile["scale"]

            robust_scores.loc[index, feature] = score

    df["Robust_Max_Abs_Score"] = (
        robust_scores.abs().max(axis=1)
    )

    df["Robust_Anomaly_Feature_Count"] = (
        robust_scores.abs() > ROBUST_Z_THRESHOLD
    ).sum(axis=1)

    df["Robust_Anomaly"] = (
        df["Robust_Anomaly_Feature_Count"] > 0
    )

    return df, robust_scores


# ---------------------------------------------------------
# ISOLATION FOREST
# ---------------------------------------------------------

def run_isolation_forest(df):

    df = df.copy()

    df["Isolation_Forest_Anomaly"] = False
    df["Isolation_Forest_Score"] = np.nan

    for group_key, group in df.groupby(
        GROUP_COLUMNS,
        sort=True,
    ):

        reference = group[
            group["Reference_Period"] == "Reference"
        ]

        if len(reference) < MIN_REFERENCE_ROWS:
            continue

        pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination=ISOLATION_CONTAMINATION,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ])

        pipeline.fit(
            reference[DETECTION_FEATURES]
        )

        all_group_features = group[
            DETECTION_FEATURES
        ]

        predictions = pipeline.predict(
            all_group_features
        )

        scores = pipeline.decision_function(
            all_group_features
        )

        df.loc[
            group.index,
            "Isolation_Forest_Anomaly",
        ] = predictions == -1

        df.loc[
            group.index,
            "Isolation_Forest_Score",
        ] = scores

    return df


# ---------------------------------------------------------
# ANOMALY RATE REPORTS
# ---------------------------------------------------------

def create_anomaly_rate_reports(df):

    methods = [
        "Robust_Anomaly",
        "Isolation_Forest_Anomaly",
    ]

    machine_rows = []
    time_rows = []
    period_rows = []

    for method in methods:

        for machine_id, group in df.groupby("Machine_ID"):

            machine_rows.append({
                "Method": method,
                "Machine_ID": machine_id,
                "Records": len(group),
                "Anomaly_Count": int(group[method].sum()),
                "Anomaly_Rate_Pct": (
                    group[method].mean() * 100
                ),
            })

        daily = (
            df.groupby(
                pd.Grouper(
                    key="Combined_Timestamp",
                    freq="D",
                )
            )[method]
            .agg(
                Records="count",
                Anomaly_Count="sum",
                Anomaly_Rate="mean",
            )
            .reset_index()
        )

        daily["Method"] = method
        daily["Anomaly_Rate_Pct"] = (
            daily["Anomaly_Rate"] * 100
        )

        time_rows.append(daily)

        for period, group in df.groupby("Reference_Period"):

            period_rows.append({
                "Method": method,
                "Period": period,
                "Records": len(group),
                "Anomaly_Count": int(group[method].sum()),
                "Anomaly_Rate_Pct": (
                    group[method].mean() * 100
                ),
            })

    machine_report = pd.DataFrame(machine_rows)
    time_report = pd.concat(time_rows, ignore_index=True)
    period_report = pd.DataFrame(period_rows)

    machine_report.to_csv(
        REPORT_DIR / "machine_anomaly_rates.csv",
        index=False,
    )

    time_report.to_csv(
        REPORT_DIR / "daily_anomaly_rates.csv",
        index=False,
    )

    period_report.to_csv(
        REPORT_DIR / "anomaly_rates_by_period.csv",
        index=False,
    )

    return machine_report, time_report, period_report


# ---------------------------------------------------------
# CHECK ASSOCIATION WITH QUALITY OUTCOMES
# ---------------------------------------------------------

def analyze_quality_association(df):

    # Use evaluation period only.
    evaluation = df[
        df["Reference_Period"] == "Evaluation"
    ].copy()

    methods = [
        "Robust_Anomaly",
        "Isolation_Forest_Anomaly",
    ]

    results = []

    for method in methods:

        for outcome in OUTCOME_COLUMNS:

            flagged = evaluation.loc[
                evaluation[method],
                outcome,
            ].dropna()

            normal = evaluation.loc[
                ~evaluation[method],
                outcome,
            ].dropna()

            if len(flagged) == 0 or len(normal) == 0:
                continue

            results.append({
                "Method": method,
                "Outcome": outcome,
                "Anomaly_Records": len(flagged),
                "Normal_Records": len(normal),
                "Anomaly_Mean": flagged.mean(),
                "Normal_Mean": normal.mean(),
                "Mean_Difference": (
                    flagged.mean() - normal.mean()
                ),
                "Anomaly_Median": flagged.median(),
                "Normal_Median": normal.median(),
            })

    report = pd.DataFrame(results)

    report.to_csv(
        REPORT_DIR / "anomaly_quality_association.csv",
        index=False,
    )

    return report


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("MACHINE-SPECIFIC ANOMALY DETECTION")
    print("=" * 65)

    df = load_data()

    print("\nAssigning chronological reference periods...")
    df = assign_reference_periods(df)

    print(
        df["Reference_Period"]
        .value_counts()
        .to_string()
    )

    print("\nBuilding machine-specific normal profiles...")
    profiles, profile_lookup = build_normal_profiles(df)

    print("Profile rows:", len(profiles))

    print("\nRunning robust statistical anomaly detection...")
    df, _ = calculate_robust_anomalies(
        df,
        profile_lookup,
    )

    print("\nRunning Isolation Forest...")
    df = run_isolation_forest(df)

    print("\nCreating anomaly rate reports...")
    machine_report, time_report, period_report = (
        create_anomaly_rate_reports(df)
    )

    print("\nChecking association with quality outcomes...")
    quality_report = analyze_quality_association(df)

    output_columns = (
        GROUP_COLUMNS
        + [
            "Combined_Timestamp",
            "Reference_Period",
        ]
        + DETECTION_FEATURES
        + OUTCOME_COLUMNS
        + [
            "Robust_Max_Abs_Score",
            "Robust_Anomaly_Feature_Count",
            "Robust_Anomaly",
            "Isolation_Forest_Score",
            "Isolation_Forest_Anomaly",
        ]
    )

    df[output_columns].to_csv(
        PROCESSED_DIR / "machine_anomaly_results.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 65)

    print("\nAnomaly rates by period:")
    print(period_report.round(3).to_string(index=False))

    print("\nHighest machine anomaly rates:")
    print(
        machine_report.sort_values(
            "Anomaly_Rate_Pct",
            ascending=False,
        )
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    print("\nQuality outcome comparison:")
    print(
        quality_report.round(4).to_string(index=False)
    )

    print("\nSaved files:")
    print(
        REPORT_DIR / "machine_normal_profiles.csv"
    )
    print(
        REPORT_DIR / "machine_anomaly_rates.csv"
    )
    print(
        REPORT_DIR / "daily_anomaly_rates.csv"
    )
    print(
        REPORT_DIR / "anomaly_rates_by_period.csv"
    )
    print(
        REPORT_DIR / "anomaly_quality_association.csv"
    )
    print(
        PROCESSED_DIR / "machine_anomaly_results.csv"
    )

    print("\nCompleted.")


if __name__ == "__main__":
    main()