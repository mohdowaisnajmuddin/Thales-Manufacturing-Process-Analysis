"""
KPI calculations for the Manufacturing Process Analysis project.

Important:
- These are descriptive analytics, not live factory monitoring.
- Defect_Rate and Error_Rate are treated as percentages supplied by the dataset.
- Sensor Deviation Index is exploratory, not a validated machine-health or
  safety score.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. Dataset configuration
# ---------------------------------------------------------

REQUIRED_COLUMNS = [
    "Machine_ID",
    "Operation_Mode",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
    "Efficiency_Status",
]

SENSOR_COLUMNS = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
]


# ---------------------------------------------------------
# 2. Prepare and validate the data
# ---------------------------------------------------------

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate column availability, convert numeric columns,
    and construct Combined_Timestamp when possible.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    data = df.copy()

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Construct timestamp using the actual CSV column names.
    if "Combined_Timestamp" not in data.columns:
        if "Date" in data.columns and "Timestamp" in data.columns:
            timestamp_text = (
                data["Date"].astype(str).str.strip()
                + " "
                + data["Timestamp"].astype(str).str.strip()
            )

            data["Combined_Timestamp"] = pd.to_datetime(
                timestamp_text,
                format="%d-%m-%Y %H:%M:%S",
                errors="coerce",
            )

    elif not pd.api.types.is_datetime64_any_dtype(
        data["Combined_Timestamp"]
    ):
        data["Combined_Timestamp"] = pd.to_datetime(
            data["Combined_Timestamp"],
            errors="coerce",
        )

    numeric_columns = [
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Quality_Control_Defect_Rate_%",
        "Production_Speed_units_per_hr",
        "Predictive_Maintenance_Score",
        "Error_Rate_%",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    return data


# ---------------------------------------------------------
# 3. Exploratory Sensor Deviation Index
# ---------------------------------------------------------

def add_sensor_deviation_index(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calculate a relative sensor-deviation score.

    Method:
    1. Calculate the median and IQR-based scale for each sensor.
    2. Calculate absolute robust z-scores.
    3. Average the available sensor deviations for each row.
    4. Convert the raw score to a percentile relative to the
       reference dataset.

    Interpretation:
    A higher score means the combined sensor readings are more
    unusual relative to the reference dataset.

    This is NOT a failure probability, safety score, or validated
    machine-health diagnosis.
    """

    data = prepare_data(df)

    reference = (
        prepare_data(reference_df)
        if reference_df is not None
        else data.copy()
    )

    available_sensors = [
        column for column in SENSOR_COLUMNS
        if column in data.columns
        and column in reference.columns
    ]

    if not available_sensors:
        data["Sensor_Deviation_Raw"] = np.nan
        data["Exploratory_Sensor_Deviation_Index"] = np.nan
        return data

    reference_deviations = []

    # Calculate robust deviations using reference-data statistics.
    for column in available_sensors:
        reference_values = reference[column].dropna()

        if reference_values.empty:
            continue

        median = reference_values.median()
        q1 = reference_values.quantile(0.25)
        q3 = reference_values.quantile(0.75)

        # IQR / 1.349 approximates standard deviation for
        # normally distributed data.
        scale = (q3 - q1) / 1.349

        if not np.isfinite(scale) or scale <= 0:
            scale = reference_values.std()

        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        data[f"{column}_Robust_Deviation"] = (
            (data[column] - median).abs() / scale
        )

        reference_deviations.append(
            (reference_values - median).abs() / scale
        )

    deviation_columns = [
        f"{column}_Robust_Deviation"
        for column in available_sensors
        if f"{column}_Robust_Deviation" in data.columns
    ]

    if not deviation_columns:
        data["Sensor_Deviation_Raw"] = np.nan
        data["Exploratory_Sensor_Deviation_Index"] = np.nan
        return data

    data["Sensor_Deviation_Raw"] = data[
        deviation_columns
    ].mean(axis=1)

    # Build the reference distribution of raw deviation scores.
    reference_frame = pd.concat(
        reference_deviations,
        axis=1,
    )

    reference_raw_scores = reference_frame.mean(axis=1).dropna()

    if reference_raw_scores.empty:
        data["Exploratory_Sensor_Deviation_Index"] = np.nan
        return data

    sorted_reference = np.sort(
        reference_raw_scores.to_numpy()
    )

    row_scores = data["Sensor_Deviation_Raw"].to_numpy()

    valid_mask = np.isfinite(row_scores)

    percentile_scores = np.full(
        len(data),
        np.nan,
        dtype=float,
    )

    percentile_scores[valid_mask] = (
        np.searchsorted(
            sorted_reference,
            row_scores[valid_mask],
            side="right",
        )
        / len(sorted_reference)
    ) * 100

    data["Exploratory_Sensor_Deviation_Index"] = (
        percentile_scores
    )

    return data


# ---------------------------------------------------------
# 4. Overall KPIs
# ---------------------------------------------------------

def calculate_overall_kpis(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> dict:
    """
    Calculate descriptive KPIs for the selected dataset.
    """

    data = prepare_data(df)

    if data.empty:
        raise ValueError("Cannot calculate KPIs for an empty dataset.")

    data = add_sensor_deviation_index(
        data,
        reference_df=reference_df,
    )

    # Production-weighted rates are used as descriptive KPI summaries.
    # They are percentages, not observed unit/event counts.
    production_weights = data["Production_Speed_units_per_hr"].clip(lower=0)

    def production_weighted_rate(rate_column: str) -> float:
        rate = data[rate_column]
        valid = production_weights.notna() & rate.notna()
        weights = production_weights.loc[valid]
        denominator = weights.sum()
        if denominator <= 0:
            return np.nan
        return float((weights * rate.loc[valid]).sum() / denominator)

    weighted_defect_rate = production_weighted_rate(
        "Quality_Control_Defect_Rate_%"
    )
    weighted_error_rate = production_weighted_rate("Error_Rate_%")

    kpis = {
        "total_records": int(len(data)),

        "total_machines": int(
            data["Machine_ID"].nunique()
        ),

        "average_production_speed": (
            data["Production_Speed_units_per_hr"].mean()
        ),

        "median_production_speed": (
            data["Production_Speed_units_per_hr"].median()
        ),

        "average_defect_rate": (
            data["Quality_Control_Defect_Rate_%"].mean()
        ),

        "median_defect_rate": (
            data["Quality_Control_Defect_Rate_%"].median()
        ),

        "average_error_rate": (
            data["Error_Rate_%"].mean()
        ),

        "defect_density_score_weighted_pct": weighted_defect_rate,

        "error_frequency_index_weighted_pct": weighted_error_rate,

        "median_error_rate": (
            data["Error_Rate_%"].median()
        ),

        "average_temperature": (
            data["Temperature_C"].mean()
        ),

        "average_vibration": (
            data["Vibration_Hz"].mean()
        ),

        "average_power_consumption": (
            data["Power_Consumption_kW"].mean()
        ),

        "average_network_latency": (
            data["Network_Latency_ms"].mean()
        ),

        "average_packet_loss": (
            data["Packet_Loss_%"].mean()
        ),

        "average_sensor_deviation_index": (
            data["Exploratory_Sensor_Deviation_Index"].mean()
        ),
    }

    return kpis


# ---------------------------------------------------------
# 5. Efficiency status distribution
# ---------------------------------------------------------

def calculate_efficiency_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate record counts and percentages for each
    supplied Efficiency_Status category.

    These labels are not treated as independent ground truth.
    """

    data = prepare_data(df)

    distribution = (
        data["Efficiency_Status"]
        .value_counts(dropna=False)
        .rename_axis("Efficiency_Status")
        .reset_index(name="Record_Count")
    )

    total_records = distribution["Record_Count"].sum()

    distribution["Percentage"] = (
        distribution["Record_Count"]
        / total_records
        * 100
    )

    return distribution


# ---------------------------------------------------------
# 6. Machine-level operational summary
# ---------------------------------------------------------

def calculate_machine_summary(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calculate descriptive metrics for each machine.

    Means are record-weighted. Because timestamps are irregular,
    they should not be interpreted as time-weighted averages.
    """

    data = prepare_data(df)

    data = add_sensor_deviation_index(
        data,
        reference_df=reference_df,
    )

    summary = (
        data.groupby("Machine_ID")
        .agg(
            Record_Count=("Machine_ID", "size"),

            Average_Production_Speed=(
                "Production_Speed_units_per_hr",
                "mean",
            ),

            Median_Production_Speed=(
                "Production_Speed_units_per_hr",
                "median",
            ),

            Average_Defect_Rate=(
                "Quality_Control_Defect_Rate_%",
                "mean",
            ),

            Average_Error_Rate=(
                "Error_Rate_%",
                "mean",
            ),

            Average_Temperature=(
                "Temperature_C",
                "mean",
            ),

            Average_Vibration=(
                "Vibration_Hz",
                "mean",
            ),

            Average_Power_Consumption=(
                "Power_Consumption_kW",
                "mean",
            ),

            Average_Network_Latency=(
                "Network_Latency_ms",
                "mean",
            ),

            Average_Packet_Loss=(
                "Packet_Loss_%",
                "mean",
            ),

            Average_Sensor_Deviation_Index=(
                "Exploratory_Sensor_Deviation_Index",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary.sort_values(
        "Average_Production_Speed",
        ascending=False,
    )


# ---------------------------------------------------------
# 7. Operation-mode summary
# ---------------------------------------------------------

def calculate_operation_mode_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare descriptive production and quality metrics
    across operation modes.
    """

    data = prepare_data(df)

    summary = (
        data.groupby("Operation_Mode")
        .agg(
            Record_Count=("Operation_Mode", "size"),

            Average_Production_Speed=(
                "Production_Speed_units_per_hr",
                "mean",
            ),

            Average_Defect_Rate=(
                "Quality_Control_Defect_Rate_%",
                "mean",
            ),

            Average_Error_Rate=(
                "Error_Rate_%",
                "mean",
            ),

            Average_Temperature=(
                "Temperature_C",
                "mean",
            ),

            Average_Vibration=(
                "Vibration_Hz",
                "mean",
            ),

            Average_Network_Latency=(
                "Network_Latency_ms",
                "mean",
            ),

            Average_Packet_Loss=(
                "Packet_Loss_%",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


# ---------------------------------------------------------
# 8. Date range summary
# ---------------------------------------------------------

def calculate_date_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Return the observed timestamp range when timestamps exist.
    """

    data = prepare_data(df)

    if "Combined_Timestamp" not in data.columns:
        return {
            "start_date": None,
            "end_date": None,
        }

    valid_timestamps = data[
        "Combined_Timestamp"
    ].dropna()

    if valid_timestamps.empty:
        return {
            "start_date": None,
            "end_date": None,
        }

    return {
        "start_date": valid_timestamps.min(),
        "end_date": valid_timestamps.max(),
    }