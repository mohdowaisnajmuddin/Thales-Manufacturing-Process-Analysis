
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"

ANOMALY_PATH = (
    ROOT / "data" / "processed" / "machine_anomaly_results.csv"
)

from src.kpi_calculations import (
    calculate_overall_kpis,
    calculate_efficiency_distribution,
    calculate_machine_summary,
    calculate_operation_mode_summary,
    add_sensor_deviation_index,
)

from src.kpi_visualizations import (
    production_speed_by_machine,
    defect_rate_by_machine,
    error_rate_by_machine,
    efficiency_distribution_chart,
    production_speed_by_operation_mode,
    all_sensor_distribution_charts,
    machine_comparison_chart,
)


# =========================================================
# 2. PAGE CONFIGURATION AND DARK THEME
# =========================================================

st.set_page_config(
    page_title="Manufacturing Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #000000;
        color: #F1F5F9;
    }

    [data-testid="stSidebar"] {
        background-color: #10141C;
    }

    [data-testid="stSidebar"] * {
        color: #E2E8F0;
    }

    h1, h2, h3, h4 {
        color: #F8FAFC !important;
    }

    p, label, li {
        color: #CBD5E1;
    }

    [data-testid="stMetric"] {
        background-color: #151922;
        border: 1px solid #293344;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    button[data-baseweb="tab"] {
        color: #CBD5E1 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60A5FA !important;
        border-bottom-color: #3B82F6 !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #E2E8F0 !important;
    }

    .stCaption {
        color: #94A3B8 !important;
    }

    [data-testid="stDataFrame"] {
        background-color: #151922;
    }

    hr {
        border-color: #293344;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. CONSTANTS
# =========================================================

REQUIRED_COLUMNS = [
    "Date",
    "Timestamp",
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

QUALITY_COLUMNS = [
    "Quality_Control_Defect_Rate_%",
    "Error_Rate_%",
]


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def display_number(value, decimals=2, suffix=""):
    """Safely format a numeric value."""

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):,.{decimals}f}{suffix}"


def show_chart(chart, height=None):
    """Display Plotly charts consistently."""

    if height is not None:
        chart.update_layout(height=height)

    st.plotly_chart(
        chart,
        use_container_width=True,
    )


def normalize_machine_id(series):
    """
    Create consistent machine IDs for filtering.

    This does not change the original Machine_ID column.
    """

    numeric = pd.to_numeric(series, errors="coerce")

    result = series.astype(str).str.strip()

    numeric_mask = numeric.notna()

    result.loc[numeric_mask] = (
        numeric.loc[numeric_mask]
        .map(lambda value: str(int(value)))
    )

    return result


def apply_dashboard_filters(
    data,
    selected_machines,
    selected_modes,
    date_range,
    time_range=None,
):
    """Apply machine, operation-mode, calendar-date and time-of-day filters."""

    result = data.copy()

    machine_keys = normalize_machine_id(
        result["Machine_ID"]
    )

    selected_machine_keys = {
        str(value).strip()
        for value in selected_machines
    }

    # Normalize selected numeric IDs too.
    normalized_selected = set()

    for value in selected_machine_keys:
        try:
            numeric_value = float(value)

            if numeric_value.is_integer():
                normalized_selected.add(str(int(numeric_value)))
            else:
                normalized_selected.add(value)

        except (TypeError, ValueError):
            normalized_selected.add(value)

    result = result[
        machine_keys.isin(normalized_selected)
        & result["Operation_Mode"].isin(selected_modes)
    ].copy()

    # Handle a date range.
    if isinstance(date_range, (tuple, list)):

        if len(date_range) == 2:
            start_date, end_date = date_range

            if start_date is not None and end_date is not None:
                result = result[
                    result["Combined_Timestamp"].notna()
                    & result["Combined_Timestamp"].dt.date.between(
                        start_date,
                        end_date,
                    )
                ]

        elif len(date_range) == 1 and date_range[0] is not None:
            selected_date = date_range[0]

            result = result[
                result["Combined_Timestamp"].notna()
                & (
                    result["Combined_Timestamp"].dt.date
                    == selected_date
                )
            ]

    elif date_range is not None:
        result = result[
            result["Combined_Timestamp"].notna()
            & (
                result["Combined_Timestamp"].dt.date
                == date_range
            )
        ]

    # Optional time-of-day filter. A range crossing midnight is supported.
    if time_range is not None and len(time_range) == 2:
        start_time, end_time = time_range
        if start_time is not None and end_time is not None:
            clock = result["Combined_Timestamp"].dt.time
            if start_time <= end_time:
                time_mask = (clock >= start_time) & (clock <= end_time)
            else:
                time_mask = (clock >= start_time) | (clock <= end_time)
            result = result[result["Combined_Timestamp"].notna() & time_mask]

    return result


def identify_anomaly_column(data):
    """
    Find the saved Isolation Forest prediction column.

    The anomaly detection script may use different naming
    conventions. This function checks known alternatives.
    """

    candidates = [
        "IsolationForest_Anomaly",
        "Isolation_Forest_Anomaly",
        "IF_Anomaly",
        "Isolation_Forest",
        "IsolationForest",
        "IF_Prediction",
        "Anomaly_Label",
    ]

    for column in candidates:
        if column in data.columns:
            return column

    raise ValueError(
        "Could not identify the Isolation Forest prediction column. "
        f"Available columns: {list(data.columns)}"
    )


def convert_anomaly_labels(series):
    """
    Convert saved anomaly predictions into boolean values.

    Important:
    -1 = anomaly in scikit-learn IsolationForest
     1 = normal observation
     1 = anomaly only for a separate 0/1 binary label
    """

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")

    numeric_values = set(
        numeric.dropna().unique().tolist()
    )

    # Standard scikit-learn Isolation Forest output.
    if numeric_values and numeric_values.issubset({-1, 1}):
        return numeric.eq(-1).fillna(False)

    # Binary 0/1 anomaly indicator.
    if numeric_values and numeric_values.issubset({0, 1}):
        return numeric.eq(1).fillna(False)

    # Text-based labels.
    normalized = series.astype(str).str.strip().str.lower()

    return normalized.isin(
        [
            "true",
            "yes",
            "anomaly",
            "outlier",
            "flagged",
            "-1",
        ]
    )


def prepare_plotly_dark(fig, height=None):
    """Apply the dashboard dark theme to a Plotly figure."""

    layout_options = {
        "template": "plotly_dark",
        "paper_bgcolor": "#000000",
        "plot_bgcolor": "#151922",
        "font": {"color": "#F1F5F9"},
    }

    if height is not None:
        layout_options["height"] = height

    fig.update_layout(**layout_options)

    return fig


# =========================================================
# 5. LOAD RAW DATASET
# =========================================================

@st.cache_data
def load_data(path, modified_time):

    data = pd.read_csv(path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required dataset columns are missing: "
            + ", ".join(missing_columns)
        )

    data["Combined_Timestamp"] = pd.to_datetime(
        data["Date"].astype(str).str.strip()
        + " "
        + data["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )

    return data


# =========================================================
# 6. LOAD PRECOMPUTED ANOMALY RESULTS
# =========================================================

@st.cache_data
def load_anomaly_data(path, modified_time):

    anomaly_df = pd.read_csv(path)

    required = [
        "Machine_ID",
        "Operation_Mode",
    ]

    missing = [
        column
        for column in required
        if column not in anomaly_df.columns
    ]

    if missing:
        raise ValueError(
            f"Anomaly file is missing required columns: {missing}. "
            f"Available columns: {list(anomaly_df.columns)}"
        )

    anomaly_column = identify_anomaly_column(anomaly_df)

    # Create a timestamp.
    if "Combined_Timestamp" in anomaly_df.columns:

        anomaly_df["Combined_Timestamp"] = pd.to_datetime(
            anomaly_df["Combined_Timestamp"],
            errors="coerce",
        )

    elif {"Date", "Timestamp"}.issubset(anomaly_df.columns):

        anomaly_df["Combined_Timestamp"] = pd.to_datetime(
            anomaly_df["Date"].astype(str).str.strip()
            + " "
            + anomaly_df["Timestamp"].astype(str).str.strip(),
            format="%d-%m-%Y %H:%M:%S",
            errors="coerce",
        )

    else:
        raise ValueError(
            "Anomaly results must contain Combined_Timestamp "
            "or both Date and Timestamp."
        )

    anomaly_df["Is_Anomaly"] = convert_anomaly_labels(
        anomaly_df[anomaly_column]
    )

    anomaly_df["Machine_ID"] = pd.to_numeric(
        anomaly_df["Machine_ID"],
        errors="coerce",
    )

    anomaly_df = anomaly_df.dropna(
        subset=[
            "Machine_ID",
            "Combined_Timestamp",
        ]
    ).copy()

    return anomaly_df


# =========================================================
# 7. DASHBOARD HEADING
# =========================================================

st.title(
    "Manufacturing Process Health & Operational Efficiency"
)

st.caption(
    "Unified Mentor internship project | "
    "Historical exploratory analytics"
)


# =========================================================
# 8. CHECK AND LOAD DATA FILES
# =========================================================

if not DATA_PATH.exists():

    st.error(
        f"Manufacturing dataset not found: {DATA_PATH}"
    )

    st.stop()


if not ANOMALY_PATH.exists():

    st.error(
        f"Anomaly results file not found: {ANOMALY_PATH}"
    )

    st.info(
        "Run src/machine_anomaly_detection.py before opening "
        "the dashboard."
    )

    st.stop()


try:

    df = load_data(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime,
    )

    anomaly_df = load_anomaly_data(
        str(ANOMALY_PATH),
        ANOMALY_PATH.stat().st_mtime,
    )

except Exception as error:

    st.error(
        f"Could not load dashboard data: {error}"
    )

    st.stop()


if df.empty:

    st.error("The manufacturing dataset contains no records.")

    st.stop()


if anomaly_df.empty:

    st.error("The anomaly results file contains no valid records.")

    st.stop()


# =========================================================
# 9. SIDEBAR FILTERS
# =========================================================

st.sidebar.title("Dashboard Filters")

machines = sorted(
    df["Machine_ID"].dropna().unique().tolist()
)

modes = sorted(
    df["Operation_Mode"].dropna().unique().tolist()
)

valid_timestamps = df["Combined_Timestamp"].dropna()

if not machines or not modes:

    st.error(
        "Machine IDs or operation modes are unavailable."
    )

    st.stop()


if valid_timestamps.empty:

    st.error(
        "No valid timestamps were found in the manufacturing dataset."
    )

    st.stop()


minimum_date = valid_timestamps.min().date()
maximum_date = valid_timestamps.max().date()


with st.sidebar:

    st.subheader("Machine selection")

    selected_machines = st.multiselect(
        "Select machines",
        options=machines,
        default=machines,
    )

    st.subheader("Operation mode")

    selected_modes = st.multiselect(
        "Select operation modes",
        options=modes,
        default=modes,
    )

    st.subheader("Date range")

    date_range = st.date_input(
        "Select observation dates",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )

    st.subheader("Time of day")
    time_start = st.time_input("Start time", value=pd.to_datetime("00:00").time())
    time_end = st.time_input("End time", value=pd.to_datetime("23:59").time())
    selected_time_range = (time_start, time_end)

    st.subheader("Metric comparison")
    enable_metric_comparison = st.checkbox(
        "Enable two-metric comparison explorer", value=False
    )
    comparison_options = {
        "Production speed": "Production_Speed_units_per_hr",
        "Defect rate": "Quality_Control_Defect_Rate_%",
        "Error rate": "Error_Rate_%",
        "Temperature": "Temperature_C",
        "Vibration": "Vibration_Hz",
        "Power consumption": "Power_Consumption_kW",
        "Network latency": "Network_Latency_ms",
        "Packet loss": "Packet_Loss_%",
    }
    comparison_x_label = st.selectbox(
        "X-axis metric", list(comparison_options), index=0,
        disabled=not enable_metric_comparison,
    )
    comparison_y_label = st.selectbox(
        "Y-axis metric", list(comparison_options), index=1,
        disabled=not enable_metric_comparison,
    )

    st.divider()

    st.caption(
        "Dashboard metrics and charts use the selected records."
    )


# =========================================================
# 10. APPLY FILTERS
# =========================================================

selected_data = df[
    df["Machine_ID"].isin(selected_machines)
    & df["Operation_Mode"].isin(selected_modes)
].copy()


invalid_timestamp_count = int(
    selected_data["Combined_Timestamp"].isna().sum()
)


filtered = apply_dashboard_filters(
    df,
    selected_machines,
    selected_modes,
    date_range,
    selected_time_range,
)


anomaly_filtered = apply_dashboard_filters(
    anomaly_df,
    selected_machines,
    selected_modes,
    date_range,
    selected_time_range,
)


if filtered.empty:

    st.warning(
        "No manufacturing records match the selected filters. "
        "Change the machine, operation mode, or date selection."
    )

    st.caption(
        "Records matching machine and operation filters before "
        f"date filtering: {len(selected_data):,}"
    )

    st.stop()


# =========================================================
# 11. CALCULATE KPIs AND SUMMARIES
# =========================================================

try:

    kpis = calculate_overall_kpis(
        filtered,
        reference_df=df,
    )

    efficiency_data = calculate_efficiency_distribution(
        filtered
    )

    machine_data = calculate_machine_summary(
        filtered,
        reference_df=df,
    )

    operation_data = calculate_operation_mode_summary(
        filtered
    )

    filtered_with_index = add_sensor_deviation_index(
        filtered,
        reference_df=df,
    )

except Exception as error:

    st.error(
        f"An error occurred while calculating dashboard metrics: {error}"
    )

    st.stop()


# =========================================================
# 12. SELECTED DATASET OVERVIEW
# =========================================================

st.subheader("Selected Dataset Overview")

info1, info2, info3 = st.columns(3)

info1.metric(
    "Selected records",
    f"{len(filtered):,}",
)

info2.metric(
    "Selected machines",
    f"{filtered['Machine_ID'].nunique():,}",
)

info3.metric(
    "Selected operation modes",
    f"{filtered['Operation_Mode'].nunique():,}",
)


# =========================================================
# 13. MAIN KPI CARDS
# =========================================================

st.subheader("Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Average production speed",
    display_number(
        kpis.get("average_production_speed")
    ),
    help="Mean production speed in units per hour.",
)

c2.metric(
    "Average defect rate",
    display_number(
        kpis.get("average_defect_rate"),
        suffix="%",
    ),
    help="Mean of the supplied quality-control defect-rate field.",
)

c3.metric(
    "Average error rate",
    display_number(
        kpis.get("average_error_rate"),
        suffix="%",
    ),
    help="Mean of the supplied error-rate field.",
)

c4.metric(
    "Average network latency",
    display_number(
        kpis.get("average_network_latency"),
        suffix=" ms",
    ),
    help="Mean recorded network latency.",
)


c5, c6, c7, c8 = st.columns(4)

c5.metric(
    "Average temperature",
    display_number(
        kpis.get("average_temperature"),
        suffix=" °C",
    ),
)

c6.metric(
    "Average vibration",
    display_number(
        kpis.get("average_vibration"),
        decimals=3,
        suffix=" Hz",
    ),
)

c7.metric(
    "Average power consumption",
    display_number(
        kpis.get("average_power_consumption"),
        decimals=3,
        suffix=" kW",
    ),
)

c8.metric(
    "Average packet loss",
    display_number(
        kpis.get("average_packet_loss"),
        suffix="%",
    ),
)

c9, c10 = st.columns(2)
c9.metric(
    "Defect Density Score (weighted)",
    display_number(kpis.get("defect_density_score_weighted_pct"), suffix="%"),
    help="Production-speed-weighted mean of the supplied defect percentage; not a defective-unit count.",
)
c10.metric(
    "Error Frequency Index (weighted)",
    display_number(kpis.get("error_frequency_index_weighted_pct"), suffix="%"),
    help="Production-speed-weighted mean of the supplied error percentage; not an event count.",
)


# =========================================================
# 14. DASHBOARD TABS
# =========================================================

(
    overview_tab,
    machine_tab,
    quality_tab,
    sensor_tab,
    anomaly_tab,
    data_tab,
) = st.tabs(
    [
        "Overview",
        "Machine Analysis",
        "Quality Analysis",
        "Sensor & Network Analysis",
        "Anomaly Detection",
        "Data Quality",
    ]
)


# =========================================================
# 15. OVERVIEW TAB
# =========================================================

with overview_tab:

    st.subheader("Operational Overview")

    left, right = st.columns(2)

    with left:

        show_chart(
            efficiency_distribution_chart(
                efficiency_data
            )
        )

    with right:

        show_chart(
            production_speed_by_operation_mode(
                operation_data
            )
        )

    st.subheader("Operation Mode Summary")

    st.dataframe(
        operation_data.round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Operation-mode comparisons are descriptive. "
        "They do not establish that a particular operating mode "
        "causes higher or lower production performance."
    )

    if enable_metric_comparison:
        x_col = comparison_options[comparison_x_label]
        y_col = comparison_options[comparison_y_label]
        if x_col == y_col:
            st.warning("Choose two different metrics for comparison.")
        else:
            comparison_sample = filtered[[x_col, y_col, "Machine_ID", "Operation_Mode"]].dropna()
            comparison_sample = comparison_sample.sample(
                n=min(12000, len(comparison_sample)), random_state=42
            ) if not comparison_sample.empty else comparison_sample
            st.subheader("Selected Metric Comparison")
            if comparison_sample.empty:
                st.info("No complete records are available for the selected metrics.")
            else:
                comparison_fig = px.scatter(
                    comparison_sample, x=x_col, y=y_col,
                    color="Operation_Mode", hover_data=["Machine_ID"],
                    opacity=0.55,
                    labels={x_col: comparison_x_label, y_col: comparison_y_label},
                    title=f"{comparison_x_label} vs {comparison_y_label}",
                )
                prepare_plotly_dark(comparison_fig, height=500)
                show_chart(comparison_fig)
                st.caption(
                    "Exploratory pairwise comparison using up to 12,000 selected records. "
                    "Associations are descriptive and do not establish causation."
                )


# =========================================================
# 16. MACHINE ANALYSIS TAB
# =========================================================

with machine_tab:

    st.subheader("Machine-Level Performance")

    show_chart(
        production_speed_by_machine(
            machine_data
        )
    )

    show_chart(
        machine_comparison_chart(
            machine_data
        )
    )

    st.subheader("Machine Production Ranking")

    ranking_columns = [
        "Machine_ID",
        "Record_Count",
        "Average_Production_Speed",
        "Median_Production_Speed",
        "Average_Defect_Rate",
        "Average_Error_Rate",
    ]

    available_ranking_columns = [
        column
        for column in ranking_columns
        if column in machine_data.columns
    ]

    st.dataframe(
        machine_data[available_ranking_columns].round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Production ranking is based only on average production speed. "
        "It is not an overall machine-health or efficiency ranking."
    )


# =========================================================
# 17. QUALITY ANALYSIS TAB
# =========================================================

with quality_tab:

    st.subheader("Quality and Error Analysis")

    left, right = st.columns(2)

    with left:

        show_chart(
            defect_rate_by_machine(
                machine_data
            )
        )

    with right:

        show_chart(
            error_rate_by_machine(
                machine_data
            )
        )

    st.divider()

    st.subheader(
        "Production Speed vs Quality Control Defect Rate"
    )

    scatter_data = filtered.sample(
        n=min(12000, len(filtered)),
        random_state=42,
    )

    fig = px.scatter(
        scatter_data,
        x="Production_Speed_units_per_hr",
        y="Quality_Control_Defect_Rate_%",
        color="Efficiency_Status",
        color_discrete_map={
            "High": "#16A34A",
            "Medium": "#F59E0B",
            "Low": "#DC2626",
        },
        hover_data=[
            "Machine_ID",
            "Operation_Mode",
            "Error_Rate_%",
        ],
        opacity=0.55,
        title="Production Speed vs Quality Control Defect Rate",
        labels={
            "Production_Speed_units_per_hr": (
                "Production Speed (units/hour)"
            ),
            "Quality_Control_Defect_Rate_%": "Defect Rate (%)",
            "Efficiency_Status": "Supplied Efficiency Label",
        },
    )

    fig.update_traces(
        marker={"size": 5}
    )

    prepare_plotly_dark(
        fig,
        height=550,
    )

    fig.update_layout(
        xaxis_title="Production Speed (units/hour)",
        yaxis_title="Defect Rate (%)",
        legend_title="Supplied Efficiency Label",
    )

    show_chart(fig)

    st.caption(
        "The scatter plot uses a reproducible sample of up to "
        "12,000 observations for readability. KPI calculations use "
        "all filtered records. Efficiency labels are derived from "
        "production speed and error rate, so their colors are not "
        "independent evidence of machine health."
    )


# =========================================================
# 18. SENSOR AND NETWORK ANALYSIS TAB
# =========================================================

with sensor_tab:

    st.subheader("Sensor Measurements")

    try:

        sensor_charts = all_sensor_distribution_charts(
            filtered
        )

        sensor_names = list(sensor_charts.keys())

        if not sensor_names:

            st.info(
                "No sensor charts are available for these records."
            )

        for index in range(0, len(sensor_names), 2):

            left, right = st.columns(2)

            with left:

                name = sensor_names[index]

                show_chart(
                    sensor_charts[name]
                )

            if index + 1 < len(sensor_names):

                with right:

                    name = sensor_names[index + 1]

                    show_chart(
                        sensor_charts[name]
                    )

    except Exception as error:

        st.error(
            f"Could not generate sensor charts: {error}"
        )

    st.divider()

    st.subheader("Exploratory Sensor Deviation")

    deviation_column = "Exploratory_Sensor_Deviation_Index"

    if (
        deviation_column in filtered_with_index.columns
        and filtered_with_index[deviation_column].notna().any()
    ):

        mean_deviation = filtered_with_index[
            deviation_column
        ].mean()

        st.metric(
            "Mean Sensor Deviation Index",
            display_number(mean_deviation),
        )

    else:

        st.metric(
            "Mean Sensor Deviation Index",
            "Not available",
        )

    st.caption(
        "This is a relative percentile based on the supplied dataset. "
        "It is not a machine-health percentage, failure probability, "
        "or safety assessment."
    )

    network_summary = (
        filtered.groupby("Operation_Mode")
        .agg(
            Record_Count=("Operation_Mode", "size"),
            Mean_Latency_ms=("Network_Latency_ms", "mean"),
            Mean_Packet_Loss_pct=("Packet_Loss_%", "mean"),
        )
        .reset_index()
    )

    st.subheader("Network Metrics by Operation Mode")

    st.dataframe(
        network_summary.round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Network latency and packet loss are descriptive fields. "
        "The dataset does not provide a controlled comparison between "
        "4G, 5G, and 6G networks."
    )


# =========================================================
# 19. ANOMALY DETECTION TAB
# =========================================================

with anomaly_tab:

    st.subheader("Machine Anomaly Detection")

    st.caption(
        "Precomputed Isolation Forest results. "
        "An anomaly indicates an unusual observation, "
        "not a confirmed equipment failure."
    )

    if anomaly_filtered.empty:

        st.warning(
            "No anomaly records match the selected filters."
        )

    else:

        total_anomaly_records = len(anomaly_filtered)

        flagged_records = int(
            anomaly_filtered["Is_Anomaly"].sum()
        )

        normal_records = (
            total_anomaly_records - flagged_records
        )

        anomaly_percentage = (
            flagged_records / total_anomaly_records * 100
            if total_anomaly_records > 0
            else 0
        )

        # -------------------------------------------------
        # Anomaly KPI cards
        # -------------------------------------------------

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "Evaluated observations",
            f"{total_anomaly_records:,}",
        )

        a2.metric(
            "Flagged observations",
            f"{flagged_records:,}",
        )

        a3.metric(
            "Normal observations",
            f"{normal_records:,}",
        )

        a4.metric(
            "Anomaly rate",
            f"{anomaly_percentage:.2f}%",
        )

        st.divider()

                # -------------------------------------------------
        # Reference vs Evaluation comparison
        # -------------------------------------------------

        if "Reference_Period" in anomaly_filtered.columns:

            st.subheader("Reference vs Evaluation")

            period_summary = (
                anomaly_filtered.groupby("Reference_Period")
                .agg(
                    Records=("Is_Anomaly", "size"),
                    Anomalies=("Is_Anomaly", "sum"),
                )
                .reset_index()
            )

            period_summary["Normal_Records"] = (
                period_summary["Records"]
                - period_summary["Anomalies"]
            )

            period_summary["Anomaly_Rate_Pct"] = (
                period_summary["Anomalies"]
                / period_summary["Records"]
                * 100
            )

            period_fig = px.bar(
                period_summary,
                x="Reference_Period",
                y="Anomaly_Rate_Pct",
                text="Anomaly_Rate_Pct",
                title="Anomaly Rate: Reference vs Evaluation",
                labels={
                    "Reference_Period": "Dataset Period",
                    "Anomaly_Rate_Pct": "Anomaly Rate (%)",
                },
            )

            period_fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
            )

            prepare_plotly_dark(period_fig)

            show_chart(period_fig)

            st.dataframe(
                period_summary.round(3),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "The Reference period was used to fit the anomaly "
                "detector. Its anomaly rate is influenced by the "
                "configured contamination parameter (2%). The "
                "Evaluation period was not used to fit the detector. "
                "A higher evaluation rate indicates more observations "
                "fall outside the learned reference boundary; it does "
                "not establish equipment deterioration or failure."
            )

        st.divider()

        # -------------------------------------------------
        # Anomaly rate by operation mode
        # -------------------------------------------------

        st.subheader("Anomaly Rate by Operation Mode")

        mode_anomalies = (
            anomaly_filtered.groupby("Operation_Mode")
            .agg(
                Records=("Is_Anomaly", "size"),
                Anomalies=("Is_Anomaly", "sum"),
            )
            .reset_index()
        )

        mode_anomalies["Anomaly_Rate_Pct"] = (
            mode_anomalies["Anomalies"]
            / mode_anomalies["Records"]
            * 100
        )

        mode_fig = px.bar(
            mode_anomalies,
            x="Operation_Mode",
            y="Anomaly_Rate_Pct",
            text="Anomaly_Rate_Pct",
            title="Anomaly Percentage by Operation Mode",
            labels={
                "Operation_Mode": "Operation Mode",
                "Anomaly_Rate_Pct": "Anomaly Rate (%)",
            },
        )

        mode_fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )

        prepare_plotly_dark(mode_fig)

        show_chart(mode_fig)

        st.dataframe(
            mode_anomalies.round(3),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # -------------------------------------------------
        # Machine-mode anomaly ranking
        # -------------------------------------------------

        st.subheader("Machine-Level Anomaly Ranking")

        machine_mode_anomalies = (
            anomaly_filtered.groupby(
                ["Machine_ID", "Operation_Mode"]
            )
            .agg(
                Records=("Is_Anomaly", "size"),
                Anomalies=("Is_Anomaly", "sum"),
            )
            .reset_index()
        )

        machine_mode_anomalies["Anomaly_Rate_Pct"] = (
            machine_mode_anomalies["Anomalies"]
            / machine_mode_anomalies["Records"]
            * 100
        )

        machine_mode_anomalies = (
            machine_mode_anomalies.sort_values(
                "Anomaly_Rate_Pct",
                ascending=False,
            )
        )

        ranking_fig = px.bar(
            machine_mode_anomalies.head(15),
            x="Machine_ID",
            y="Anomaly_Rate_Pct",
            color="Operation_Mode",
            barmode="group",
            title="Top 15 Machine-Mode Anomaly Rates",
            labels={
                "Machine_ID": "Machine",
                "Anomaly_Rate_Pct": "Anomaly Rate (%)",
            },
        )

        prepare_plotly_dark(ranking_fig)

        show_chart(ranking_fig)

        st.dataframe(
            machine_mode_anomalies.head(20).round(3),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # -------------------------------------------------
        # Daily anomaly rate
        # -------------------------------------------------

        st.subheader("Daily Anomaly Rate")

        daily_anomalies = (
            anomaly_filtered.groupby(
                anomaly_filtered["Combined_Timestamp"].dt.date
            )
            .agg(
                Records=("Is_Anomaly", "size"),
                Anomalies=("Is_Anomaly", "sum"),
            )
            .reset_index()
        )

        daily_anomalies = daily_anomalies.rename(
            columns={
                "Combined_Timestamp": "Date",
            }
        )

        daily_anomalies["Anomaly_Rate_Pct"] = (
            daily_anomalies["Anomalies"]
            / daily_anomalies["Records"]
            * 100
        )

        daily_anomalies = daily_anomalies.sort_values(
            "Date"
        )

        daily_fig = px.line(
            daily_anomalies,
            x="Date",
            y="Anomaly_Rate_Pct",
            markers=True,
            title="Daily Isolation Forest Anomaly Rate",
            labels={
                "Date": "Date",
                "Anomaly_Rate_Pct": "Anomaly Rate (%)",
            },
        )

        prepare_plotly_dark(daily_fig)

        show_chart(daily_fig)

        st.dataframe(
            daily_anomalies.round(3),
            use_container_width=True,
            hide_index=True,
        )

        # March 1 sampling-frequency warning.
        march_1_records = anomaly_filtered[
            anomaly_filtered["Combined_Timestamp"].dt.date
            == pd.Timestamp("2025-03-01").date()
        ]

        if not march_1_records.empty:

            st.warning(
                "March 1, 2025 has a higher observation frequency "
                "than neighboring days in the supplied dataset. "
                "Interpret daily raw anomaly counts cautiously. "
                "Anomaly percentages are displayed instead."
            )

        st.divider()

        # -------------------------------------------------
        # Quality association
        # -------------------------------------------------

        st.subheader(
            "Anomaly Association with Quality Metrics"
        )

        available_quality = [
            column
            for column in QUALITY_COLUMNS
            if column in anomaly_filtered.columns
        ]

        if available_quality:

            quality_comparison = (
                anomaly_filtered.groupby("Is_Anomaly")[
                    available_quality
                ]
                .mean()
                .reset_index()
            )

            quality_comparison["Observation_Type"] = (
                quality_comparison["Is_Anomaly"]
                .map({
                    True: "Flagged",
                    False: "Normal",
                })
            )

            quality_long = quality_comparison.melt(
                id_vars=["Observation_Type"],
                value_vars=available_quality,
                var_name="Metric",
                value_name="Mean_Value",
            )

            quality_fig = px.bar(
                quality_long,
                x="Metric",
                y="Mean_Value",
                color="Observation_Type",
                barmode="group",
                title="Mean Quality Metrics: Flagged vs Normal",
                labels={
                    "Mean_Value": "Mean value (%)",
                    "Metric": "Quality Metric",
                },
            )

            prepare_plotly_dark(quality_fig)

            show_chart(quality_fig)

            st.dataframe(
                quality_comparison.round(4),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "These are pooled descriptive means. They do not "
                "control for machine or operation mode and should "
                "not be interpreted as a controlled comparison."
            )

        else:

            st.info(
                "Quality columns were not found in the anomaly file. "
                "The quality association chart is unavailable."
            )

        st.info(
            "Isolation Forest flags statistical deviations from "
            "the learned reference profiles. They are not confirmed "
            "machine failures, safety events, or maintenance diagnoses."
        )

        st.divider()

        # -------------------------------------------------
        # Export flagged observations
        # -------------------------------------------------

        st.subheader("Export Anomaly Results")

        anomaly_export = anomaly_filtered[
            anomaly_filtered["Is_Anomaly"]
        ].copy()

        st.download_button(
            label="Download flagged observations (CSV)",
            data=anomaly_export.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="flagged_machine_observations.csv",
            mime="text/csv",
        )


# =========================================================
# 20. DATA QUALITY TAB
# =========================================================

with data_tab:

    st.subheader("Data Quality Overview")

    total_rows = len(filtered)

    duplicate_rows = int(
        filtered.duplicated().sum()
    )

    missing_cells = int(
        filtered.isna().sum().sum()
    )

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Selected rows",
        f"{total_rows:,}",
    )

    d2.metric(
        "Missing cells",
        f"{missing_cells:,}",
    )

    d3.metric(
        "Exact duplicate rows",
        f"{duplicate_rows:,}",
    )

    d4.metric(
        "Invalid timestamps in machine/mode selection",
        f"{invalid_timestamp_count:,}",
    )

    st.caption(
        "Invalid timestamp count is calculated before date filtering "
        "because records without valid timestamps cannot be assigned "
        "to the selected date range."
    )

    st.subheader("Missing Values by Column")

    missing_summary = (
        filtered.isna()
        .sum()
        .reset_index()
    )

    missing_summary.columns = [
        "Column",
        "Missing_Count",
    ]

    missing_summary["Missing_Percentage"] = (
        missing_summary["Missing_Count"]
        / total_rows
        * 100
    )

    st.dataframe(
        missing_summary.sort_values(
            "Missing_Count",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Dataset Scope")

    first_timestamp = filtered["Combined_Timestamp"].min()
    last_timestamp = filtered["Combined_Timestamp"].max()

    st.write(
        f"**First observed timestamp:** {first_timestamp}"
    )

    st.write(
        f"**Last observed timestamp:** {last_timestamp}"
    )

    st.write(
        f"**Unique machines:** "
        f"{filtered['Machine_ID'].nunique()}"
    )

    st.write(
        f"**Unique operation modes:** "
        f"{filtered['Operation_Mode'].nunique()}"
    )

    st.subheader("Methodological Limitations")

    st.markdown(
        """
        - Observations are irregularly spaced in time.
        - Averages are record-weighted, not time-weighted.
        - Efficiency labels are reconstructable from other fields.
        - No validated machine failure labels are supplied.
        - No machine-specific engineering safety limits are supplied.
        - Network measurements do not establish 6G deployment.
        - Statistical anomalies are not confirmed equipment failures.
        - The dataset's attribution to Thales Group has not been
          independently verified.
        - This dashboard is historical exploratory analysis, not live
          factory monitoring.
        """
    )


# =========================================================
# 21. EXPORT SELECTED RAW DATA
# =========================================================

st.divider()

st.subheader("Export Selected Data")

csv_data = filtered.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download filtered dataset (CSV)",
    data=csv_data,
    file_name="manufacturing_filtered_data.csv",
    mime="text/csv",
)

st.caption(
    "Source: Dataset supplied for the Unified Mentor internship. "
    "Its attribution to Thales Group in the project brief has not "
    "been independently verified."
)
