
"""
Interactive Plotly visualizations for the manufacturing analytics dashboard.

All charts are descriptive. They do not establish causality,
machine safety, or validated failure conditions.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------
# 1. Dashboard color configuration
# ---------------------------------------------------------

COLORS = {
    "production": "#2563EB",
    "high": "#16A34A",
    "medium": "#F59E0B",
    "low": "#DC2626",
    "network": "#7C3AED",
    "sensor": "#0891B2",
    "error": "#DC2626",
    "defect": "#F97316",
    "neutral": "#64748B",
}

EFFICIENCY_COLORS = {
    "High": COLORS["high"],
    "Medium": COLORS["medium"],
    "Low": COLORS["low"],
}

OPERATION_MODE_COLORS = {
    "Active": COLORS["production"],
    "Idle": COLORS["neutral"],
    "Maintenance": COLORS["medium"],
}

CHART_TEMPLATE = "plotly_dark"


# ---------------------------------------------------------
# 2. Common helper functions
# ---------------------------------------------------------

def _validate_columns(df, required_columns):
    """Check whether the required columns exist."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns required for visualization: {missing}"
        )


def _empty_figure(message):
    """Return an informative empty chart."""

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14},
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=350,
    )

    return fig


def _finish_chart(fig, title, y_title, height=450):
    """Apply consistent formatting to charts."""

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        yaxis_title=y_title,
        hovermode="closest",
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
        font={"family": "Arial", "size": 12},
    )

    return fig


# ---------------------------------------------------------
# 3. Production speed by machine
# ---------------------------------------------------------


def production_speed_by_machine(machine_summary):

    _validate_columns(
        machine_summary,
        ["Machine_ID", "Average_Production_Speed"],
    )

    if machine_summary.empty:
        return _empty_figure("No machine data available.")

    data = machine_summary.sort_values("Machine_ID")

    fig = px.bar(
        data,
        x="Machine_ID",
        y="Average_Production_Speed",
        title="Average Production Speed by Machine",
        labels={
            "Machine_ID": "Machine ID",
            "Average_Production_Speed": "Production Speed (units/hour)",
        },
        color_discrete_sequence=[COLORS["production"]],
        hover_data=["Average_Production_Speed"],
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Machine ID",
        yaxis_title="Production Speed (units/hour)",
        showlegend=False,
    )

    return fig



# ---------------------------------------------------------
# 4. Defect rate by machine
# ---------------------------------------------------------


def defect_rate_by_machine(machine_summary):

    _validate_columns(
        machine_summary,
        ["Machine_ID", "Average_Defect_Rate"],
    )

    if machine_summary.empty:
        return _empty_figure("No machine data available.")

    data = machine_summary.sort_values("Machine_ID")

    fig = px.bar(
        data,
        x="Machine_ID",
        y="Average_Defect_Rate",
        title="Average Quality Control Defect Rate by Machine",
        labels={
            "Machine_ID": "Machine ID",
            "Average_Defect_Rate": "Defect Rate (%)",
        },
        color_discrete_sequence=[COLORS["defect"]],
        hover_data=["Average_Defect_Rate"],
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Machine ID",
        yaxis_title="Defect Rate (%)",
        showlegend=False,
    )

    return fig



# ---------------------------------------------------------
# 5. Error rate by machine
# ---------------------------------------------------------


def error_rate_by_machine(machine_summary):

    _validate_columns(
        machine_summary,
        ["Machine_ID", "Average_Error_Rate"],
    )

    if machine_summary.empty:
        return _empty_figure("No machine data available.")

    data = machine_summary.sort_values("Machine_ID")

    fig = px.bar(
        data,
        x="Machine_ID",
        y="Average_Error_Rate",
        title="Average Error Rate by Machine",
        labels={
            "Machine_ID": "Machine ID",
            "Average_Error_Rate": "Error Rate (%)",
        },
        color_discrete_sequence=[COLORS["error"]],
        hover_data=["Average_Error_Rate"],
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Machine ID",
        yaxis_title="Error Rate (%)",
        showlegend=False,
    )

    return fig



# ---------------------------------------------------------
# 6. Efficiency status distribution
# ---------------------------------------------------------

def efficiency_distribution_chart(distribution):
    """
    Display the distribution of supplied efficiency labels.

    These labels are not independent proof of machine health.
    """

    _validate_columns(
        distribution,
        ["Efficiency_Status", "Record_Count", "Percentage"],
    )

    if distribution.empty:
        return _empty_figure("No efficiency data available.")

    fig = px.pie(
        distribution,
        names="Efficiency_Status",
        values="Record_Count",
        title="Distribution of Supplied Efficiency Status Labels",
        color="Efficiency_Status",
        color_discrete_map=EFFICIENCY_COLORS,
        hole=0.45,
        hover_data=["Percentage"],
    )

    fig.update_traces(
        textposition="inside",
        textinfo="label+percent",
        marker={"line": {"color": "white", "width": 2}},
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        height=420,
        margin={"l": 30, "r": 30, "t": 70, "b": 30},
        legend_title_text="Efficiency Status",
    )

    return fig


# ---------------------------------------------------------
# 7. Production speed by operation mode
# ---------------------------------------------------------



def production_speed_by_operation_mode(operation_summary):

    _validate_columns(
        operation_summary,
        ["Operation_Mode", "Average_Production_Speed"],
    )

    if operation_summary.empty:
        return _empty_figure("No operation-mode data available.")

    data = operation_summary.sort_values(
        "Average_Production_Speed",
        ascending=False,
    )

    fig = px.bar(
        data,
        x="Operation_Mode",
        y="Average_Production_Speed",
        title="Average Production Speed by Operation Mode",
        labels={
            "Operation_Mode": "Operation Mode",
            "Average_Production_Speed": "Production Speed (units/hour)",
        },
        color="Operation_Mode",
        color_discrete_map=OPERATION_MODE_COLORS,
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        xaxis_title="Operation Mode",
        yaxis_title="Production Speed (units/hour)",
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------
# 8. Individual sensor distribution
# ---------------------------------------------------------

def sensor_distribution_chart(df, sensor_column, title, unit):
    """
    Display a histogram for one sensor or operational metric.
    """

    _validate_columns(df, [sensor_column])

    values = pd.to_numeric(
        df[sensor_column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return _empty_figure(
            f"No valid observations available for {sensor_column}."
        )

    fig = px.histogram(
        x=values,
        nbins=40,
        title=title,
        labels={
            "x": unit,
            "count": "Observation Count",
        },
        color_discrete_sequence=[COLORS["sensor"]],
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        height=380,
        xaxis_title=unit,
        yaxis_title="Observation Count",
        showlegend=False,
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
    )

    return fig


# ---------------------------------------------------------
# 9. Collection of sensor distribution charts
# ---------------------------------------------------------

def all_sensor_distribution_charts(df):
    """
    Return a dictionary containing sensor distribution charts.
    """

    sensor_definitions = {
        "Temperature_C": (
            "Temperature Distribution",
            "Temperature (°C)",
        ),
        "Vibration_Hz": (
            "Vibration Distribution",
            "Vibration (Hz)",
        ),
        "Power_Consumption_kW": (
            "Power Consumption Distribution",
            "Power Consumption (kW)",
        ),
        "Network_Latency_ms": (
            "Network Latency Distribution",
            "Network Latency (ms)",
        ),
        "Packet_Loss_%": (
            "Network Packet Loss Distribution",
            "Packet Loss (%)",
        ),
    }

    charts = {}

    for column, (title, unit) in sensor_definitions.items():
        charts[column] = sensor_distribution_chart(
            df,
            column,
            title,
            unit,
        )

    return charts


# ---------------------------------------------------------
# 10. Machine comparison chart
# ---------------------------------------------------------

def machine_comparison_chart(machine_summary):
    """
    Compare production speed, defect rate and error rate.

    A secondary axis is used for percentage metrics.
    The metrics are not a composite score.
    """

    required_columns = [
        "Machine_ID",
        "Average_Production_Speed",
        "Average_Defect_Rate",
        "Average_Error_Rate",
    ]

    _validate_columns(machine_summary, required_columns)

    if machine_summary.empty:
        return _empty_figure("No machine comparison data available.")

    data = machine_summary.sort_values("Machine_ID")

    fig = go.Figure()

    # Production speed
    fig.add_trace(
        go.Scatter(
            x=data["Machine_ID"],
            y=data["Average_Production_Speed"],
            name="Production Speed",
            mode="lines+markers",
            line={
                "color": COLORS["production"],
                "width": 2,
            },
            yaxis="y",
        )
    )

    # Defect rate
    fig.add_trace(
        go.Scatter(
            x=data["Machine_ID"],
            y=data["Average_Defect_Rate"],
            name="Defect Rate",
            mode="lines+markers",
            line={
                "color": COLORS["defect"],
                "width": 2,
            },
            yaxis="y2",
        )
    )

    # Error rate
    fig.add_trace(
        go.Scatter(
            x=data["Machine_ID"],
            y=data["Average_Error_Rate"],
            name="Error Rate",
            mode="lines+markers",
            line={
                "color": COLORS["error"],
                "width": 2,
                "dash": "dash",
            },
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Machine-Level Production and Quality Comparison",
        template=CHART_TEMPLATE,
        height=550,
        hovermode="x unified",
        xaxis={
            "title": "Machine ID",
        },
        yaxis={
            "title": "Production Speed (units/hour)",
            "side": "left",
        },
        yaxis2={
            "title": "Defect Rate / Error Rate (%)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={
            "title": "Metric",
            "orientation": "h",
            "y": -0.2,
        },
        margin={"l": 70, "r": 70, "t": 80, "b": 90},
    )

    return fig
