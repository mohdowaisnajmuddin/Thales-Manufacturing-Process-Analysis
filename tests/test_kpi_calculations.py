import pandas as pd

from src.kpi_calculations import calculate_overall_kpis


def sample_frame():
    return pd.DataFrame({
        "Machine_ID": [1, 1],
        "Operation_Mode": ["Active", "Idle"],
        "Temperature_C": [50.0, 60.0],
        "Vibration_Hz": [1.0, 2.0],
        "Power_Consumption_kW": [3.0, 4.0],
        "Network_Latency_ms": [10.0, 20.0],
        "Packet_Loss_%": [1.0, 2.0],
        "Quality_Control_Defect_Rate_%": [2.0, 8.0],
        "Production_Speed_units_per_hr": [100.0, 300.0],
        "Predictive_Maintenance_Score": [0.5, 0.6],
        "Error_Rate_%": [4.0, 10.0],
        "Efficiency_Status": ["High", "Low"],
    })


def test_production_weighted_rate_kpis():
    result = calculate_overall_kpis(sample_frame())
    assert result["defect_density_score_weighted_pct"] == 6.5
    assert result["error_frequency_index_weighted_pct"] == 8.5
