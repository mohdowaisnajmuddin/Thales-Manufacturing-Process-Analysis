
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.kpi_calculations import (
    calculate_efficiency_distribution,
    calculate_machine_summary,
    calculate_operation_mode_summary,
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


def main():
    df = pd.read_csv(
        "data/raw/Thales_Group_Manufacturing.csv"
    )

    machine_summary = calculate_machine_summary(df)
    efficiency = calculate_efficiency_distribution(df)
    operation_summary = calculate_operation_mode_summary(df)

    charts = {
        "Production speed": production_speed_by_machine(
            machine_summary
        ),
        "Defect rate": defect_rate_by_machine(
            machine_summary
        ),
        "Error rate": error_rate_by_machine(
            machine_summary
        ),
        "Efficiency distribution": efficiency_distribution_chart(
            efficiency
        ),
        "Operation mode": production_speed_by_operation_mode(
            operation_summary
        ),
        "Machine comparison": machine_comparison_chart(
            machine_summary
        ),
    }

    charts.update(
        all_sensor_distribution_charts(df)
    )

    print("\nCHART VALIDATION RESULTS")

    for name, figure in charts.items():
        assert figure is not None
        assert len(figure.data) > 0

        print(
            f"PASS: {name} | "
            f"Traces: {len(figure.data)}"
        )

    print("\nAll visualization tests passed.")


if __name__ == "__main__":
    main()
