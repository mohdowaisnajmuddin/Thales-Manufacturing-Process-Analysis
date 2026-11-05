
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

files = {
    30: DATA_DIR / "forecast_dataset_30min.csv",
    60: DATA_DIR / "forecast_dataset_60min.csv",
    120: DATA_DIR / "forecast_dataset_120min.csv",
}

current_col = "Current_Production_Speed_units_per_hr"
future_col = "Future_Production_Speed"

results = []

for horizon, path in files.items():
    df = pd.read_csv(path)

    df[current_col] = pd.to_numeric(
        df[current_col], errors="coerce"
    )
    df[future_col] = pd.to_numeric(
        df[future_col], errors="coerce"
    )

    df = df.dropna(
        subset=[current_col, future_col, "Machine_ID"]
    )

    overall_pearson = df[current_col].corr(
        df[future_col], method="pearson"
    )
    overall_spearman = df[current_col].corr(
        df[future_col], method="spearman"
    )

    machine_results = []

    for machine_id, group in df.groupby("Machine_ID"):
        if len(group) < 3:
            continue

        correlation = group[current_col].corr(
            group[future_col]
        )

        machine_results.append({
            "horizon_minutes": horizon,
            "Machine_ID": machine_id,
            "records": len(group),
            "pearson_correlation": correlation,
            "mean_absolute_change": (
                group[future_col] - group[current_col]
            ).abs().mean(),
        })

    results.append({
        "horizon_minutes": horizon,
        "records": len(df),
        "overall_pearson": overall_pearson,
        "overall_spearman": overall_spearman,
        "machines": df["Machine_ID"].nunique(),
        "machine_mean_correlation": pd.DataFrame(
            machine_results
        )["pearson_correlation"].mean(),
        "machines_positive_correlation": sum(
            item["pearson_correlation"] > 0
            for item in machine_results
        ),
        "machines_negative_correlation": sum(
            item["pearson_correlation"] < 0
            for item in machine_results
        ),
    })

    print(f"\n{horizon}-MINUTE HORIZON")
    print("Records:", len(df))
    print("Overall Pearson:", round(overall_pearson, 5))
    print("Overall Spearman:", round(overall_spearman, 5))
    print(
        "Mean machine correlation:",
        round(results[-1]["machine_mean_correlation"], 5)
    )
    print(
        "Machines with positive correlation:",
        results[-1]["machines_positive_correlation"]
    )
    print(
        "Machines with negative correlation:",
        results[-1]["machines_negative_correlation"]
    )

    pd.DataFrame(machine_results).to_csv(
        REPORT_DIR / f"forecast_machine_pair_relationship_{horizon}min.csv",
        index=False
    )

summary = pd.DataFrame(results)
summary.to_csv(
    REPORT_DIR / "forecast_pair_relationship_summary.csv",
    index=False
)

print("\nSaved summary:")
print(REPORT_DIR / "forecast_pair_relationship_summary.csv")
