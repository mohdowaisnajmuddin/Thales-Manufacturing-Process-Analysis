
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
OUT = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA)

    required = [
        "Efficiency_Status",
        "Production_Speed_units_per_hr",
        "Error_Rate_%"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required).copy()

    speed = "Production_Speed_units_per_hr"
    error = "Error_Rate_%"
    target = "Efficiency_Status"

    print("\nTARGET DISTRIBUTION")
    print(df[target].value_counts(normalize=True).mul(100).round(2))

    # Conditional distributions by speed bands
    df["speed_band"] = pd.cut(
        df[speed],
        bins=[0, 100, 200, 300, 400, 500],
        include_lowest=True
    )

    speed_table = pd.crosstab(
        df["speed_band"],
        df[target],
        normalize="index"
    ) * 100

    speed_table.to_csv(
        OUT / "label_investigation_by_speed.csv"
    )

    # Conditional distributions by error bands
    df["error_band"] = pd.cut(
        df[error],
        bins=[-0.01, 1, 2, 3, 5, 10, 15, 20],
        include_lowest=True
    )

    error_table = pd.crosstab(
        df["error_band"],
        df[target],
        normalize="index"
    ) * 100

    error_table.to_csv(
        OUT / "label_investigation_by_error.csv"
    )

    # Joint distribution
    joint_table = pd.crosstab(
        [df["speed_band"], df["error_band"]],
        df[target],
        normalize="index"
    ) * 100

    joint_table.to_csv(
        OUT / "label_investigation_speed_error.csv"
    )

    # Candidate threshold combinations.
    # These are diagnostic candidates, NOT assumed true rules.
    results = []

    speed_thresholds = [200, 250, 300, 350, 400, 425]
    error_thresholds = [1, 2, 3, 4, 5, 6]

    for s in speed_thresholds:
        for e in error_thresholds:

            predicted = np.select(
                [
                    (df[speed] >= s) & (df[error] <= e),
                    (df[speed] >= 200) & (df[error] <= 5)
                ],
                ["High", "Medium"],
                default="Low"
            )

            actual = df[target].to_numpy()

            accuracy = np.mean(predicted == actual)

            results.append({
                "speed_threshold": s,
                "error_threshold": e,
                "accuracy": accuracy,
                "high_precision": (
                    np.sum((predicted == "High") & (actual == "High")) /
                    max(np.sum(predicted == "High"), 1)
                ),
                "high_recall": (
                    np.sum((predicted == "High") & (actual == "High")) /
                    max(np.sum(actual == "High"), 1)
                )
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["high_recall", "accuracy"],
        ascending=False
    )

    results_df.to_csv(
        OUT / "label_candidate_thresholds.csv",
        index=False
    )

    print("\nSPEED BAND DISTRIBUTION")
    print(speed_table.round(2))

    print("\nERROR BAND DISTRIBUTION")
    print(error_table.round(2))

    print("\nTOP CANDIDATE THRESHOLDS")
    print(results_df.head(10).to_string(index=False))

    print("\nSaved investigation reports to:", OUT)


if __name__ == "__main__":
    main()
