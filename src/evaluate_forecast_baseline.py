
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [30, 60, 120]

TARGET = "Future_Production_Speed"
CURRENT_SPEED = "Current_Production_Speed_units_per_hr"


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    denominator = np.sum((actual - actual.mean()) ** 2)

    r2 = (
        1 - np.sum(errors ** 2) / denominator
        if denominator > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def evaluate_one_horizon(horizon):
    path = DATA_DIR / f"forecast_dataset_{horizon}min.csv"

    df = pd.read_csv(path)

    df["Source_Timestamp"] = pd.to_datetime(
        df["Source_Timestamp"],
        errors="coerce"
    )

    df["Target_Timestamp"] = pd.to_datetime(
        df["Target_Timestamp"],
        errors="coerce"
    )

    df[TARGET] = pd.to_numeric(
        df["Future_Production_Speed"],
        errors="coerce"
    )

    df[CURRENT_SPEED] = pd.to_numeric(
        df["Current_Production_Speed_units_per_hr"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Machine_ID",
            "Source_Timestamp",
            "Target_Timestamp",
            TARGET,
            CURRENT_SPEED
        ]
    ).copy()

    # Global chronological boundaries based on source timestamps.
    unique_times = np.sort(
        df["Source_Timestamp"].unique()
    )

    train_end = unique_times[
        int(len(unique_times) * 0.70)
    ]

    validation_end = unique_times[
        int(len(unique_times) * 0.85)
    ]

    # Keep targets inside their respective chronological periods.
    train = df[
        (df["Source_Timestamp"] < train_end)
        & (df["Target_Timestamp"] < train_end)
    ].copy()

    validation = df[
        (df["Source_Timestamp"] >= train_end)
        & (df["Source_Timestamp"] < validation_end)
        & (df["Target_Timestamp"] < validation_end)
    ].copy()

    test = df[
        (df["Source_Timestamp"] >= validation_end)
    ].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            f"An empty split was created for {horizon} minutes."
        )

    # Calculate all learned baselines from training data only.
    global_mean = train[TARGET].mean()

    machine_means = train.groupby("Machine_ID")[TARGET].mean()

    results = []

    for split_name, split in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test)
    ]:
        actual = split[TARGET].to_numpy()

        predictions = {
            "Persistence": split[CURRENT_SPEED].to_numpy(),
            "Global_Mean": np.full(len(split), global_mean),
            "Machine_Mean": (
                split["Machine_ID"]
                .map(machine_means)
                .fillna(global_mean)
                .to_numpy()
            )
        }

        for baseline_name, predicted in predictions.items():
            metrics = calculate_metrics(actual, predicted)

            results.append({
                "horizon_minutes": horizon,
                "split": split_name,
                "baseline": baseline_name,
                "records": len(split),
                **metrics
            })

    # Split diagnostics.
    split_summary = []

    for name, split in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test)
    ]:
        split_summary.append({
            "horizon_minutes": horizon,
            "split": name,
            "records": len(split),
            "machines": split["Machine_ID"].nunique(),
            "source_start": split["Source_Timestamp"].min(),
            "source_end": split["Source_Timestamp"].max(),
            "target_start": split["Target_Timestamp"].min(),
            "target_end": split["Target_Timestamp"].max(),
            "mean_actual_speed": split[TARGET].mean(),
            "std_actual_speed": split[TARGET].std(),
            "mean_current_speed": split[CURRENT_SPEED].mean()
        })

    # Count pairs removed because their target crossed a split boundary.
    train_candidates = df[
        df["Source_Timestamp"] < train_end
    ]

    validation_candidates = df[
        (df["Source_Timestamp"] >= train_end)
        & (df["Source_Timestamp"] < validation_end)
    ]

    discarded = {
        "horizon_minutes": horizon,
        "train_boundary_crossing_pairs": (
            len(train_candidates) - len(train)
        ),
        "validation_boundary_crossing_pairs": (
            len(validation_candidates) - len(validation)
        )
    }

    print(f"\n{'=' * 60}")
    print(f"{horizon}-MINUTE HORIZON")
    print(f"{'=' * 60}")

    print("\nSplit sizes:")
    print(
        pd.DataFrame(split_summary)[
            ["split", "records", "machines",
             "source_start", "source_end"]
        ].to_string(index=False)
    )

    print("\nBaseline metrics:")
    print(
        pd.DataFrame([
            r for r in results
        ])[["split", "baseline", "records",
            "MAE", "RMSE", "R2"]]
        .round(4)
        .to_string(index=False)
    )

    print("\nBoundary-crossing pairs excluded:")
    print(discarded)

    return results, split_summary, discarded


def main():
    all_results = []
    all_splits = []
    all_discarded = []

    for horizon in HORIZONS:
        results, splits, discarded = evaluate_one_horizon(
            horizon
        )

        all_results.extend(results)
        all_splits.extend(splits)
        all_discarded.append(discarded)

    metrics_df = pd.DataFrame(all_results)
    splits_df = pd.DataFrame(all_splits)
    discarded_df = pd.DataFrame(all_discarded)

    metrics_df.to_csv(
        REPORT_DIR / "forecast_baseline_metrics.csv",
        index=False
    )

    splits_df.to_csv(
        REPORT_DIR / "forecast_chronological_splits.csv",
        index=False
    )

    discarded_df.to_csv(
        REPORT_DIR / "forecast_boundary_exclusions.csv",
        index=False
    )

    print("\nFINAL TEST COMPARISON")
    print(
        metrics_df[
            metrics_df["split"] == "Test"
        ][
            ["horizon_minutes", "baseline", "records",
             "MAE", "RMSE", "R2"]
        ].round(4).to_string(index=False)
    )

    print("\nReports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
