
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/Thales_Group_Manufacturing.csv"
OUT = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [30, 60, 120]
TOLERANCE_MINUTES = 15


def find_future_match(times_ns, source_ns, horizon, tolerance):
    """Find the closest future timestamp within the requested tolerance."""

    desired_ns = source_ns + int(horizon * 60 * 1e9)
    tolerance_ns = int(tolerance * 60 * 1e9)

    lower = desired_ns - tolerance_ns
    upper = desired_ns + tolerance_ns

    start = np.searchsorted(times_ns, lower, side="left")
    end = np.searchsorted(times_ns, upper, side="right")

    if start >= end:
        return None

    candidates = times_ns[start:end]
    differences = np.abs(candidates - desired_ns)

    return candidates[np.argmin(differences)]


def main():
    df = pd.read_csv(DATA)

    df["Combined_Timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip()
        + " "
        + df["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    df = df.dropna(
        subset=["Machine_ID", "Combined_Timestamp"]
    ).copy()

    # Use unique timestamps for coverage assessment.
    # This avoids counting simultaneous duplicate records as
    # separate future observation opportunities.
    unique_data = (
        df[["Machine_ID", "Combined_Timestamp"]]
        .drop_duplicates()
        .sort_values(["Machine_ID", "Combined_Timestamp"])
    )

    overall_results = []
    machine_results = []
    match_details = []

    for machine_id, group in unique_data.groupby("Machine_ID"):
       # Explicitly convert timestamps to nanoseconds.
        times_ns = (
    group["Combined_Timestamp"]
    .to_numpy(dtype="datetime64[ns]")
    .astype("int64")
)
        total_sources = len(times_ns)

        for horizon in HORIZONS:
            matched = 0
            matched_targets = set()

            for source_ns in times_ns:
                target_ns = find_future_match(
                    times_ns,
                    source_ns,
                    horizon,
                    TOLERANCE_MINUTES
                )

                if target_ns is not None:
                    matched += 1
                    matched_targets.add(int(target_ns))

                    match_details.append({
                        "Machine_ID": machine_id,
                        "horizon_minutes": horizon,
                        "source_timestamp": pd.to_datetime(
                            source_ns
                        ),
                        "matched_future_timestamp": pd.to_datetime(
                            target_ns
                        ),
                        "actual_minutes_ahead": (
                            target_ns - source_ns
                        ) / 60e9
                    })

            machine_results.append({
                "Machine_ID": machine_id,
                "horizon_minutes": horizon,
                "unique_source_timestamps": total_sources,
                "matched_sources": matched,
                "unmatched_sources": total_sources - matched,
                "match_rate_percent": (
                    matched / total_sources * 100
                ),
                "unique_future_targets_matched": len(
                    matched_targets
                )
            })

    machine_df = pd.DataFrame(machine_results)
    details_df = pd.DataFrame(match_details)

    for horizon in HORIZONS:
        subset = machine_df[
            machine_df["horizon_minutes"] == horizon
        ]

        total_sources = subset[
            "unique_source_timestamps"
        ].sum()

        total_matched = subset["matched_sources"].sum()

        overall_results.append({
            "horizon_minutes": horizon,
            "tolerance_minutes": TOLERANCE_MINUTES,
            "total_source_timestamps": total_sources,
            "matched_sources": total_matched,
            "unmatched_sources": total_sources - total_matched,
            "match_rate_percent": (
                total_matched / total_sources * 100
            ),
            "machines": subset["Machine_ID"].nunique(),
            "machines_with_match_rate_below_50_percent": (
                subset["match_rate_percent"] < 50
            ).sum()
        })

    overall_df = pd.DataFrame(overall_results)

    overall_df.to_csv(
        OUT / "forecast_horizon_overall.csv",
        index=False
    )

    machine_df.to_csv(
        OUT / "forecast_horizon_by_machine.csv",
        index=False
    )

    details_df.to_csv(
        OUT / "forecast_horizon_match_details.csv",
        index=False
    )

    print("\nFORECAST HORIZON FEASIBILITY")
    print(f"Matching tolerance: +/- {TOLERANCE_MINUTES} minutes")
    print("\nOVERALL RESULTS")
    print(overall_df.round(2).to_string(index=False))

    print("\nPER-MACHINE MATCH RATES")
    print(
        machine_df.pivot(
            index="Machine_ID",
            columns="horizon_minutes",
            values="match_rate_percent"
        ).round(2).to_string()
    )

    print("\nSaved reports:")
    print(OUT / "forecast_horizon_overall.csv")
    print(OUT / "forecast_horizon_by_machine.csv")
    print(OUT / "forecast_horizon_match_details.csv")


if __name__ == "__main__":
    main()
