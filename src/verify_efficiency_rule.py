
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
OUT = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA)

    speed_col = "Production_Speed_units_per_hr"
    error_col = "Error_Rate_%"
    target_col = "Efficiency_Status"

    required = [speed_col, error_col, target_col]
    df = df.dropna(subset=required).copy()

    # Candidate rule discovered in the threshold investigation.
    conditions = [
        (df[speed_col] >= 400) & (df[error_col] <= 2),
        (df[speed_col] >= 200) & (df[error_col] <= 5)
    ]

    df["Predicted_Efficiency_Status"] = np.select(
        conditions,
        ["High", "Medium"],
        default="Low"
    )

    # Compare predicted labels against the supplied labels.
    df["Label_Match"] = (
        df[target_col] == df["Predicted_Efficiency_Status"]
    )

    mismatches = df.loc[~df["Label_Match"]].copy()

    accuracy = df["Label_Match"].mean()

    print("\nRULE VERIFICATION")
    print("Records evaluated:", len(df))
    print("Correct predictions:", int(df["Label_Match"].sum()))
    print("Incorrect predictions:", len(mismatches))
    print(f"Accuracy: {accuracy:.6%}")

    print("\nCONFUSION MATRIX")
    matrix = pd.crosstab(
        df[target_col],
        df["Predicted_Efficiency_Status"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False
    )

    print(matrix)
    matrix.to_csv(OUT / "efficiency_rule_confusion_matrix.csv")

    print("\nMISMATCHED RECORDS")
    if mismatches.empty:
        print("No mismatches found.")
    else:
        display_columns = [
            c for c in [
                "Date", "Timestamp", "Machine_ID",
                "Operation_Mode", speed_col, error_col,
                target_col, "Predicted_Efficiency_Status"
            ]
            if c in mismatches.columns
        ]

        print(mismatches[display_columns].to_string(index=False))

    mismatches.to_csv(
        OUT / "efficiency_rule_mismatches.csv",
        index=False
    )

    # Examine records near the candidate thresholds.
    near_boundary = df[
        (df[speed_col].between(395, 405)) |
        (df[error_col].between(1.8, 2.2)) |
        (df[error_col].between(4.8, 5.2))
    ].copy()

    near_boundary.to_csv(
        OUT / "efficiency_rule_boundary_records.csv",
        index=False
    )

    print("\nSaved reports:")
    print(OUT / "efficiency_rule_confusion_matrix.csv")
    print(OUT / "efficiency_rule_mismatches.csv")
    print(OUT / "efficiency_rule_boundary_records.csv")


if __name__ == "__main__":
    main()
