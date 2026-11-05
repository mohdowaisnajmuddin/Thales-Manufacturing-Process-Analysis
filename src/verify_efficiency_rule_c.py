
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/Thales_Group_Manufacturing.csv"
OUT = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA)

    speed = "Production_Speed_units_per_hr"
    error = "Error_Rate_%"
    target = "Efficiency_Status"

    df = df.dropna(subset=[speed, error, target]).copy()

    # Rule C: test the interaction at exactly 5% error.
    high_condition = (
        (df[speed] >= 400) &
        (df[error] <= 2)
    )

    medium_condition = (
        (df[speed] >= 200) &
        (df[error] <= 5) &
        (
            (df[error] < 5) |
            (df[speed] < 400)
        )
    )

    df["Rule_C"] = np.select(
        [high_condition, medium_condition],
        ["High", "Medium"],
        default="Low"
    )

    # Compare against supplied labels.
    df["Label_Match"] = df[target] == df["Rule_C"]

    correct = int(df["Label_Match"].sum())
    incorrect = len(df) - correct

    print("\nRULE C VERIFICATION")
    print("Records:", len(df))
    print("Correct:", correct)
    print("Incorrect:", incorrect)
    print(f"Accuracy: {accuracy_score(df[target], df['Rule_C']):.6%}")

    labels = ["High", "Medium", "Low"]

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        df[target],
        df["Rule_C"],
        labels=labels,
        zero_division=0
    ))

    print("\nCONFUSION MATRIX")
    matrix = confusion_matrix(
        df[target],
        df["Rule_C"],
        labels=labels
    )

    print(pd.DataFrame(
        matrix,
        index=[f"Actual_{x}" for x in labels],
        columns=[f"Predicted_{x}" for x in labels]
    ))

    # Save all remaining exceptions.
    mismatches = df.loc[~df["Label_Match"]].copy()

    mismatches.to_csv(
        OUT / "efficiency_rule_c_mismatches.csv",
        index=False
    )

    # Save records exactly at the 5% boundary.
    boundary = df[df[error] == 5].copy()

    boundary.to_csv(
        OUT / "efficiency_rule_c_five_percent_boundary.csv",
        index=False
    )

    print("\nRecords at exactly 5% error:", len(boundary))
    print("Remaining mismatches:", len(mismatches))

    print("\nSaved:")
    print(OUT / "efficiency_rule_c_mismatches.csv")
    print(OUT / "efficiency_rule_c_five_percent_boundary.csv")


if __name__ == "__main__":
    main()
