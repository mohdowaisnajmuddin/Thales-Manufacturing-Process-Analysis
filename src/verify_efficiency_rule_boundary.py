
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

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

    # Candidate A: inclusive 5% boundary
    conditions_a = [
        (df[speed] >= 400) & (df[error] <= 2),
        (df[speed] >= 200) & (df[error] <= 5)
    ]

    df["Rule_A"] = np.select(
        conditions_a,
        ["High", "Medium"],
        default="Low"
    )

    # Candidate B: exclusive 5% boundary
    conditions_b = [
        (df[speed] >= 400) & (df[error] <= 2),
        (df[speed] >= 200) & (df[error] < 5)
    ]

    df["Rule_B"] = np.select(
        conditions_b,
        ["High", "Medium"],
        default="Low"
    )

    for rule in ["Rule_A", "Rule_B"]:
        correct = (df[target] == df[rule]).sum()
        accuracy = correct / len(df)

        print(f"\n{rule}")
        print("Correct:", correct)
        print("Incorrect:", len(df) - correct)
        print(f"Accuracy: {accuracy:.6%}")

        print(classification_report(
            df[target],
            df[rule],
            labels=["High", "Medium", "Low"],
            zero_division=0
        ))

        print(confusion_matrix(
            df[target],
            df[rule],
            labels=["High", "Medium", "Low"]
        ))

    mismatches_b = df[df[target] != df["Rule_B"]].copy()

    mismatches_b.to_csv(
        OUT / "efficiency_rule_boundary_mismatches.csv",
        index=False
    )

    print("\nRule B mismatches:")
    print(len(mismatches_b))
    print("\nSaved boundary mismatch report.")


if __name__ == "__main__":
    main()
