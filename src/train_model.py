"""ML training is deliberately gated pending target provenance review."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/Thales_Group_Manufacturing.csv"

def main():
    df = pd.read_csv(DATA)
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("Efficiency_Status distribution:")
    print(df["Efficiency_Status"].value_counts(dropna=False))
    print("\nSTOP: Do not train a production-claim classifier yet.")
    print("First obtain or document the rule/model used to create Efficiency_Status.")
    print("Then audit feature leakage, time dependence, machine grouping, and class imbalance.")
    print("A baseline model can be added after the target definition is defensible.")

if __name__ == "__main__":
    main()
