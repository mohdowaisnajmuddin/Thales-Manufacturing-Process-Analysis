"""Feature engineering scaffold. No undocumented thresholds are imposed."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/Thales_Group_Manufacturing.csv"
OUT = ROOT / "data/processed"

def main():
    df = pd.read_csv(DATA)
    df["Combined_Timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " +
        df["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )
    df["hour"] = df["Combined_Timestamp"].dt.hour
    df["day_of_week"] = df["Combined_Timestamp"].dt.dayofweek
    df["date"] = df["Combined_Timestamp"].dt.date.astype("string")
    df.to_csv(OUT / "manufacturing_features.csv", index=False)
    print(f"Saved {OUT / 'manufacturing_features.csv'}")
    print("These are calendar/time features only; no machine-health threshold was assumed.")

if __name__ == "__main__":
    main()
