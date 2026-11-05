
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

TARGET = "Future_Production_Speed"
CURRENT = "Current_Production_Speed_units_per_hr"


def inspect_csv(path):
    try:
        df = pd.read_csv(path, low_memory=False)

        if TARGET in df.columns and CURRENT in df.columns:
            return df

    except Exception as exc:
        print(f"Could not inspect {path.name}: {exc}")

    return None


def describe(series, label):
    values = pd.to_numeric(series, errors="coerce").dropna()

    print(f"\n{label}")
    print("-" * 55)
    print("Valid records:", len(values))
    print("Mean:", round(values.mean(), 4))
    print("Std:", round(values.std(), 4))
    print("Median:", round(values.median(), 4))
    print("Minimum:", round(values.min(), 4))
    print("Maximum:", round(values.max(), 4))
    print("1% quantile:", round(values.quantile(0.01), 4))
    print("99% quantile:", round(values.quantile(0.99), 4))


def main():
    print("=" * 65)
    print("FORECAST TARGET INTEGRITY AUDIT")
    print("=" * 65)

    forecast_files = []

    for path in PROCESSED.rglob("*.csv"):
        df = inspect_csv(path)

        if df is not None:
            forecast_files.append((path, df))

    if not forecast_files:
        raise FileNotFoundError(
            "No processed CSV containing both forecast columns was found."
        )

    for path, df in forecast_files:
        print(f"\nFILE: {path.name}")
        describe(df[CURRENT], "Current production speed")
        describe(df[TARGET], "Future production speed")

    raw_files = list(RAW.glob("*.csv"))

    print("\n" + "=" * 65)
    print("RAW DATASET PRODUCTION SPEED")
    print("=" * 65)

    for path in raw_files:
        try:
            columns = pd.read_csv(path, nrows=5).columns

            production_column = (
                "Production_Speed_units_per_hr"
            )

            if production_column not in columns:
                continue

            raw_df = pd.read_csv(
                path,
                usecols=[production_column]
            )

            describe(
                raw_df[production_column],
                f"Raw: {path.name}"
            )

        except Exception as exc:
            print(f"Could not inspect raw file {path.name}: {exc}")


if __name__ == "__main__":
    main()
