
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/Thales_Group_Manufacturing.csv"

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
)

df = (
    df[["Machine_ID", "Combined_Timestamp"]]
    .drop_duplicates()
    .sort_values(["Machine_ID", "Combined_Timestamp"])
)

df["Interval_Minutes"] = (
    df.groupby("Machine_ID")["Combined_Timestamp"]
    .diff()
    .dt.total_seconds() / 60
)

intervals = df["Interval_Minutes"].dropna()

print("\nDIRECT INTERVAL CHECK")
print("Timestamp dtype:", df["Combined_Timestamp"].dtype)
print("Interval count:", len(intervals))
print("90th percentile:", intervals.quantile(0.90))
print("95th percentile:", intervals.quantile(0.95))

print("\nINTERVAL COUNTS")
for lower, upper in [(15, 45), (45, 75), (105, 135)]:
    count = ((intervals >= lower) & (intervals <= upper)).sum()
    print(f"{lower}-{upper} minutes: {count}")

print("\nEXACT INTERVALS BETWEEN 105 AND 135 MINUTES")
print(
    intervals[
        intervals.between(105, 135)
    ].value_counts().sort_index().to_string()
)

print("\nSAMPLE MATCHES FOR 120-MINUTE HORIZON")
matches = []

for machine_id, group in df.groupby("Machine_ID"):
    times = group["Combined_Timestamp"].tolist()

    for i in range(len(times) - 1):
        actual_gap = (
            times[i + 1] - times[i]
        ).total_seconds() / 60

        if 105 <= actual_gap <= 135:
            matches.append({
                "Machine_ID": machine_id,
                "source": times[i],
                "future": times[i + 1],
                "actual_gap_minutes": actual_gap
            })

print("Adjacent qualifying intervals:", len(matches))

if matches:
    print(pd.DataFrame(matches).head(20).to_string(index=False))
