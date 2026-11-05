from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "Date", "Timestamp", "Machine_ID", "Operation_Mode", "Temperature_C",
    "Vibration_Hz", "Power_Consumption_kW", "Network_Latency_ms",
    "Packet_Loss_%", "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr", "Predictive_Maintenance_Score",
    "Error_Rate_%", "Efficiency_Status",
]

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    missing_columns = [c for c in REQUIRED if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    # Timestamp is time-of-day; combine with DD-MM-YYYY Date.
    combined = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " +
        df["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )
    key_frame = pd.DataFrame({
        "Machine_ID": df["Machine_ID"],
        "Combined_Timestamp": combined,
    })
    duplicate_mask = key_frame.duplicated(
        ["Machine_ID", "Combined_Timestamp"], keep=False
    )

    summary = [
        "DATA VALIDATION SUMMARY",
        f"File: {DATA.name}",
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns)}",
        f"Missing cells: {int(df.isna().sum().sum()):,}",
        f"Fully duplicated rows: {int(df.duplicated().sum()):,}",
        f"Unique machines: {df['Machine_ID'].nunique()}",
        f"Operation modes: {df['Operation_Mode'].nunique()}",
        f"Efficiency statuses: {df['Efficiency_Status'].nunique()}",
        f"Invalid dates: {int(pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce').isna().sum())}",
        f"Invalid combined timestamps: {int(combined.isna().sum())}",
        f"Duplicate machine + timestamp keys (excluding first): {int(key_frame.duplicated(['Machine_ID', 'Combined_Timestamp']).sum())}",
        f"Rows participating in duplicate machine + timestamp keys: {int(duplicate_mask.sum())}",
        f"Combined datetime start: {combined.min()}",
        f"Combined datetime end: {combined.max()}",
        "",
        "CAUTIONS",
        "Duplicate machine/timestamp keys are flagged, not deleted.",
        "Validate units and engineering limits with an authoritative source.",
        "Efficiency_Status and Predictive_Maintenance_Score provenance must be confirmed.",
    ]
    (OUT / "data_quality_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    duplicate_rows = df.loc[duplicate_mask].copy()
    duplicate_rows.insert(2, "Combined_Timestamp", combined.loc[duplicate_mask].values)
    duplicate_rows.to_csv(OUT / "duplicate_machine_timestamps.csv", index=False)
    print("\n".join(summary))

if __name__ == "__main__":
    main()
