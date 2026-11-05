from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
OUT = ROOT / "reports"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EXCLUDE_FROM_NUMERIC_FEATURES = {"Machine_ID"}

def main():
    df = pd.read_csv(DATA)
    df["Combined_Timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " +
        df["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )
    numeric = [
        c for c in df.select_dtypes(include="number").columns
        if c not in EXCLUDE_FROM_NUMERIC_FEATURES
    ]

    df[numeric].describe().T.to_csv(OUT / "eda_numeric_summary.csv")
    df.groupby("Efficiency_Status")[numeric].agg(
        ["count", "mean", "median", "std"]
    ).to_csv(OUT / "eda_metrics_by_efficiency.csv")
    df.groupby("Operation_Mode")[numeric].agg(
        ["count", "mean", "median", "std"]
    ).to_csv(OUT / "eda_metrics_by_operation_mode.csv")
    df.groupby("Machine_ID")["Production_Speed_units_per_hr"].agg(
        ["count", "mean", "median", "std"]
    ).sort_values("mean").to_csv(OUT / "eda_machine_production_summary.csv")

    for col in ["Efficiency_Status", "Operation_Mode"]:
        counts = df[col].value_counts(dropna=False)
        counts.rename_axis(col).reset_index(name="records").to_csv(
            OUT / f"eda_{col.lower()}_counts.csv", index=False
        )
        ax = counts.plot(kind="bar", figsize=(8, 4))
        ax.set(title=f"Records by {col}", xlabel=col, ylabel="Records")
        plt.tight_layout()
        plt.savefig(FIG / f"{col.lower()}_distribution.png", dpi=150)
        plt.close()

    df[numeric].corr(method="pearson").to_csv(OUT / "eda_numeric_correlations.csv")
    corr = df[numeric].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12, 9))
    image = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    ax.set_title("Pearson correlation matrix (Machine_ID excluded)")
    fig.colorbar(image, ax=ax, label="Correlation")
    plt.tight_layout()
    plt.savefig(FIG / "numeric_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    valid = df.dropna(subset=["Combined_Timestamp"]).copy()
    valid["day"] = valid["Combined_Timestamp"].dt.floor("D")
    trend_columns = [
        "Production_Speed_units_per_hr", "Temperature_C",
        "Quality_Control_Defect_Rate_%", "Error_Rate_%",
        "Network_Latency_ms", "Packet_Loss_%"
    ]
    daily = valid.groupby("day")[trend_columns].mean()
    daily.to_csv(OUT / "eda_daily_trends.csv")
    for col in trend_columns:
        ax = daily[col].plot(figsize=(10, 4))
        ax.set(title=f"Daily mean: {col}", xlabel="Date", ylabel=col)
        plt.tight_layout()
        plt.savefig(FIG / f"daily_{col.lower()}.png", dpi=150)
        plt.close()

    status_counts = df["Efficiency_Status"].value_counts()
    summary = [
        "EDA SUMMARY",
        f"Records: {len(df):,}",
        f"Machines: {df['Machine_ID'].nunique()}",
        f"Operation modes: {df['Operation_Mode'].nunique()}",
        f"Combined timestamp parsing failures: {int(df['Combined_Timestamp'].isna().sum())}",
        "",
        "Efficiency distribution:"
    ]
    for status, count in status_counts.items():
        summary.append(f"{status}: {count:,} ({count / len(df) * 100:.2f}%)")
    summary += [
        "",
        "Interpretation cautions:",
        "Descriptive associations do not establish causation.",
        "Machine_ID is an identifier and is excluded from numeric correlations.",
        "Confirm target-label construction before supervised classification.",
        "Investigate duplicate machine/timestamp keys before time-series modeling.",
        "The CSV alone does not establish a causal 6G performance effect."
    ]
    (OUT / "eda_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"Tables: {OUT}\nCharts: {FIG}")

if __name__ == "__main__":
    main()
