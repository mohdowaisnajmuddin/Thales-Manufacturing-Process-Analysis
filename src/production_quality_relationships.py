
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIABLES = [
    "Production_Speed_units_per_hr",
    "Error_Rate_%",
    "Quality_Control_Defect_Rate_%"
]

GROUP_COLUMNS = ["Operation_Mode"]

def load_data():
    df = pd.read_csv(DATA_PATH)

    for col in VARIABLES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=VARIABLES + GROUP_COLUMNS)

    print("\nDATA VALIDATION")
    print("-" * 50)
    print("Rows used:", len(df))
    print("Operation modes:", df["Operation_Mode"].nunique())

    return df


def calculate_correlations(df, group_name):
    pairs = [
        ("Production_Speed_units_per_hr", "Error_Rate_%"),
        ("Production_Speed_units_per_hr", "Quality_Control_Defect_Rate_%"),
        ("Error_Rate_%", "Quality_Control_Defect_Rate_%")
    ]

    results = []

    for x_col, y_col in pairs:
        subset = df[[x_col, y_col]].dropna()

        if len(subset) < 3:
            continue

        if subset[x_col].nunique() < 2 or subset[y_col].nunique() < 2:
            continue

        pearson_r, pearson_p = pearsonr(
            subset[x_col], subset[y_col]
        )

        spearman_r, spearman_p = spearmanr(
            subset[x_col], subset[y_col]
        )

        results.append({
            "Group": group_name,
            "X_Variable": x_col,
            "Y_Variable": y_col,
            "Sample_Size": len(subset),
            "Pearson_Correlation": pearson_r,
            "Pearson_P_Value": pearson_p,
            "Spearman_Correlation": spearman_r,
            "Spearman_P_Value": spearman_p
        })

    return results


def interpret_correlation(value):
    strength = abs(value)

    if strength < 0.10:
        level = "Negligible"
    elif strength < 0.30:
        level = "Weak"
    elif strength < 0.50:
        level = "Moderate"
    else:
        level = "Strong"

    direction = "Positive" if value > 0 else "Negative"

    return f"{level} {direction}"


def main():
    df = load_data()

    all_results = []

    print("\nCALCULATING OVERALL CORRELATIONS...")
    all_results.extend(calculate_correlations(df, "Overall"))

    print("\nCALCULATING OPERATION-MODE CORRELATIONS...")
    for mode, group in df.groupby("Operation_Mode", observed=True):
        all_results.extend(calculate_correlations(group, mode))

    results = pd.DataFrame(all_results)

    if results.empty:
        raise ValueError("No valid correlation results were calculated.")

    results["Pearson_Interpretation"] = results[
        "Pearson_Correlation"
    ].apply(interpret_correlation)

    results["Spearman_Interpretation"] = results[
        "Spearman_Correlation"
    ].apply(interpret_correlation)

    output_path = OUTPUT_DIR / "production_quality_correlations.csv"
    results.to_csv(output_path, index=False)

    print("\nCORRELATION RESULTS")
    print("-" * 80)

    display_columns = [
        "Group",
        "X_Variable",
        "Y_Variable",
        "Sample_Size",
        "Pearson_Correlation",
        "Pearson_P_Value",
        "Spearman_Correlation",
        "Spearman_P_Value",
        "Pearson_Interpretation"
    ]

    print(
        results[display_columns]
        .round(5)
        .to_string(index=False)
    )

    print("\nIMPORTANT LIMITATIONS")
    print("-" * 50)
    print("Correlation does not establish causation.")
    print("Statistical significance does not imply practical importance.")
    print("P-values are not adjusted for multiple comparisons.")
    print("Rows may not be independent because observations repeat")
    print("across machines and time.")

    print("\nFILE SAVED:")
    print(output_path)


if __name__ == "__main__":
    main()
