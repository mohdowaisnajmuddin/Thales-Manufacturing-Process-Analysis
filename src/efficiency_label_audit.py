
"""Audit the construction and consistency of Efficiency_Status.

This is an exploratory target-label audit, not proof of the original
label-generation formula.

The decision tree is deliberately shallow. Its purpose is to identify
simple relationships that may indicate target leakage.

Generalization is tested by holding out entire machines, rather than
randomly splitting individual rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Thales_Group_Manufacturing.csv"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "Efficiency_Status"
GROUP = "Machine_ID"

NUMERIC_FEATURES = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
]

REQUIRED_COLUMNS = [
    "Date",
    "Timestamp",
    GROUP,
    "Operation_Mode",
    TARGET,
] + NUMERIC_FEATURES


def load_data():
    """Load and validate the columns needed for the audit."""
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[GROUP] = df[GROUP].astype(str).str.strip()
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df["Operation_Mode"] = df["Operation_Mode"].astype(str).str.strip()

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Combined_Timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip()
        + " "
        + df["Timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.dropna(subset=[GROUP, TARGET, "Combined_Timestamp"])

    return df


def audit_label_distribution(df):
    """Calculate overall and machine-level label distributions."""
    counts = (
        df[TARGET]
        .value_counts(dropna=False)
        .rename_axis(TARGET)
        .reset_index(name="records")
    )
    counts["percentage"] = counts["records"] / len(df) * 100
    counts.to_csv(OUT / "label_audit_distribution.csv", index=False)

    machine_counts = (
        df.groupby([GROUP, TARGET])
        .size()
        .unstack(fill_value=0)
    )

    machine_percentages = machine_counts.div(
        machine_counts.sum(axis=1), axis=0
    ) * 100

    machine_counts.to_csv(OUT / "label_audit_machine_counts.csv")
    machine_percentages.to_csv(
        OUT / "label_audit_machine_percentages.csv"
    )

    machine_summary = pd.DataFrame({
        "records": machine_counts.sum(axis=1),
        "number_of_labels_present": (machine_counts > 0).sum(axis=1),
        "dominant_label": machine_counts.idxmax(axis=1),
        "dominant_label_percentage": machine_percentages.max(axis=1),
    })

    machine_summary.to_csv(
        OUT / "label_audit_machine_summary.csv"
    )

    return counts, machine_summary


def audit_metric_distributions(df):
    """Compare candidate numerical features across target labels."""
    summary = (
        df.groupby(TARGET)[NUMERIC_FEATURES]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )

    summary.to_csv(
        OUT / "label_audit_metrics_by_efficiency.csv"
    )

    quantiles = (
        df.groupby(TARGET)[NUMERIC_FEATURES]
        .quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        .unstack()
    )

    quantiles.to_csv(
        OUT / "label_audit_metric_quantiles.csv"
    )

    return summary


def audit_score_relationship(df):
    """Inspect whether the maintenance score varies by efficiency label."""
    score_summary = (
        df.groupby(TARGET)["Predictive_Maintenance_Score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )

    score_summary.to_csv(
        OUT / "label_audit_maintenance_score.csv"
    )

    return score_summary


def audit_label_changes_over_time(df):
    """Count consecutive label changes within each machine."""
    ordered = df.sort_values(
        [GROUP, "Combined_Timestamp"],
        kind="stable"
    ).copy()

    ordered["previous_label"] = (
        ordered.groupby(GROUP)[TARGET].shift(1)
    )

    ordered["label_changed"] = (
        ordered["previous_label"].notna()
        & (ordered[TARGET] != ordered["previous_label"])
    )

    machine_changes = (
        ordered.groupby(GROUP)
        .agg(
            records=(TARGET, "size"),
            label_changes=("label_changed", "sum"),
            distinct_labels=(TARGET, "nunique"),
        )
    )

    machine_changes["change_rate_between_adjacent_records"] = (
        machine_changes["label_changes"]
        / (machine_changes["records"] - 1).replace(0, np.nan)
    )

    machine_changes.to_csv(
        OUT / "label_audit_temporal_changes_by_machine.csv"
    )

    transition_rows = ordered.dropna(
        subset=["previous_label"]
    )

    transitions = pd.crosstab(
        transition_rows["previous_label"],
        transition_rows[TARGET],
        rownames=["previous_label"],
        colnames=["current_label"],
    )

    transitions.to_csv(
        OUT / "label_audit_transition_matrix.csv"
    )

    return machine_changes, transitions


def audit_decision_tree(df):
    """Test whether numerical fields can reconstruct labels on unseen machines."""
    usable = df.dropna(
        subset=NUMERIC_FEATURES + [TARGET, GROUP]
    ).copy()

    # Operation mode is included as a categorical candidate feature.
    mode_features = pd.get_dummies(
        usable["Operation_Mode"],
        prefix="Operation_Mode",
        dtype=int,
    )

    X = pd.concat(
        [usable[NUMERIC_FEATURES], mode_features],
        axis=1,
    )

    y = usable[TARGET]
    groups = usable[GROUP]

    if groups.nunique() < 5:
        raise ValueError(
            "Not enough unique machines for a machine-held-out audit."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_machines = sorted(groups.iloc[train_idx].unique())
    test_machines = sorted(groups.iloc[test_idx].unique())

    # Baseline: always predict the most common training label.
    majority_label = y_train.value_counts().idxmax()
    baseline_predictions = np.repeat(majority_label, len(y_test))

    # A shallow tree makes simple label rules easier to inspect.
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=0.02,
        random_state=42,
    )

    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )
    tree_accuracy = accuracy_score(y_test, predictions)
    tree_balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )

    metrics = pd.DataFrame([{
        "training_rows": len(train_idx),
        "testing_rows": len(test_idx),
        "training_machines": len(train_machines),
        "testing_machines": len(test_machines),
        "majority_baseline_accuracy": baseline_accuracy,
        "tree_accuracy": tree_accuracy,
        "tree_balanced_accuracy": tree_balanced_accuracy,
        "tree_depth": tree.get_depth(),
        "tree_leaf_count": tree.get_n_leaves(),
        "classes": ", ".join(map(str, tree.classes_)),
    }])

    metrics.to_csv(
        OUT / "label_audit_tree_performance.csv",
        index=False,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=tree.classes_,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUT / "label_audit_tree_classification_report.csv"
    )

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": tree.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        OUT / "label_audit_tree_feature_importance.csv",
        index=False,
    )

    rules = export_text(
        tree,
        feature_names=list(X.columns),
        decimals=4,
    )

    (OUT / "label_audit_tree_rules.txt").write_text(
        rules,
        encoding="utf-8",
    )

    split_details = [
        "MACHINE-HELD-OUT DECISION TREE AUDIT",
        "",
        f"Training machines: {train_machines}",
        f"Testing machines: {test_machines}",
        "",
        f"Training rows: {len(train_idx)}",
        f"Testing rows: {len(test_idx)}",
        f"Majority baseline accuracy: {baseline_accuracy:.4f}",
        f"Decision tree accuracy: {tree_accuracy:.4f}",
        f"Decision tree balanced accuracy: {tree_balanced_accuracy:.4f}",
        "",
        "TREE RULES:",
        rules,
        "",
        "Caution: high predictive performance does not prove the original "
        "label-generation formula. It may indicate a proxy relationship "
        "or target leakage.",
    ]

    (OUT / "label_audit_tree_summary.txt").write_text(
        "\n".join(split_details),
        encoding="utf-8",
    )

    return metrics, importance, rules


def main():
    df = load_data()

    print("EFFICIENCY STATUS LABEL AUDIT")
    print("=" * 45)
    print("Rows:", len(df))
    print("Machines:", df[GROUP].nunique())
    print("Labels:", sorted(df[TARGET].unique()))
    print("Operation modes:", sorted(df["Operation_Mode"].unique()))

    counts, machine_summary = audit_label_distribution(df)
    metric_summary = audit_metric_distributions(df)
    score_summary = audit_score_relationship(df)
    machine_changes, transitions = audit_label_changes_over_time(df)
    tree_metrics, importance, rules = audit_decision_tree(df)

    print("\nOverall label distribution:")
    print(counts.to_string(index=False))

    print("\nMachine label consistency:")
    print(machine_summary.describe().round(4).to_string())

    print("\nMachine-held-out decision tree:")
    print(tree_metrics.to_string(index=False))

    print("\nMost important tree features:")
    print(importance.head(10).to_string(index=False))

    print("\nTree rules:")
    print(rules)

    print("\nTemporal label changes:")
    print(machine_changes["label_changes"].describe().to_string())

    print("\nSaved audit outputs in:", OUT)


if __name__ == "__main__":
    main()
