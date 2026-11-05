
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [30, 60, 120]
TARGET = "Future_Production_Speed"

CATEGORICAL_FEATURES = [
    "Machine_ID",
    "Current_Operation_Mode"
]

ORIGINAL_NUMERIC_FEATURES = [
    "Current_Temperature_C",
    "Current_Vibration_Hz",
    "Current_Power_Consumption_kW",
    "Current_Network_Latency_ms",
    "Current_Packet_Loss_%",
    "Current_Quality_Control_Defect_Rate_%",
    "Current_Production_Speed_units_per_hr",
    "Current_Predictive_Maintenance_Score",
    "Current_Error_Rate_%"
]


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_dataset(horizon):
    path = DATA_DIR / f"forecast_dataset_{horizon}min.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        "Machine_ID",
        "Source_Timestamp",
        "Target_Timestamp",
        TARGET
    ] + CATEGORICAL_FEATURES + ORIGINAL_NUMERIC_FEATURES

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Source_Timestamp"] = pd.to_datetime(
        df["Source_Timestamp"], errors="coerce"
    )

    df["Target_Timestamp"] = pd.to_datetime(
        df["Target_Timestamp"], errors="coerce"
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="coerce"
    )

    # Identify all historical features.
    historical_features = [
        column for column in df.columns
        if column.startswith("History_")
    ]

    numeric_features = (
        ORIGINAL_NUMERIC_FEATURES + historical_features
    )

    # Convert all numeric inputs.
    for column in numeric_features:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=[
            "Machine_ID",
            "Source_Timestamp",
            "Target_Timestamp",
            TARGET
        ]
    ).copy()

    df = df[
        df["Target_Timestamp"] > df["Source_Timestamp"]
    ].copy()

    df["Machine_ID"] = df["Machine_ID"].astype(str)

    df["Current_Operation_Mode"] = (
        df["Current_Operation_Mode"]
        .fillna("Unknown")
        .astype(str)
    )

    # Do not use the target timestamp or target value as features.
    features = CATEGORICAL_FEATURES + numeric_features

    print(f"\nHistorical features: {len(historical_features)}")
    print(f"Total numeric features: {len(numeric_features)}")

    return df, features, numeric_features


# --------------------------------------------------
# CHRONOLOGICAL SPLIT
# --------------------------------------------------

def chronological_split(df):
    unique_times = np.sort(
        df["Source_Timestamp"].unique()
    )

    train_end = unique_times[
        int(len(unique_times) * 0.70)
    ]

    validation_end = unique_times[
        int(len(unique_times) * 0.85)
    ]

    train = df[
        (df["Source_Timestamp"] < train_end)
        & (df["Target_Timestamp"] < train_end)
    ].copy()

    validation = df[
        (df["Source_Timestamp"] >= train_end)
        & (df["Source_Timestamp"] < validation_end)
        & (df["Target_Timestamp"] < validation_end)
    ].copy()

    test = df[
        df["Source_Timestamp"] >= validation_end
    ].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Empty chronological partition.")

    # Formal boundary checks.
    assert (train["Target_Timestamp"] < train_end).all()
    assert (validation["Target_Timestamp"] < validation_end).all()
    assert (test["Source_Timestamp"] >= validation_end).all()

    print("\nChronological split checks: PASS")

    return train, validation, test


# --------------------------------------------------
# MODEL PIPELINES
# --------------------------------------------------

def create_preprocessor(numeric_features, scale_numeric):
    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        )
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])


def create_ridge_model(numeric_features):
    preprocessing = create_preprocessor(
        numeric_features,
        scale_numeric=True
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", Ridge(alpha=10.0))
    ])


def create_random_forest(numeric_features):
    preprocessing = create_preprocessor(
        numeric_features,
        scale_numeric=False
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model)
    ])


# --------------------------------------------------
# BASELINES
# --------------------------------------------------

def get_baseline_predictions(train, split):
    global_mean = train[TARGET].mean()

    machine_means = train.groupby(
        "Machine_ID"
    )[TARGET].mean()

    return {
        "Global_Mean": np.full(
            len(split), global_mean
        ),

        "Machine_Mean": (
            split["Machine_ID"]
            .map(machine_means)
            .fillna(global_mean)
            .to_numpy()
        ),

        "Persistence": (
            split["Current_Production_Speed_units_per_hr"]
            .to_numpy()
        )
    }


# --------------------------------------------------
# TRAIN AND EVALUATE
# --------------------------------------------------

def evaluate_horizon(horizon):
    print("\n" + "=" * 65)
    print(f"HISTORICAL FEATURE EXPERIMENT: {horizon} MINUTES")
    print("=" * 65)

    df, features, numeric_features = load_dataset(horizon)

    train, validation, test = chronological_split(df)

    print("\nPartition sizes:")
    for name, split in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test)
    ]:
        print(f"{name}: {len(split):,}")

    X_train = train[features]
    y_train = train[TARGET]

    models = {
        "Ridge_Historical": create_ridge_model(
            numeric_features
        ),

        "Random_Forest_Historical": create_random_forest(
            numeric_features
        )
    }

    results = []
    predictions_output = []
    importance_output = []

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        for split_name, split in [
            ("Train", train),
            ("Validation", validation),
            ("Test", test)
        ]:

            X = split[features]
            actual = split[TARGET].to_numpy()

            predictions = model.predict(X)

            metrics = calculate_metrics(
                actual, predictions
            )

            results.append({
                "horizon_minutes": horizon,
                "split": split_name,
                "model": model_name,
                "records": len(split),
                **metrics
            })

            if split_name == "Test":
                output = pd.DataFrame({
                    "Machine_ID": split["Machine_ID"].to_numpy(),
                    "Source_Timestamp": split["Source_Timestamp"].to_numpy(),
                    "Target_Timestamp": split["Target_Timestamp"].to_numpy(),
                    "Actual": actual,
                    "Prediction": predictions
                })

                output.insert(0, "horizon_minutes", horizon)
                output.insert(1, "model", model_name)

                predictions_output.append(output)

        # Feature importance for Random Forest only.
        if model_name == "Random_Forest_Historical":

            preprocessing = model.named_steps["preprocessing"]

            feature_names = preprocessing.get_feature_names_out()

            importances = model.named_steps[
                "model"
            ].feature_importances_

            importance_df = pd.DataFrame({
                "horizon_minutes": horizon,
                "feature": feature_names,
                "importance": importances
            }).sort_values(
                "importance",
                ascending=False
            )

            importance_output.append(importance_df)

            print("\nTop 15 Random Forest features:")
            print(
                importance_df.head(15).round(5).to_string(index=False)
            )

    # Baseline evaluation.
    for split_name, split in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test)
    ]:

        actual = split[TARGET].to_numpy()

        baseline_predictions = get_baseline_predictions(
            train, split
        )

        for baseline_name, predicted in baseline_predictions.items():

            metrics = calculate_metrics(
                actual, predicted
            )

            results.append({
                "horizon_minutes": horizon,
                "split": split_name,
                "model": baseline_name,
                "records": len(split),
                **metrics
            })

    result_df = pd.DataFrame(results)

    print("\nTEST COMPARISON")
    print(
        result_df[result_df["split"] == "Test"][
            ["model", "records", "MAE", "RMSE", "R2"]
        ].round(4).to_string(index=False)
    )

    return (
        results,
        predictions_output,
        importance_output
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    all_results = []
    all_predictions = []
    all_importances = []

    for horizon in HORIZONS:

        results, predictions, importances = evaluate_horizon(
            horizon
        )

        all_results.extend(results)
        all_predictions.extend(predictions)
        all_importances.extend(importances)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        REPORT_DIR / "historical_feature_model_metrics.csv",
        index=False
    )

    pd.concat(
        all_predictions,
        ignore_index=True
    ).to_csv(
        REPORT_DIR / "historical_feature_test_predictions.csv",
        index=False
    )

    pd.concat(
        all_importances,
        ignore_index=True
    ).to_csv(
        REPORT_DIR / "historical_feature_importance.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("FINAL TEST COMPARISON")
    print("=" * 65)

    print(
        results_df[results_df["split"] == "Test"][
            ["horizon_minutes", "model", "records", "MAE", "RMSE", "R2"]
        ].round(4).to_string(index=False)
    )

    print("\nSaved reports:")
    print(REPORT_DIR / "historical_feature_model_metrics.csv")
    print(REPORT_DIR / "historical_feature_test_predictions.csv")
    print(REPORT_DIR / "historical_feature_importance.csv")


if __name__ == "__main__":
    main()
