"""Train and evaluate machine-learning models for earthquake response prediction."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ARTIFACT_DIR = Path("ml/artifacts")
DATASET_PATH = ARTIFACT_DIR / "training_dataset.csv"

TARGET_COLUMNS = [
    "target_max_displacement",
    "target_max_drift_ratio",
    "target_max_base_shear",
]

RANDOM_STATE = 42


def load_dataset(dataset_path=DATASET_PATH):
    """Load and validate the generated training dataset."""

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run ml.dataset first."
        )

    df = pd.read_csv(dataset_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    missing_targets = [
        col for col in TARGET_COLUMNS if col not in df.columns
    ]

    if missing_targets:
        raise ValueError(
            f"Missing target columns: {missing_targets}"
        )

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if not np.all(np.isfinite(df.to_numpy(dtype=float))):
        raise ValueError("Dataset contains non-finite values.")

    return df


def evaluate_model(model, X_test, y_test):
    """Calculate evaluation metrics for each prediction target."""

    predictions = model.predict(X_test)

    results = {}

    for index, target in enumerate(TARGET_COLUMNS):
        actual = y_test.iloc[:, index]
        predicted = predictions[:, index]

        results[target] = {
            "MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(
                np.sqrt(mean_squared_error(actual, predicted))
            ),
            "R2": float(r2_score(actual, predicted)),
        }

    results["mean_R2"] = float(
        np.mean([results[target]["R2"] for target in TARGET_COLUMNS])
    )

    return results


def train_models(dataset_path=DATASET_PATH):
    """Train, compare, and save the best regression model."""

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)

    feature_columns = [
        col for col in df.columns if col not in TARGET_COLUMNS
    ]

    X = df[feature_columns]
    y = df[TARGET_COLUMNS]

    if len(df) < 10:
        raise ValueError(
            "At least 10 scenarios are required for training."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    print(f"Total scenarios: {len(df)}")
    print(f"Training scenarios: {len(X_train)}")
    print(f"Testing scenarios: {len(X_test)}")
    print(f"Number of input features: {len(feature_columns)}")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # XGBoost is optional. It is included when installed.
    try:
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor

        models["XGBoost"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        )

    except ImportError:
        print("XGBoost is not installed. Skipping XGBoost.")
        evaluation_results = {}
    trained_models = {}

    evaluation_results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        evaluation_results[name] = metrics
        trained_models[name] = model

        print(f"{name} results:")

        for target in TARGET_COLUMNS:
            target_metrics = metrics[target]

            print(
                f"  {target}: "
                f"MAE={target_metrics['MAE']:.6f}, "
                f"RMSE={target_metrics['RMSE']:.6f}, "
                f"R2={target_metrics['R2']:.4f}"
            )

        print(f"  Mean R2: {metrics['mean_R2']:.4f}")

    # Select the model with the highest average R2.
    best_model_name = max(
        evaluation_results,
        key=lambda name: evaluation_results[name]["mean_R2"],
    )

    best_model = trained_models[best_model_name]

    # Save the selected model and its metadata.
    model_path = ARTIFACT_DIR / "best_model.joblib"
    evaluation_path = ARTIFACT_DIR / "evaluation.json"
    features_path = ARTIFACT_DIR / "feature_columns.json"

    joblib.dump(best_model, model_path)

    with open(evaluation_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model": best_model_name,
                "results": evaluation_results,
                "test_size": 0.20,
                "random_state": RANDOM_STATE,
                "number_of_samples": len(df),
            },
            file,
            indent=4,
        )

    with open(features_path, "w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=4)

    print("\n" + "=" * 50)
    print(f"Best model: {best_model_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved evaluation: {evaluation_path}")
    print(f"Saved feature list: {features_path}")
    print("=" * 50)

    return best_model, evaluation_results


if __name__ == "main":
    train_models()