import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from ml.dataset import generate_dataset


def run_unseen_validation(n_samples=50, seed=2026):
    artifacts = Path("ml/artifacts")
    model = joblib.load(artifacts / "best_model.joblib")
    features = json.loads((artifacts / "feature_columns.json").read_text())
    data = generate_dataset(n_samples=n_samples, seed=seed)
    targets = ["target_max_displacement", "target_max_drift_ratio", "target_max_base_shear"]
    predictions = model.predict(data[features])
    rows = []
    for i, name in enumerate(["Displacement", "Drift Ratio", "Base Shear"]):
        actual = data[targets[i]]
        predicted = predictions[:, i]
        rows.append({"Target": name, "R2": r2_score(actual, predicted), "MAE": mean_absolute_error(actual, predicted), "RMSE": mean_squared_error(actual, predicted) ** 0.5})
    return pd.DataFrame(rows)
