import json
import pandas as pd
from pathlib import Path


EVALUATION_PATH = Path("ml/artifacts/evaluation.json")


def load_evaluation():
    with open(EVALUATION_PATH, "r") as file:
        return json.load(file)


def get_model_comparison():
    evaluation = load_evaluation()
    results = evaluation["results"]

    rows = []

    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "Mean R2": metrics["mean_R2"]
        })

    return pd.DataFrame(rows).sort_values(
        by="Mean R2",
        ascending=False
    )


def get_target_metrics():
    evaluation = load_evaluation()
    results = evaluation["results"]

    rows = []

    for model_name, targets in results.items():
        for target_name, metrics in targets.items():
            if target_name.startswith("target_"):
                rows.append({
                    "Model": model_name,
                    "Target": target_name.replace("target_max_", ""),
                    "R2": metrics["R2"],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"]
                })

    return pd.DataFrame(rows)


if __name__ == "main":
    print("\nMODEL COMPARISON")
    print(get_model_comparison().to_string(index=False))

    print("\nTARGET METRICS")
    print(get_target_metrics().to_string(index=False))