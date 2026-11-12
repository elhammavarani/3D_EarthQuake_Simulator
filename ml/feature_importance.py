import json
import joblib
import pandas as pd
from pathlib import Path


ARTIFACTS_DIR = Path("ml/artifacts")


def get_feature_importance():
    model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")

    with open(ARTIFACTS_DIR / "feature_columns.json", "r") as file:
        feature_columns = json.load(file)

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    return importance_df


if __name__ == "main":
    get_feature_importance()