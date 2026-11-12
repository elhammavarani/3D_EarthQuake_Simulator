"""Predict earthquake response using the trained machine-learning model."""

import json
from pathlib import Path

import joblib
import pandas as pd

from ml.dataset import _extract_ground_motion_features
from physics.earthquake import generate_ground_motion


ARTIFACT_DIR = Path("ml/artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
FEATURES_PATH = ARTIFACT_DIR / "feature_columns.json"


def predict_response(
    n_floors=10,
    mass=250_000,
    stiffness=150_000_000,
    damping_ratio=0.05,
    story_height=3.2,
    pga_g=0.3,
    duration=15.0,
    dt=0.02,
    seed=42,
):
    """Predict maximum displacement, drift ratio, and base shear."""

    # Validate inputs
    if n_floors < 1:
        raise ValueError("n_floors must be positive.")

    if mass <= 0 or stiffness <= 0:
        raise ValueError("Mass and stiffness must be positive.")

    if not 0 < damping_ratio < 1:
        raise ValueError("damping_ratio must be between 0 and 1.")

    if story_height <= 0 or pga_g <= 0:
        raise ValueError("story_height and pga_g must be positive.")

    if duration <= 0 or dt <= 0 or dt >= duration:
        raise ValueError("Invalid duration or time step.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature columns not found: {FEATURES_PATH}")

    # Load trained model and feature names
    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    # Generate ground motion
    t, ag = generate_ground_motion(
        duration=duration,
        dt=dt,
        pga=pga_g * 9.81,
        seed=seed,
    )

    # Extract ground motion features
    ground_features = _extract_ground_motion_features(t, ag)

    # Prepare model input
    input_data = {
        "n_floors": int(n_floors),
        "mass": float(mass),
        "stiffness": float(stiffness),
        "damping_ratio": float(damping_ratio),
        "story_height": float(story_height),
        "duration": float(duration),
        "time_step": float(dt),
        **ground_features,
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_columns]

    # Make prediction
    predictions = model.predict(input_df)[0]

    results = {
        "max_displacement": float(predictions[0]),
        "max_drift_ratio": float(predictions[1]),
        "max_base_shear": float(predictions[2]),
    }

    print("\nEarthquake Response Prediction")
    print("-" * 40)
    print(f"Number of floors: {n_floors}")
    print(f"PGA: {pga_g:.3f} g")
    print(f"Max displacement: {results['max_displacement']:.4f} m")
    print(f"Max drift ratio: {results['max_drift_ratio']:.4%}")
    print(f"Max base shear: {results['max_base_shear']:,.2f} N")
    print("-" * 40)

    return results


if __name__ == "main":
    predict_response()