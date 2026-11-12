"""Generate machine-learning datasets using the existing physics solver."""

from pathlib import Path

import numpy as np
import pandas as pd

from physics.building import create_building
from physics.earthquake import generate_ground_motion
from physics.solver import solve_response
from physics.engineering_metrics import calculate_engineering_metrics


GRAVITY = 9.81


def _extract_ground_motion_features(t, ag):
    """Extract numerical features from a ground acceleration record."""

    dt = float(t[1] - t[0])

    pga = float(np.max(np.abs(ag)))
    rms_acceleration = float(np.sqrt(np.mean(ag**2)))

    # Approximate peak ground velocity by numerical integration.
    ground_velocity = np.zeros_like(ag)
    ground_velocity[1:] = np.cumsum(
        0.5 * (ag[:-1] + ag[1:]) * dt
    )

    pgv = float(np.max(np.abs(ground_velocity)))

    # Dominant frequency from the Fourier spectrum.
    frequencies = np.fft.rfftfreq(len(ag), d=dt)
    spectrum = np.abs(np.fft.rfft(ag))

    if len(spectrum) > 1:
        dominant_index = int(np.argmax(spectrum[1:]) + 1)
        dominant_frequency = float(frequencies[dominant_index])
    else:
        dominant_frequency = 0.0

    return {
        "pga": pga,
        "pga_g": pga / GRAVITY,
        "rms_acceleration": rms_acceleration,
        "pgv": pgv,
        "dominant_frequency": dominant_frequency,
    }


def generate_dataset(
    n_samples=200,
    seed=42,
    duration=15.0,
    dt=0.02,
    output_path=None,
):
    """
    Generate a dataset from physics-based earthquake simulations.

    Each row represents one building-earthquake scenario.

    Features:
        Building properties and ground-motion characteristics.

    Targets:
        Maximum floor displacement,
        maximum inter-story drift ratio,
        maximum base shear.
    """

    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")

    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite.")

    if not np.isfinite(dt) or dt <= 0 or dt >= duration:
        raise ValueError("dt must be positive and smaller than duration.")

    rng = np.random.default_rng(seed)
    records = []

    for sample_index in range(n_samples):

        # -------------------------------------------------
        # 1. Random building properties
        # -------------------------------------------------

        n_floors = int(rng.integers(3, 16))

        mass = float(rng.uniform(150_000, 400_000))
        stiffness = float(rng.uniform(50e6, 300e6))
        damping_ratio = float(rng.uniform(0.02, 0.08))
        story_height = float(rng.uniform(2.8, 4.0))

        # -------------------------------------------------
        # 2. Random earthquake properties
        # -------------------------------------------------

        pga_g = float(rng.uniform(0.05, 0.60))
        earthquake_seed = int(rng.integers(0, 2**31 - 1))

        # -------------------------------------------------
        # 3. Create structural model
        # -------------------------------------------------

        M, C, K = create_building(
            n_floors=n_floors,
            mass=mass,
            stiffness=stiffness,
            damping_ratio=damping_ratio,
        )

        # -------------------------------------------------
        # 4. Generate ground motion
        # -------------------------------------------------

        t, ag = generate_ground_motion(
            duration=duration,
            dt=dt,
            pga=pga_g * GRAVITY,
            seed=earthquake_seed,
        )

        # -------------------------------------------------
        # 5. Solve structural response
        # -------------------------------------------------

        u, velocity, acceleration = solve_response(
            M,
            C,
            K,
            t,
            ag,
        )

        # -------------------------------------------------
        # 6. Calculate engineering metrics
        # -------------------------------------------------

        metrics = calculate_engineering_metrics(
            M,
            K,
            u,
            acceleration,
            ag,
            story_height=story_height,
        )

        # -------------------------------------------------
        # 7. Extract input features
        # -------------------------------------------------

        ground_features = _extract_ground_motion_features(
            t,
            ag,
        )

        record = {
            "n_floors": n_floors,
            "mass": mass,
            "stiffness": stiffness,
            "damping_ratio": damping_ratio,
            "story_height": story_height,
            "duration": duration,
            "time_step": dt,
            **ground_features,

            # Target variables
            "target_max_displacement": float(
                np.max(metrics["max_floor_displacement"])
            ),
            "target_max_drift_ratio": float(
                np.max(metrics["max_drift_ratio"])
            ),
            "target_max_base_shear": float(
                metrics["max_base_shear"]
            ),
        }

        records.append(record)

        if (sample_index + 1) % 25 == 0:
            print(
                f"Generated {sample_index + 1}/{n_samples} scenarios"
            )

    # -----------------------------------------------------
    # 8. Create DataFrame
    # -----------------------------------------------------

    dataset = pd.DataFrame(records)

    if not np.all(np.isfinite(dataset.to_numpy(dtype=float))):
        raise ValueError("Generated dataset contains non-finite values.")

    # -----------------------------------------------------
    # 9. Save dataset if requested
    # -----------------------------------------------------

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset.to_csv(output_file, index=False)

        print(f"Dataset saved to: {output_file}")

    print(f"\nDataset generation completed.")
    print(f"Number of scenarios: {len(dataset)}")
    print(f"Number of features and targets: {len(dataset.columns)}")

    return dataset


if __name__ == "main":
    generate_dataset(
        n_samples=200,
        seed=42,
        output_path="ml/artifacts/training_dataset.csv",
    )