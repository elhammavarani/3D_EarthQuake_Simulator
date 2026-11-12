"""Compare machine-learning predictions with physics-based results."""

import numpy as np

from ml.predict import predict_response
from physics.building import create_building
from physics.earthquake import generate_ground_motion
from physics.solver import solve_response
from physics.engineering_metrics import calculate_engineering_metrics


def compare_models(
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
    """Compare ML predictions against physics solver results."""

    # 1. Machine-learning prediction
    ml_results = predict_response(
        n_floors=n_floors,
        mass=mass,
        stiffness=stiffness,
        damping_ratio=damping_ratio,
        story_height=story_height,
        pga_g=pga_g,
        duration=duration,
        dt=dt,
        seed=seed,
    )

    # 2. Build structural matrices
    M, C, K = create_building(
        n_floors=n_floors,
        mass=mass,
        stiffness=stiffness,
        damping_ratio=damping_ratio,
    )

    # 3. Generate the same earthquake
    t, ag = generate_ground_motion(
        duration=duration,
        dt=dt,
        pga=pga_g * 9.81,
        seed=seed,
    )

    # 4. Physics-based analysis
    u, velocity, acceleration = solve_response(M, C, K, t, ag)

    physics_metrics = calculate_engineering_metrics(
        M, K, u, acceleration, ag,
        story_height=story_height,
    )

    physics_results = {
        "max_displacement": float(
            np.max(physics_metrics["max_floor_displacement"])
        ),
        "max_drift_ratio": float(
            np.max(physics_metrics["max_drift_ratio"])
        ),
        "max_base_shear": float(
            physics_metrics["max_base_shear"]
        ),
    }

    # 5. Compare results
    print("\nML vs Physics Solver Comparison")
    print("=" * 65)

    for key in ml_results:
        ml_value = ml_results[key]
        physics_value = physics_results[key]

        absolute_error = abs(ml_value - physics_value)

        percentage_error = (
            absolute_error / abs(physics_value) * 100
            if physics_value != 0 else 0.0
        )

        print(f"\n{key}")
        print(f"ML prediction:   {ml_value:.6f}")
        print(f"Physics result:  {physics_value:.6f}")
        print(f"Absolute error:  {absolute_error:.6f}")
        print(f"Percentage error: {percentage_error:.2f}%")

    return {
        "ml": ml_results,
        "physics": physics_results,
    }


if __name__ == "main":
    compare_models()