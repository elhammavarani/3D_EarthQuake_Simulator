import numpy as np

from physics.building import create_building
from physics.earthquake import generate_ground_motion
from physics.solver import solve_response

from physics.engineering_metrics import (
    calculate_engineering_metrics
)


def test_engineering_metrics():

    print("1. Creating building...", flush=True)

    M, C, K = create_building()

    print("2. Generating earthquake...", flush=True)

    t, ag = generate_ground_motion()

    print("3. Solving structural response...", flush=True)

    u, v, a = solve_response(M, C, K, t, ag)

    print("4. Calculating engineering metrics...", flush=True)

    results = calculate_engineering_metrics(
        M=M,
        K=K,
        u=u,
        a=a,
        ag=ag,
        story_height=3.2
    )

    print("5. Checking results...", flush=True)

    assert results["max_floor_displacement"].shape == (10,)
    assert results["max_story_drift"].shape == (10,)
    assert results["max_drift_ratio"].shape == (10,)
    assert results["first_mode_shape"].shape == (10,)

    assert results["max_base_shear"] > 0
    assert results["first_natural_frequency_hz"] > 0

    assert np.all(results["max_floor_displacement"] >= 0)
    assert np.all(results["max_story_drift"] >= 0)
    assert np.all(results["max_drift_ratio"] >= 0)

    assert np.all(np.isfinite(results["first_mode_shape"]))
    assert np.isclose(
        np.max(np.abs(results["first_mode_shape"])),
        1.0
    )

    print("\nSUCCESS: Engineering metrics tests passed!", flush=True)


if __name__ == "main":
    test_engineering_metrics()