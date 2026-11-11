import numpy as np

from physics.building import create_building
from physics.earthquake import generate_ground_motion
from physics.solver import solve_response


def test_solver():

    print("1. Creating building model...", flush=True)
    M, C, K = create_building()

    print("2. Generating earthquake...", flush=True)
    t, ag = generate_ground_motion()

    print("3. Solving dynamic response...", flush=True)
    u, v, a = solve_response(M, C, K, t, ag)

    print("4. Checking response dimensions...", flush=True)

    expected_shape = (len(t), 10)

    assert u.shape == expected_shape
    assert v.shape == expected_shape
    assert a.shape == expected_shape

    print("5. Checking numerical stability...", flush=True)

    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(a))

    print("6. Checking initial conditions...", flush=True)

    assert np.allclose(u[0], 0)
    assert np.allclose(v[0], 0)

    print("7. Checking nonzero response...", flush=True)

    assert np.max(np.abs(u)) > 0

    print("\nSUCCESS: All dynamic response tests passed!", flush=True)


if __name__ == "main":
    test_solver()