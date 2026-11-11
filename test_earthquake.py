import numpy as np
from physics.earthquake import generate_ground_motion


def test_ground_motion():

    t, ag = generate_ground_motion()

    # Check array dimensions
    assert len(t) == len(ag)
    assert len(t) > 0

    # Check time step
    assert np.isclose(t[1] - t[0], 0.01)

    # Check initial time
    assert t[0] == 0

    # Check PGA
    assert np.isclose(np.max(np.abs(ag)), 0.3 * 9.81)

    # Check finite values
    assert np.all(np.isfinite(ag))

    # Check reproducibility
    t2, ag2 = generate_ground_motion()
    assert np.allclose(ag, ag2)

    print("All earthquake ground motion tests passed!")


if __name__ == "main":
    test_ground_motion()