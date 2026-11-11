import numpy as np
import pytest

from physics.engineering_metrics import calculate_engineering_metrics


def test_story_drift_and_drift_ratio():
    M = np.array([[2.0]])
    K = np.array([[8.0]])

    u = np.array([
        [0.10],
        [0.20],
        [-0.15]
    ])

    a = np.zeros_like(u)
    ag = np.zeros(3)

    result = calculate_engineering_metrics(
        M, K, u, a, ag, story_height=3.0
    )

    expected_drift = np.array([
        [0.10],
        [0.20],
        [-0.15]
    ])

    assert np.allclose(result["story_drift"], expected_drift)
    assert np.allclose(result["max_story_drift"], [0.20])
    assert np.allclose(result["max_drift_ratio"], [0.20 / 3.0])


def test_base_shear_calculation():
    M = np.array([[2.0]])
    K = np.array([[8.0]])

    u = np.zeros((2, 1))
    a = np.array([
        [1.0],
        [-1.0]
    ])
    ag = np.array([0.5, 0.5])

    result = calculate_engineering_metrics(
        M, K, u, a, ag
    )

    expected_absolute_acceleration = np.array([
        [1.5],
        [-0.5]
    ])

    expected_base_shear = np.array([3.0, -1.0])

    assert np.allclose(
        result["absolute_acceleration"],
        expected_absolute_acceleration
    )

    assert np.allclose(
        result["base_shear_history"],
        expected_base_shear
    )

    assert np.isclose(result["max_base_shear"], 3.0)


def test_natural_frequency_and_mode_shape():
    M = np.array([[2.0]])
    K = np.array([[8.0]])

    u = np.zeros((2, 1))
    a = np.zeros_like(u)
    ag = np.zeros(2)

    result = calculate_engineering_metrics(
        M, K, u, a, ag
    )

    # omega_n = sqrt(K/M) = 2 rad/s
    # f_n = omega_n / (2*pi)

    expected_frequency = 2.0 / (2.0 * np.pi)

    assert np.isclose(
        result["first_natural_frequency_hz"],
        expected_frequency
    )

    assert np.allclose(
        result["first_mode_shape"],
        [1.0]
    )


def test_invalid_non_positive_stiffness_is_rejected():
    M = np.array([[2.0]])
    K = np.array([[0.0]])

    u = np.zeros((2, 1))
    a = np.zeros_like(u)
    ag = np.zeros(2)

    with pytest.raises(ValueError):
        calculate_engineering_metrics(
            M, K, u, a, ag
        )