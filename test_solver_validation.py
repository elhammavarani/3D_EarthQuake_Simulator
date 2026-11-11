import numpy as np

from physics.solver import solve_response


def create_sdof_system(omega):
    """Create a single-degree-of-freedom system."""
    M = np.array([[1.0]])
    K = np.array([[omega**2]])
    C = np.array([[0.0]])

    return M, C, K


def test_zero_ground_motion_produces_zero_response():
    """Zero initial conditions and zero excitation must give zero response."""

    M, C, K = create_sdof_system(omega=2.0)

    t = np.linspace(0.0, 5.0, 501)
    ag = np.zeros_like(t)

    u, v, a = solve_response(M, C, K, t, ag)

    assert np.allclose(u, 0.0)
    assert np.allclose(v, 0.0)
    assert np.allclose(a, 0.0)


def test_newmark_matches_analytical_free_vibration():
    """Compare Newmark response with the analytical SDOF solution."""

    omega = 2.0

    M, C, K = create_sdof_system(omega)

    dt = 0.001
    t = np.arange(0.0, 2.0 + dt / 2, dt)
    ag = np.zeros_like(t)

    initial_displacement = np.array([1.0])
    initial_velocity = np.array([0.0])

    u, v, a = solve_response(
        M,
        C,
        K,
        t,
        ag,
        initial_displacement=initial_displacement,
        initial_velocity=initial_velocity,
    )

    analytical_u = np.cos(omega * t)

    error = np.max(np.abs(u[:, 0] - analytical_u))

    assert error < 1e-4


def test_newmark_time_step_refinement_improves_accuracy():
    """A smaller time step should improve the numerical solution."""

    omega = 2.0

    M, C, K = create_sdof_system(omega)

    errors = []

    for dt in (0.1, 0.05):

        t = np.arange(0.0, 2.0 + dt / 2, dt)
        ag = np.zeros_like(t)

        u, v, a = solve_response(
            M,
            C,
            K,
            t,
            ag,
            initial_displacement=np.array([1.0]),
            initial_velocity=np.array([0.0]),
        )

        analytical_u = np.cos(omega * t)

        error = np.max(
            np.abs(u[:, 0] - analytical_u)
        )

        errors.append(error)

    coarse_error, fine_error = errors

    assert fine_error < coarse_error

    # Average acceleration Newmark should show
    # approximately second-order convergence.
    assert coarse_error / fine_error > 3.5