import numpy as np
from scipy.linalg import cho_factor, cho_solve


def solve_response(M, C, K, t, ag, beta=0.25, gamma=0.5):
    """
    Solve the dynamic response of a multi-story building
    using the Newmark-Beta method.

    Parameters
    ----------
    M, C, K : numpy.ndarray
        Mass, damping, and stiffness matrices.
    t : numpy.ndarray
        Time array in seconds.
    ag : numpy.ndarray
        Ground acceleration in m/s^2.
    beta : float
        Newmark beta parameter.
    gamma : float
        Newmark gamma parameter.

    Returns
    -------
    displacement : numpy.ndarray
        Relative displacement response (m).
    velocity : numpy.ndarray
        Relative velocity response (m/s).
    acceleration : numpy.ndarray
        Relative acceleration response (m/s^2).
    """

    M = np.asarray(M, dtype=float)
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    t = np.asarray(t, dtype=float)
    ag = np.asarray(ag, dtype=float)

    n = M.shape[0]

    # Validate inputs
    if M.shape != (n, n) or C.shape != (n, n) or K.shape != (n, n):
        raise ValueError("M, C and K must be square matrices of equal size.")

    if t.ndim != 1 or ag.ndim != 1 or len(t) != len(ag):
        raise ValueError("Time and ground acceleration must be 1D arrays of equal length.")

    if len(t) < 2:
        raise ValueError("At least two time points are required.")

    if not all(np.all(np.isfinite(x)) for x in (M, C, K, t, ag)):
        raise ValueError("Inputs must contain only finite values.")

    dt_values = np.diff(t)

    if np.any(dt_values <= 0):
        raise ValueError("Time values must be strictly increasing.")

    dt = dt_values[0]

    if not np.allclose(dt_values, dt):
        raise ValueError("Time steps must be equally spaced.")

    if beta <= 0 or gamma <= 0:
        raise ValueError("Beta and gamma must be positive.")

    # Newmark coefficients
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    # Effective stiffness matrix
    K_eff = K + a0 * M + a1 * C

    # Factorize once for efficiency
    K_factor = cho_factor(K_eff)

    # Allocate response arrays
    num_steps = len(t)

    displacement = np.zeros((num_steps, n))
    velocity = np.zeros((num_steps, n))
    acceleration = np.zeros((num_steps, n))

    influence = np.ones(n)

    # Initial acceleration
    p0 = -M @ influence * ag[0]

    acceleration[0] = np.linalg.solve(
        M,
        p0 - C @ velocity[0] - K @ displacement[0]
    )

    # Time integration
    for i in range(num_steps - 1):

        p_next = -M @ influence * ag[i + 1]

        effective_force = (
            p_next
            + M @ (
                a0 * displacement[i]
                + a2 * velocity[i]
                + a3 * acceleration[i]
            )
            + C @ (
                a1 * displacement[i]
                + a4 * velocity[i]
                + a5 * acceleration[i]
            )
        )

        displacement[i + 1] = cho_solve(
            K_factor, effective_force
        )

        acceleration[i + 1] = (
            a0 * (displacement[i + 1] - displacement[i])
            - a2 * velocity[i]
            - a3 * acceleration[i]
        )

        velocity[i + 1] = (
            velocity[i]
            + dt * (
                (1.0 - gamma) * acceleration[i]
                + gamma * acceleration[i + 1]
            )
        )

    return displacement, velocity, acceleration