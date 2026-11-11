import numpy as np
from scipy.linalg import cho_factor, cho_solve


def solve_response(
    M,
    C,
    K,
    t,
    ag,
    beta=0.25,
    gamma=0.5,
    initial_displacement=None,
    initial_velocity=None,
):
    """
    Solve the dynamic response of a multi-degree-of-freedom
    structural system using the Newmark-Beta method.

    The equation of motion is:

        M*u_ddot + C*u_dot + K*u = -M*r*ag

    Parameters
    ----------
    M, C, K : numpy.ndarray
        Mass, damping and stiffness matrices.
    t : numpy.ndarray
        Equally spaced time array (seconds).
    ag : numpy.ndarray
        Ground acceleration array (m/s^2).
    beta, gamma : float
        Newmark integration parameters.
        Defaults correspond to the average acceleration method.
    initial_displacement : numpy.ndarray, optional
        Initial relative displacement vector.
    initial_velocity : numpy.ndarray, optional
        Initial relative velocity vector.

    Returns
    -------
    displacement : numpy.ndarray
        Relative displacement (m).
    velocity : numpy.ndarray
        Relative velocity (m/s).
    acceleration : numpy.ndarray
        Relative acceleration (m/s^2).
    """

    # Convert inputs
    M = np.asarray(M, dtype=float)
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    t = np.asarray(t, dtype=float)
    ag = np.asarray(ag, dtype=float)

    # Validate matrix dimensions
    if M.ndim != 2 or M.shape[0] == 0:
        raise ValueError("M must be a non-empty 2D matrix.")

    n = M.shape[0]

    if M.shape != (n, n):
        raise ValueError("M must be square.")

    if C.shape != (n, n) or K.shape != (n, n):
        raise ValueError(
            "M, C and K must be square matrices of equal size."
        )

    # Validate finite values
    for name, matrix in (("M", M), ("C", C), ("K", K)):
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} contains non-finite values.")

    if (
        t.ndim != 1
        or ag.ndim != 1
        or len(t) != len(ag)
    ):
        raise ValueError(
            "Time and ground acceleration must be "
            "1D arrays of equal length."
        )

    if len(t) < 2:
        raise ValueError("At least two time points are required.")

    if not np.all(np.isfinite(t)):
        raise ValueError("Time array contains non-finite values.")

    if not np.all(np.isfinite(ag)):
        raise ValueError(
            "Ground acceleration contains non-finite values."
        )

    # Validate matrix symmetry
    for name, matrix in (("M", M), ("C", C), ("K", K)):
        if not np.allclose(
            matrix,
            matrix.T,
            rtol=1e-8,
            atol=1e-10
        ):
            raise ValueError(f"{name} must be symmetric.")

    # Validate positive definiteness of mass and stiffness
    try:
        cho_factor(M, lower=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Mass matrix M must be positive definite."
        ) from exc

    try:
        cho_factor(K, lower=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Stiffness matrix K must be positive definite."
        ) from exc

    # Validate time step
    dt_values = np.diff(t)

    if np.any(dt_values <= 0):
        raise ValueError(
            "Time values must be strictly increasing."
        )

    dt = dt_values[0]

    if not np.allclose(dt_values, dt):
        raise ValueError(
            "Time steps must be equally spaced."
        )

    # Validate Newmark parameters
    if (
        not np.isfinite(beta)
        or not np.isfinite(gamma)
        or beta <= 0
        or gamma <= 0
    ):
        raise ValueError(
            "Beta and gamma must be finite and positive."
        )

    # Initial conditions
    if initial_displacement is None:
        u0 = np.zeros(n)
    else:
        u0 = np.asarray(initial_displacement, dtype=float)

    if initial_velocity is None:
        v0 = np.zeros(n)
    else:
        v0 = np.asarray(initial_velocity, dtype=float)

    if u0.shape != (n,) or v0.shape != (n,):
        raise ValueError(
            "Initial displacement and velocity "
            "must have shape (number_of_floors,)."
        )

    if not np.all(np.isfinite(u0)) or not np.all(np.isfinite(v0)):
        raise ValueError(
            "Initial conditions must contain only finite values."
        )

    # Newmark coefficients
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    # Effective stiffness matrix
    K_eff = K + a0 * M + a1 * C

    try:
        K_factor = cho_factor(K_eff, lower=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Effective stiffness matrix is not positive definite. "
            "Check system matrices and integration parameters."
        ) from exc

    # Allocate response arrays
    num_steps = len(t)

    displacement = np.zeros((num_steps, n))
    velocity = np.zeros((num_steps, n))
    acceleration = np.zeros((num_steps, n))

    # Influence vector for uniform ground excitation
    influence = np.ones(n)

    # Set initial conditions
    displacement[0] = u0
    velocity[0] = v0

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
            K_factor,
            effective_force
        )

        acceleration[i + 1] = (
            a0 * (
                displacement[i + 1] - displacement[i]
            )
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