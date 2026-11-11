import numpy as np
from scipy.linalg import eigh


def calculate_engineering_metrics(
    M,
    K,
    u,
    a,
    ag,
    story_height=3.2
):
    """
    Calculate engineering response metrics for a linear shear building.

    Parameters
    ----------
    M : ndarray
        Mass matrix (kg), shape (n_floors, n_floors).
    K : ndarray
        Stiffness matrix (N/m), shape (n_floors, n_floors).
    u : ndarray
        Relative floor displacement history (m).
        Shape: (time_steps, n_floors).
    a : ndarray
        Relative floor acceleration history (m/s^2).
        Shape: (time_steps, n_floors).
    ag : ndarray
        Ground acceleration history (m/s^2).
        Shape: (time_steps,).
    story_height : float
        Height of each story (m).

    Returns
    -------
    dict
        Engineering metrics and response arrays.
    """

    # -----------------------------------------------------
    # 1. INPUT CONVERSION AND VALIDATION
    # -----------------------------------------------------

    M = np.asarray(M, dtype=float)
    K = np.asarray(K, dtype=float)
    u = np.asarray(u, dtype=float)
    a = np.asarray(a, dtype=float)
    ag = np.asarray(ag, dtype=float)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix.")

    n_floors = M.shape[0]

    if n_floors < 1:
        raise ValueError("The building must have at least one floor.")

    if K.shape != M.shape:
        raise ValueError("K must have the same shape as M.")

    if u.ndim != 2 or u.shape[1] != n_floors:
        raise ValueError(
            "u must have shape (time_steps, n_floors)."
        )

    if u.shape[0] == 0:
        raise ValueError("Response arrays cannot be empty.")

    if a.shape != u.shape:
        raise ValueError("a must have the same shape as u.")

    if ag.ndim != 1 or len(ag) != len(u):
        raise ValueError(
            "ag must have one value per time step."
        )

    if not all(
        np.all(np.isfinite(x))
        for x in (M, K, u, a, ag)
    ):
        raise ValueError("Inputs must contain only finite values.")

    if not np.isscalar(story_height):
        raise ValueError("story_height must be a scalar.")

    if not np.isfinite(story_height) or story_height <= 0:
        raise ValueError("story_height must be finite and positive.")

    if not np.allclose(M, M.T, rtol=1e-8, atol=1e-10):
        raise ValueError("M must be symmetric.")

    if not np.allclose(K, K.T, rtol=1e-8, atol=1e-10):
        raise ValueError("K must be symmetric.")

    # Both matrices must be positive definite
    # for this fixed-base shear-building model.

    try:
        np.linalg.cholesky(M)
        np.linalg.cholesky(K)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "M and K must be positive definite."
        ) from exc

    # -----------------------------------------------------
    # 2. MAXIMUM FLOOR DISPLACEMENT
    # -----------------------------------------------------

    max_floor_displacement = np.max(
        np.abs(u),
        axis=0
    )

    # -----------------------------------------------------
    # 3. STORY DRIFT AND DRIFT RATIO
    # -----------------------------------------------------

    # Ground floor displacement is assumed to be zero.

    floor_displacement = np.column_stack([
        np.zeros(len(u)),
        u
    ])

    story_drift = (
        floor_displacement[:, 1:]
        - floor_displacement[:, :-1]
    )

    max_story_drift = np.max(
        np.abs(story_drift),
        axis=0
    )

    # Drift ratio is dimensionless.
    # Multiply by 100 when displaying as percentage.

    max_drift_ratio = max_story_drift / story_height

    # -----------------------------------------------------
    # 4. ABSOLUTE ACCELERATION AND BASE SHEAR
    # -----------------------------------------------------

    # Uniform base excitation:
    # a_absolute = a_relative + influence_vector * ag

    influence_vector = np.ones(n_floors)

    absolute_acceleration = (
        a + ag[:, np.newaxis] * influence_vector
    )
    # Floor inertial forces.
    # Each row represents one time step.

    floor_inertial_forces = (
        absolute_acceleration @ M.T
    )

    # Total horizontal base shear.
    # Sign follows the chosen inertial-force convention.

    base_shear_history = np.sum(
        floor_inertial_forces,
        axis=1
    )

    max_base_shear = np.max(
        np.abs(base_shear_history)
    )

    # -----------------------------------------------------
    # 5. NATURAL FREQUENCY AND FIRST MODE SHAPE
    # -----------------------------------------------------

    try:
        eigenvalues, eigenvectors = eigh(K, M)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Unable to solve the generalized eigenvalue problem."
        ) from exc

    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Eigenvalues must be finite.")

    if np.any(eigenvalues <= 0):
        raise ValueError(
            "All natural frequencies must be positive."
        )

    first_eigenvalue = eigenvalues[0]

    first_natural_frequency_hz = (
        np.sqrt(first_eigenvalue) / (2.0 * np.pi)
    )

    first_mode_shape = eigenvectors[:, 0].copy()

    # Normalize maximum absolute component to 1.

    mode_max = np.max(np.abs(first_mode_shape))

    if mode_max <= 0:
        raise ValueError("Invalid first mode shape.")

    first_mode_shape /= mode_max

    # Keep the top-floor mode component positive.

    if first_mode_shape[-1] < 0:
        first_mode_shape *= -1

    # -----------------------------------------------------
    # 6. RETURN RESULTS
    # -----------------------------------------------------

    return {
        "max_floor_displacement": max_floor_displacement,
        "story_drift": story_drift,
        "max_story_drift": max_story_drift,
        "max_drift_ratio": max_drift_ratio,
        "absolute_acceleration": absolute_acceleration,
        "base_shear_history": base_shear_history,
        "max_base_shear": max_base_shear,
        "first_natural_frequency_hz": first_natural_frequency_hz,
        "first_mode_shape": first_mode_shape
    }