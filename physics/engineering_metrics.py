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
    Calculate basic engineering response metrics for a shear building.

    Parameters
    ----------
    M : ndarray
        Mass matrix (kg).
    K : ndarray
        Stiffness matrix (N/m).
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
    # 1. INPUT VALIDATION
    # -----------------------------------------------------

    M = np.asarray(M, dtype=float)
    K = np.asarray(K, dtype=float)
    u = np.asarray(u, dtype=float)
    a = np.asarray(a, dtype=float)
    ag = np.asarray(ag, dtype=float)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix.")

    if K.ndim != 2 or K.shape != M.shape:
        raise ValueError("K must have the same square shape as M.")

    n_floors = M.shape[0]

    if u.ndim != 2 or u.shape[1] != n_floors:
        raise ValueError("u must have shape (time_steps, n_floors).")

    if a.shape != u.shape:
        raise ValueError("a must have the same shape as u.")

    if ag.ndim != 1 or len(ag) != len(u):
        raise ValueError("ag must have one value per time step.")

    if len(u) == 0:
        raise ValueError("Response arrays cannot be empty.")

    if not all(np.all(np.isfinite(x)) for x in (M, K, u, a, ag)):
        raise ValueError("Inputs must contain only finite values.")

    if not np.allclose(M, M.T) or not np.allclose(K, K.T):
        raise ValueError("M and K must be symmetric.")

    if story_height <= 0:
        raise ValueError("story_height must be positive.")

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

    max_drift_ratio = max_story_drift / story_height

    # -----------------------------------------------------
    # 4. BASE SHEAR
    # -----------------------------------------------------

    # Absolute floor acceleration = relative acceleration + ag.
    absolute_acceleration = a + ag[:, np.newaxis]

    # Total horizontal inertial force at each time step.
    floor_inertial_forces = absolute_acceleration @ M.T

    base_shear_history = np.sum(
        floor_inertial_forces,
        axis=1
    )

    max_base_shear = np.max(
        np.abs(base_shear_history)
    )

    # -----------------------------------------------------
    # 5. FIRST NATURAL FREQUENCY AND MODE SHAPE
    # -----------------------------------------------------

    eigenvalues, eigenvectors = eigh(K, M)

    if np.any(eigenvalues <= 0):
        raise ValueError(
            "The structure must have positive natural frequencies."
        )

    first_eigenvalue = eigenvalues[0]

    first_natural_frequency_hz = (
        np.sqrt(first_eigenvalue) / (2.0 * np.pi)
    )

    first_mode_shape = eigenvectors[:, 0].copy()

    # Normalize the first mode shape.
    first_mode_shape /= np.max(np.abs(first_mode_shape))
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
        "base_shear_history": base_shear_history,
        "max_base_shear": max_base_shear,
        "first_natural_frequency_hz": first_natural_frequency_hz,
        "first_mode_shape": first_mode_shape
    }