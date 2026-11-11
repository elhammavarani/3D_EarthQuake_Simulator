import numpy as np
from scipy.linalg import eigh


def create_building(
    n_floors=10,
    mass=250_000,
    stiffness=150_000_000,
    damping_ratio=0.05
):
    """
    Create a multi-story shear building model.

    Parameters
    ----------
    n_floors : int
        Number of building floors.
    mass : float
        Lumped mass of each floor (kg).
    stiffness : float
        Lateral stiffness of each story (N/m).
    damping_ratio : float
        Target damping ratio for the first two modes.

    Returns
    -------
    M, C, K : numpy.ndarray
        Mass, damping, and stiffness matrices.
    """

    if not isinstance(n_floors, int) or n_floors < 1:
        raise ValueError("n_floors must be a positive integer.")

    if mass <= 0 or stiffness <= 0:
        raise ValueError("Mass and stiffness must be positive.")

    if not 0 <= damping_ratio < 1:
        raise ValueError("damping_ratio must be between 0 and 1.")

    # Mass matrix
    M = np.eye(n_floors) * mass

    # Stiffness matrix
    K = np.zeros((n_floors, n_floors))

    for i in range(n_floors):
        K[i, i] += stiffness

        if i > 0:
            K[i, i] += stiffness
            K[i, i - 1] -= stiffness
            K[i - 1, i] -= stiffness

    # Natural frequencies
    eigenvalues, _ = eigh(K, M)
    omega = np.sqrt(eigenvalues)

    # Rayleigh damping
    if n_floors >= 2:
        w1, w2 = omega[0], omega[1]

        A = np.array([
            [1 / (2 * w1), w1 / 2],
            [1 / (2 * w2), w2 / 2]
        ])

        alpha, beta = np.linalg.solve(
            A,
            [damping_ratio, damping_ratio]
        )

    else:
        alpha = 2 * damping_ratio * omega[0]
        beta = 0.0

    C = alpha * M + beta * K

    return M, C, K