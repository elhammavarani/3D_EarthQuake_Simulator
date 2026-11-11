import numpy as np
from physics.building import create_building


def test_building_model():
    print("1. Creating building model...", flush=True)

    M, C, K = create_building()

    print("2. Checking matrix dimensions...", flush=True)
    assert M.shape == (10, 10)
    assert C.shape == (10, 10)
    assert K.shape == (10, 10)

    print("3. Checking symmetry...", flush=True)
    assert np.allclose(M, M.T)
    assert np.allclose(C, C.T)
    assert np.allclose(K, K.T)

    print("4. Checking positive definiteness...", flush=True)
    assert np.all(np.linalg.eigvalsh(M) > 0)
    assert np.all(np.linalg.eigvalsh(K) > 0)

    print("5. Checking mass values...", flush=True)
    assert np.allclose(np.diag(M), 250_000)

    print("\nSUCCESS: All building model tests passed!", flush=True)


if __name__ == "main":
    test_building_model()
