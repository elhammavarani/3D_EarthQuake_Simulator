import numpy as np

from physics.building import create_building
from physics.earthquake import generate_ground_motion
from physics.solver import solve_response
from visualization.building_3d import create_building_3d


def test_building_3d():

    print("1. Creating building...", flush=True)
    M, C, K = create_building()

    print("2. Generating earthquake...", flush=True)
    t, ag = generate_ground_motion()

    print("3. Solving structural response...", flush=True)
    u, v, a = solve_response(M, C, K, t, ag)

    print("4. Creating animated 3D model...", flush=True)

    fig = create_building_3d(
        displacement=u,
        time=t,
        time_index=0,
        amplification=10,
        max_frames=120,
        playback_speed=1.0
    )

    print("5. Checking animation frames...", flush=True)

    assert fig is not None
    assert len(fig.data) > 0
    assert len(fig.frames) >= 2

    print("6. Checking animation controls...", flush=True)

    assert len(fig.layout.updatemenus) > 0
    assert len(fig.layout.sliders) > 0

    print("\nSUCCESS: 3D earthquake animation test passed!", flush=True)


if __name__ == "main":
    test_building_3d()