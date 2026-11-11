from physics.building import create_building
from physics.earthquake import generate_ground_motion
from physics.solver import solve_response

from visualization.plots import (
    plot_ground_motion,
    plot_story_response,
    plot_max_story_response
)


def test_plots():
    M, C, K = create_building()
    t, ag = generate_ground_motion()

    u, v, a = solve_response(M, C, K, t, ag)

    fig1 = plot_ground_motion(t, ag)
    fig2 = plot_story_response(
        t, u, "Story Displacement", "Displacement (m)"
    )
    fig3 = plot_story_response(
        t, v, "Story Velocity", "Velocity (m/s)"
    )
    fig4 = plot_story_response(
        t, a, "Story Acceleration", "Acceleration (m/s²)"
    )
    fig5 = plot_max_story_response(
        u, "Maximum Story Displacement", "Displacement (m)"
    )

    figures = [fig1, fig2, fig3, fig4, fig5]

    assert all(len(fig.data) > 0 for fig in figures)

    print("SUCCESS: All visualization tests passed!")


if __name__ == "main":
    test_plots()