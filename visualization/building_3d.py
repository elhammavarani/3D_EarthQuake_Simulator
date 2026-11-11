import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


def create_building_3d(
    displacement,
    time_index=0,
    floor_height=3.2,
    width=12.0,
    depth=8.0,
    amplification=10.0,
    time=None,
    max_frames=120,
    playback_speed=1.0
):
    """
    Animated 3D shear building response.

    displacement: array with shape (time_steps, floors), in meters.
    time: optional time array in seconds.
    amplification: visual deformation scale, not physical displacement.
    """

    displacement = np.asarray(displacement, dtype=float)

    if displacement.ndim != 2 or displacement.shape[1] < 1:
        raise ValueError(
            "Displacement must be a 2D array with at least one floor."
        )

    if not np.all(np.isfinite(displacement)):
        raise ValueError("Displacement must contain finite values.")

    n_steps, n_floors = displacement.shape

    if not 0 <= time_index < n_steps:
        raise IndexError("time_index is outside the displacement array.")

    if min(
        floor_height, width, depth, amplification, playback_speed
    ) <= 0:
        raise ValueError(
            "Dimensions, amplification and speed must be positive."
        )

    if max_frames < 2:
        raise ValueError("max_frames must be at least 2.")

    if time is None:
        time = np.arange(n_steps, dtype=float)
    else:
        time = np.asarray(time, dtype=float)

    if time.ndim != 1 or len(time) != n_steps:
        raise ValueError(
            "Time must have one value for every displacement row."
        )

    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError(
            "Time values must be finite and strictly increasing."
        )

    # Select frames for responsive animation.
    frame_indices = np.unique(
        np.linspace(
            0,
            n_steps - 1,
            min(n_steps, max_frames),
            dtype=int
        )
    )

    frame_indices = np.unique(
        np.append(frame_indices, time_index)
    )

    corners = [
        (-width / 2, -depth / 2),
        (width / 2, -depth / 2),
        (width / 2, depth / 2),
        (-width / 2, depth / 2)
    ]

    z_levels = np.arange(n_floors + 1) * floor_height

    global_max = max(
        np.max(np.abs(displacement)),
        1e-12
    )

    # Maximum displacement of each floor over the entire simulation.
    max_floor_displacement = np.max(
        np.abs(displacement),
        axis=0
    )

    def get_color(value):
        ratio = np.clip(value / global_max, 0, 1)
        return sample_colorscale("Turbo", [ratio])[0]

    def make_frame_data(index):

        floor_u = displacement[index]

        shifts = np.concatenate(
            ([0.0], floor_u)
        ) * amplification

        traces = []

        # 1. Vertical columns
        for corner_x, corner_y in corners:

            traces.append(
                go.Scatter3d(
                    x=[corner_x + shift for shift in shifts],
                    y=[corner_y] * (n_floors + 1),
                    z=z_levels,
                    mode="lines",
                    line=dict(
                        color="#38BDF8",
                        width=7
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )

        # 2. Floor beams with response-based colors
        for floor in range(n_floors + 1):

            shift = shifts[floor]
            z = z_levels[floor]

            response = abs(shifts[floor]) / max(
                global_max * amplification,
                1e-12
            )

            color = get_color(response * global_max)

            for j in range(4):

                x1, y1 = corners[j]
                x2, y2 = corners[(j + 1) % 4]
                traces.append(
                    go.Scatter3d(
                        x=[
                            x1 + shift,
                            x2 + shift
                        ],
                        y=[y1, y2],
                        z=[z, z],
                        mode="lines",
                        line=dict(
                            color=color,
                            width=6
                        ),
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )

        # 3. Current floor displacement markers
        traces.append(
            go.Scatter3d(
                x=shifts[1:],
                y=np.zeros(n_floors),
                z=z_levels[1:],
                mode="markers",
                marker=dict(
                    size=7,
                    color=np.abs(floor_u),
                    colorscale="Turbo",
                    cmin=0,
                    cmax=global_max,
                    showscale=True,
                    colorbar=dict(
                        title="|Displacement| (m)"
                    )
                ),
                name="Current displacement",
                customdata=np.column_stack((
                    np.arange(1, n_floors + 1),
                    floor_u,
                    max_floor_displacement
                )),
                hovertemplate=(
                    "Floor %{customdata[0]}<br>"
                    "Current displacement: "
                    "%{customdata[1]:.5f} m<br>"
                    "Maximum displacement: "
                    "%{customdata[2]:.5f} m"
                    "<extra></extra>"
                )
            )
        )

        # 4. Maximum displacement labels
        traces.append(
            go.Scatter3d(
                x=shifts[1:],
                y=np.full(n_floors, depth / 2 + 1.0),
                z=z_levels[1:],
                mode="text",
                text=[
                    f"{value:.4f} m"
                    for value in max_floor_displacement
                ],
                textposition="middle right",
                textfont=dict(
                    color="white",
                    size=10
                ),
                name="Maximum floor displacement",
                showlegend=False,
                hoverinfo="skip"
            )
        )

        return traces

    # Initial figure
    initial_data = make_frame_data(time_index)
    trace_indices = list(range(len(initial_data)))

    fig = go.Figure(data=initial_data)

    # Animation frames
    frames = []

    for index in frame_indices:

        frame_time = time[index]
        max_u = np.max(np.abs(displacement[index]))

        frames.append(
            go.Frame(
                name=str(index),
                data=make_frame_data(index),
                traces=trace_indices,
                layout=go.Layout(
                    title=(
                        "3D Earthquake Simulation | "
                        f"Time: {frame_time:.2f} s | "
                        f"Current max displacement: {max_u:.4f} m"
                    )
                )
            )
        )

    fig.frames = frames

    # Timeline slider
    slider_steps = []

    for index in frame_indices:

        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [str(index)],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": 0,
                            "redraw": True
                        },
                        "transition": {
                            "duration": 0
                        }
                    }
                ],
                label=f"{time[index]:.1f}"
            )
        )

    frame_duration = max(
        1,
        int(40 / playback_speed)
    )
    # Layout and animation controls
    fig.update_layout(
        title=(
            "3D Earthquake Simulation | "
            f"Time: {time[time_index]:.2f} s | "
            f"Current max displacement: "
            f"{np.max(np.abs(displacement[time_index])):.4f} m"
        ),
        template="plotly_dark",
        height=750,
        margin=dict(
            l=0,
            r=0,
            t=100,
            b=40
        ),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Height (m)",
            aspectmode="manual",
            aspectratio=dict(
                x=1.3,
                y=1.0,
                z=1.8
            ),
            camera=dict(
                eye=dict(
                    x=1.8,
                    y=1.8,
                    z=1.2
                )
            )
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=1.15,
                buttons=[
                    dict(
                        label="▶️ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {
                                    "duration": frame_duration,
                                    "redraw": True
                                },
                                "transition": {
                                    "duration": 0
                                },
                                "fromcurrent": True,
                                "mode": "immediate"
                            }
                        ]
                    ),
                    dict(
                        label="Ⅱ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False
                                },
                                "mode": "immediate",
                                "transition": {
                                    "duration": 0
                                }
                            }
                        ]
                    ),
                    dict(
                        label="↻ Reset",
                        method="animate",
                        args=[
                            [str(frame_indices[0])],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": True
                                },
                                "mode": "immediate",
                                "transition": {
                                    "duration": 0
                                }
                            }
                        ]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=int(
                    np.where(frame_indices == time_index)[0][0]
                ),
                currentvalue={
                    "prefix": "Earthquake time: "
                },
                pad={
                    "t": 50
                },
                steps=slider_steps
            )
        ]
    )

    return fig