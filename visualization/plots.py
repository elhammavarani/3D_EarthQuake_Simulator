import numpy as np
import plotly.graph_objects as go


def plot_ground_motion(t, ag):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t,
        y=ag,
        mode="lines",
        name="Ground Acceleration"
    ))

    fig.update_layout(
        title="Earthquake Ground Acceleration",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (m/s²)",
        template="plotly_white"
    )

    return fig


def plot_story_response(t, response, title, y_label):
    response = np.asarray(response)

    fig = go.Figure()

    for i in range(response.shape[1]):
        fig.add_trace(go.Scatter(
            x=t,
            y=response[:, i],
            mode="lines",
            name=f"Floor {i + 1}"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="x unified",
        legend_title="Building Floors"
    )

    return fig


def plot_max_story_response(response, title, x_label):
    max_values = np.max(np.abs(response), axis=0)

    floors = np.arange(1, response.shape[1] + 1)

    fig = go.Figure(go.Bar(
        x=floors,
        y=max_values
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Floor",
        yaxis_title=x_label,
        template="plotly_white"
    )

    return fig