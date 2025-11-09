import numpy as np
from plotly import graph_objects as go
from .core import create_new_figure


@create_new_figure
def render_np_3d_points(
    points_3d: np.ndarray,
    color=None,
    fig: go.Figure = None,
    **kwargs,
) -> go.Figure:
    fig.add_trace(
        go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode="markers",
            marker=dict(
                # symbol="x",
                size=1,
                color=color,
            ),
            **kwargs,
        )
    )
    return fig
