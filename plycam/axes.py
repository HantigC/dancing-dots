import numpy as np
from plotly import graph_objects as go
import itertools as it

from .utils.dict import extract_kwargs


def render_axes(
    fig,
    position,
    xaxis,
    yaxis,
    zaxis,
    scale=1,
    **kwargs,
):
    named_kwargs = extract_kwargs(kwargs, ["xaxis", "yaxis", "zaxis"], merge=True)
    xaxis_kwargs = named_kwargs["xaxis"]
    yaxis_kwargs = named_kwargs["yaxis"]
    zaxis_kwargs = named_kwargs["zaxis"]
    xaxis_kwargs.setdefault("color", "rgb(255, 0, 0)")
    yaxis_kwargs.setdefault("color", "rgb(0, 255, 0)")
    zaxis_kwargs.setdefault("color", "rgb(0, 0, 255)")

    xaxis_kwargs.setdefault("name", "xaxis")
    yaxis_kwargs.setdefault("name", "yaxis")
    zaxis_kwargs.setdefault("name", "zaxis")

    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + scale * xaxis[0]],
            y=[position[1], position[1] + scale * xaxis[1]],
            z=[position[2], position[2] + scale * xaxis[2]],
            mode="lines",
            name=xaxis_kwargs["name"],
            showlegend=False,
            marker=dict(color=xaxis_kwargs["color"]),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + scale * yaxis[0]],
            y=[position[1], position[1] + scale * yaxis[1]],
            z=[position[2], position[2] + scale * yaxis[2]],
            mode="lines",
            name=yaxis_kwargs["name"],
            showlegend=False,
            marker=dict(color=yaxis_kwargs["color"]),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + scale * zaxis[0]],
            y=[position[1], position[1] + scale * zaxis[1]],
            z=[position[2], position[2] + scale * zaxis[2]],
            mode="lines",
            name=zaxis_kwargs["name"],
            showlegend=False,
            marker=dict(color=zaxis_kwargs["color"]),
        )
    )
    return fig


def render_od_rays(
    fig: go.Figure,
    origin: np.ndarray,
    directions: np.ndarray,
    scale: float = 1,
    rgbs: list[str] = None,
    row=None,
    col=None,
) -> go.Figure:
    if rgbs is None:
        rgbs = it.cycle(["rgb(255, 0, 0)"])

    for (x, y, z), rgb in zip(directions, rgbs):
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], origin[0] + scale * x],
                y=[origin[1], origin[1] + scale * y],
                z=[origin[2], origin[2] + scale * z],
                mode="lines",
                marker=dict(color=rgb),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    return fig


def render_diff_rays(
    fig: go.Figure,
    origin: np.ndarray,
    points: np.ndarray,
    scale: float = 1,
    rgbs: list[str] = None,
    row=None,
    col=None,
) -> go.Figure:
    directions = normalize(
        points - origin,
        axis=1,
        keepdims=True,
    )
    return render_od_rays(
        fig,
        origin,
        directions,
        scale,
        rgbs,
        row=row,
        col=col,
    )


def render_frustum(
    fig: go.Figure,
    position,
    rect,
    scale=None,
    name="Pyramid",
    rgb=None,
    opacity=0.5,
    row=None,
    col=None,
) -> go.Figure:
    if scale is None:
        scale = 1

    if rgb is None:
        rgb = "rgb(255, 0, 255)"

    rect = scale * rect
    rows = row
    if rows is not None:
        rows = [row, row, row, row]

    cols = col
    if cols is not None:
        cols = [col, col, col, col]

    fig.add_traces(
        [
            go.Scatter3d(
                x=[position[0], rect[0, 0]],
                y=[position[1], rect[0, 1]],
                z=[position[2], rect[0, 2]],
                mode="lines",
                name=name,
                showlegend=False,
                marker=dict(color=rgb),
            ),
            go.Scatter3d(
                x=[position[0], rect[1, 0]],
                y=[position[1], rect[1, 1]],
                z=[position[2], rect[1, 2]],
                mode="lines",
                name=name,
                showlegend=False,
                marker=dict(color=rgb),
            ),
            go.Scatter3d(
                x=[position[0], rect[2, 0]],
                y=[position[1], rect[2, 1]],
                z=[position[2], rect[2, 2]],
                mode="lines",
                name=name,
                showlegend=False,
                marker=dict(color=rgb),
            ),
            go.Scatter3d(
                x=[position[0], rect[3, 0]],
                y=[position[1], rect[3, 1]],
                z=[position[2], rect[3, 2]],
                mode="lines",
                name=name,
                showlegend=False,
                marker=dict(color=rgb),
            ),
        ],
        rows=rows,
        cols=cols,
    )
    rect = np.concatenate(
        [rect, np.expand_dims(rect[0], axis=0)],
        axis=0,
    )

    fig.add_trace(
        go.Mesh3d(
            x=rect[:, 0],
            y=rect[:, 1],
            z=rect[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=rgb,
            opacity=opacity,
        ),
        row=row,
        col=col,
    )
    return fig
