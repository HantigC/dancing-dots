from typing import Any

from plotly import graph_objects as go

import pycolmap

from mts.viz.plotly import axes
from mts.viz.plotly.figure import create_new_figure


@create_new_figure
def render_axes(
    rigid3d: pycolmap.Rigid3d,
    scale: float = 1,
    fig: go.Figure = None,
    axes_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    position = rigid3d.inverse().translation
    rotation = rigid3d.rotation.matrix()
    return axes.render_axes(
        fig,
        position=position,
        xaxis=rotation[0],
        yaxis=rotation[1],
        zaxis=rotation[2],
        scale=scale,
        axes_kwargs=axes_kwargs,
    )