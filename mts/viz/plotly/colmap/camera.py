import pycolmap

from mts.viz.plotly.axes import render_axes as _render_axes
from mts.viz.plotly.figure import create_new_figure
from plotly import graph_objects as go


@create_new_figure
def render_axes(
    cam_from_world: pycolmap.Rigid3d,
    fig: go.Figure | None,
    **kwargs,
) -> go.Figure:
    position = cam_from_world.inverse().matrix() @ [0, 0, 0, 1]
    xaxis, yaxis, zaxis = cam_from_world.rotation.matrix()
    return _render_axes(
        position,
        xaxis,
        yaxis,
        zaxis,
        fig=fig,
        **kwargs,
    )


@create_new_figure
def render_camera_axes(
    image: pycolmap.Image,
    fig: go.Figure | None = None,
    **kwargs,
) -> go.Figure:
    position = image.cam_from_world.inverse().matrix() @ [0, 0, 0, 1]
    xaxis, yaxis, zaxis = image.cam_from_world.rotation.matrix()
    return _render_axes(
        position,
        xaxis,
        yaxis,
        zaxis,
        fig=fig,
        **kwargs,
    )
