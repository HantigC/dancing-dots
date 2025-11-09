from plotly import graph_objects as go
import pycolmap
from plycam.axes import render_axes as _render_axes


def render_axes(
    fig: go.Figure, cam_from_world: pycolmap.Rigid3d, **kwargs
) -> go.Figure:
    position = cam_from_world.inverse().matrix() @ [0, 0, 0, 1]
    xaxis, yaxis, zaxis = cam_from_world.rotation.matrix()
    return _render_axes(fig, position, xaxis, yaxis, zaxis, **kwargs)


def render_camera_axes(fig: go.Figure, image: pycolmap.Image, **kwargs) -> go.Figure:
    position = image.cam_from_world.inverse().matrix() @ [0, 0, 0, 1]
    xaxis, yaxis, zaxis = image.cam_from_world.rotation.matrix()
    return _render_axes(fig, position, xaxis, yaxis, zaxis, **kwargs)
