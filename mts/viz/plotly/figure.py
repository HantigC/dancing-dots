from functools import wraps
from typing import Any
from plotly import graph_objects as go


def init_figure(height: int = 800) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        scene_camera=dict(
            eye=dict(x=0.0, y=-0.1, z=-2),
            up=dict(x=0, y=-1.0, z=0),
            projection=dict(type="orthographic"),
        ),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.1),
    )
    return fig


def create_new_figure(
    _func=None,
    *,
    init_figure=init_figure,
    force=True,
):
    def _wrapper(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            if "fig" not in kwargs:
                fig = init_figure()
                return fn(*args, fig=fig, **kwargs)
            elif kwargs.get("fig") is None and force:
                fig = init_figure()
                kwargs["fig"] = fig
                return fn(*args, **kwargs)

            return fn(*args, **kwargs)

        return wrapper

    if _func is None:
        return _wrapper
    else:
        return _wrapper(_func)