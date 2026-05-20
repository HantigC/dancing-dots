from typing import Sequence

import pycolmap

from mts.core.types import Rigid3dDict
from mts.utils.iterate import cycle_or_no
from mts.viz.plotly import colmap as plotly_colmap
from mts.viz.plotly.figure import create_new_figure
from plotly import graph_objects as go


@create_new_figure
def plot_pose_dicts(
    rigid3d_dicts: Sequence[Rigid3dDict],
    names: list[str] | None = None,
    colors: list[str] | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    names = cycle_or_no(names, "pose")
    colors = cycle_or_no(colors, "rgb(0, 255, 255)")

    for name, rigid_3d_dict, color in zip(names, rigid3d_dicts, colors):
        pose = pycolmap.Rigid3d(
            rigid_3d_dict["R"],
            rigid_3d_dict["t"],
        )
        plotly_colmap.rigid.render_axes(
            pose,
            fig=fig,
            axes_kwargs={
                "rig:name": name,
                "rig:color": color,
            },
        )
    return fig
