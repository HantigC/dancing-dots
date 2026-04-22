import pycolmap

from mts.core.types import ImageId
from mts.pipeline.repository.base import BaseImageRepository
from mts.viz import plotly as viz_plotly
from mts.viz.plotly.figure import create_new_figure
from plotly import graph_objects as go


@create_new_figure
def plot_image_id_poses(
    images_ids: list[ImageId],
    image_repository: BaseImageRepository,
    fig: go.Figure | None = None,
) -> go.Figure:
    for image_id in images_ids:
        rigid_3d = image_repository.get_pose(image_id)
        filepath = image_repository.get_filepath(image_id)
        filename = filepath.rsplit("/", 1)[1]
        pose = pycolmap.Rigid3d(
            rigid_3d.rotation,
            rigid_3d.translation,
        )
        viz_plotly.colmap.rigid.render_axes(
            pose,
            fig=fig,
            axes_kwargs={"rig:name": filename},
        )
    return fig
