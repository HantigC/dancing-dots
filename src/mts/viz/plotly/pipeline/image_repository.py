import math
from pathlib import Path

import pycolmap
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from mts.core.types import ImageId
from mts.pipeline.repository.base import BaseImageRepository
from mts.viz import plotly as viz_plotly
from mts.viz.plotly.figure import create_new_figure


def plot_image_kpts_grid(
    image_repository: BaseImageRepository,
    image_ids: list[ImageId],
    cols: int = 5,
    name: str = "keypoints",
) -> go.Figure:
    image_ids = list(image_ids)
    rows = math.ceil(len(image_ids) / cols)

    titles = [Path(image_repository.get_filepath(image_id)).name for image_id in image_ids]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    scatter_visible_off = []

    for i, image_id in enumerate(image_ids):
        row = i // cols + 1
        col = i % cols + 1
        img = image_repository.load_image(image_id)
        kpts = image_repository.get_keypoints(image_id, name=name)
        fig.add_trace(go.Image(z=img), row=row, col=col)
        scatter_visible_off.append(True)
        if kpts is not None and len(kpts) > 0:
            fig.add_trace(
                go.Scatter(
                    x=kpts[:, 0], y=kpts[:, 1],
                    mode="markers",
                    marker=dict(color="lime", size=2),
                    showlegend=False,
                ),
                row=row, col=col,
            )
            scatter_visible_off.append(False)

    scatter_visible_on = [True] * len(scatter_visible_off)

    fig.update_annotations(font_size=10)
    fig.update_layout(
        height=rows * 350,
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Show keypoints", "method": "restyle",
                 "args": [{"visible": scatter_visible_on}]},
                {"label": "Hide keypoints", "method": "restyle",
                 "args": [{"visible": scatter_visible_off}]},
            ],
            "x": 0, "y": 1.02, "xanchor": "left",
        }],
    )
    return fig


def plot_image_kpts_interactive(
    image_repository: BaseImageRepository,
    image_ids: list[ImageId],
) -> go.Figure:
    image_ids = list(image_ids)
    filenames = [Path(image_repository.get_filepath(image_id)).name for image_id in image_ids]

    fig = go.Figure()

    for i, image_id in enumerate(image_ids):
        img = image_repository.load_image(image_id)
        kpts = image_repository.get_keypoints(image_id)
        visible = i == 0
        fig.add_trace(go.Image(z=img, visible=visible))
        fig.add_trace(go.Scatter(
            x=kpts[:, 0], y=kpts[:, 1],
            mode="markers",
            marker=dict(color="lime", size=3, opacity=0.4),
            name="keypoints",
            visible=visible,
            showlegend=False,
        ))

    n = len(image_ids)
    image_indices = list(range(0, 2 * n, 2))
    kpts_indices = list(range(1, 2 * n, 2))

    image_buttons = [
        {
            "label": filename,
            "method": "restyle",
            "args": [{"visible": [j // 2 == i for j in range(n)]}, image_indices],
        }
        for i, filename in enumerate(filenames)
    ]

    fig.update_layout(
        updatemenus=[
            {
                "type": "dropdown",
                "buttons": image_buttons,
                "x": 0, "y": 1.08, "xanchor": "left",
            },
            {
                "type": "buttons",
                "buttons": [
                    {"label": "Show keypoints", "method": "restyle",
                     "args": [{"visible": True}, kpts_indices]},
                    {"label": "Hide keypoints", "method": "restyle",
                     "args": [{"visible": False}, kpts_indices]},
                ],
                "x": 1, "y": 1.08, "xanchor": "right",
            },
        ],
    )
    return fig


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
