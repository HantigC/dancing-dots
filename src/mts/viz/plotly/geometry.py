from typing import Any, Literal
from plotly import graph_objects as go

import numpy as np


class Line2D:
    @staticmethod
    def to_trace(
        x: np.ndarray,
        y: np.ndarray,
        fill: Literal["tozeroy", "tozerox", "tonexty", "none"] = "tozeroy",
        scatter_kwargs: dict[str, Any] = None,
    ) -> go.Scatter:
        if scatter_kwargs is None:
            scatter_kwargs = {}
        scatter_kwargs.setdefault("showlegend", False)
        return go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill=fill,
            **scatter_kwargs,
        )


class Rectangle:
    @staticmethod
    def to_trace(
        rectangle_points: np.ndarray[
            tuple[
                Literal[4],
                Literal[3],
            ],
            np.dtype[np.float32],
        ],
        scatter_kwargs: dict[str, Any] = None
    ) -> go.Scatter3d:
        if scatter_kwargs is None:
            scatter_kwargs = {}
        scatter_kwargs.setdefault("showlegend", False)
        rectangle_points = np.concatenate(
            [rectangle_points, rectangle_points[:1]],
            axis=0,
        )

        rectangle_trace = go.Scatter3d(
            x=rectangle_points[:, 0],
            y=rectangle_points[:, 1],
            z=rectangle_points[:, 2],
            mode="lines",
            **scatter_kwargs,
        )
        return rectangle_trace