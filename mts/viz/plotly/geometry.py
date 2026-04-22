from typing import Literal
from plotly import graph_objects as go

import numpy as np


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
    ) -> go.Scatter3d:
        rectangle_points = np.concatenate(
            [rectangle_points, rectangle_points[:1]],
            axis=0,
        )

        rectangle_trace = go.Scatter3d(
            x=rectangle_points[:, 0],
            y=rectangle_points[:, 1],
            z=rectangle_points[:, 2],
            mode="lines",
        )
        return rectangle_trace