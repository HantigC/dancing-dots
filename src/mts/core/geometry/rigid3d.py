from dataclasses import dataclass

import numpy as np

from mts.core.types import Rigid3dDict


@dataclass
class Rigid3D:
    rotation: np.ndarray
    translation: np.ndarray

    def as_rigid3d_dict(self) -> Rigid3dDict:
        return {
            "R": self.rotation,
            "t": self.translation,
        }


def as_4x4_Rt(
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    return Rt
