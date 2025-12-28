from dataclasses import dataclass

import numpy as np


@dataclass
class Rigid3D:
    rotation: np.ndarray
    translation: np.ndarray