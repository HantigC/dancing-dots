
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Prediction:
    image_id: str | None
    dataset: str
    filename: str
    image_filepath: Path
    cluster_index: int | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None



class IMC2025ReconstructionPipeline:

    def __init__(self) -> None:
        pass