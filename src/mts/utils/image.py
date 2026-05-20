from pathlib import Path
import cv2
import numpy as np


def imread_rgb(image_filepath: str | Path) -> np.ndarray:
    if isinstance(image_filepath, Path):
        image_filepath = str(image_filepath)

    img = cv2.imread(image_filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
