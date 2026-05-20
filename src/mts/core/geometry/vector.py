import numpy as np


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / np.linalg.norm(v, axis=axis, keepdims=True)
