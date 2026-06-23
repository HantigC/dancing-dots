import numpy as np
import torch


def to_torch_format(
    array: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    if not array.ndim == 4:
        if array.ndim != 3:
            raise ValueError("it should either 3d or 4d, not %d", array.ndim)
        array = array[None]
    if array.shape[1] != 3:
        array = torch.permute(array, (0, 3, 1, 2))
    return array
