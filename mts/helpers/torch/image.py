import numpy as np
import torch


def to_torch_format(
    array: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    if not array.ndim == 4:
        array = array.unsqueeze(0)

    array = torch.permute(array, (0, 3, 1, 2))
    return array
