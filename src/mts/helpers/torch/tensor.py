import numpy as np
import torch

from .types import NPArrayCollectionType, TensorCollectionType


def to(
    tensor_collection: TensorCollectionType,
    dtype=None,
    device=None,
    non_blocking: bool = False,
) -> TensorCollectionType:
    # TODO: Maybe use a stack for recursive calls. Only if needed
    if isinstance(tensor_collection, torch.Tensor):
        return tensor_collection.to(device=device, dtype=dtype, non_blocking=non_blocking,)
    elif isinstance(tensor_collection, dict):
        return {
            key: to(value, dtype, device) for key, value in tensor_collection.items()
        }
    elif isinstance(tensor_collection, list):
        return [to(value, dtype, device) for value in tensor_collection]
    elif isinstance(tensor_collection, tuple):
        return tuple(to(value, dtype, device) for value in tensor_collection)
    return tensor_collection


def to_2d(
    tensor: torch.Tensor | list[torch.Tensor],
) -> torch.Tensor:
    if isinstance(tensor, list | tuple):
        if not isinstance(tensor[0], torch.Tensor):
            raise ValueError("It should be a list of tensors or a 2d tensor")
        tensor = torch.stack(tensor)
    if tensor.ndim != 2:
        raise ValueError(
            f"it should be a list of 1d tensors or a 2d tensor, not {tensor.ndim}d"
        )

    return tensor


def to_numpy(
    tensor_collection: TensorCollectionType,
) -> NPArrayCollectionType:
    if isinstance(tensor_collection, torch.Tensor):
        return tensor_collection.detach().cpu().numpy()
    elif isinstance(tensor_collection, dict):
        return {
            key: to_numpy(value) for key, value in tensor_collection.items()
        }
    elif isinstance(tensor_collection, list):
        return [to_numpy(value) for value in tensor_collection]
    elif isinstance(tensor_collection, tuple):
        return tuple(to_numpy(value) for value in tensor_collection)
    return tensor_collection


def from_np(
    arrays: NPArrayCollectionType,
) -> TensorCollectionType:
    if isinstance(arrays, np.ndarray):
        return torch.from_numpy(arrays)
    elif isinstance(arrays, torch.Tensor):
        return arrays
    elif isinstance(arrays, dict):
        return {key: from_np(value) for key, value in arrays.items()}
    elif isinstance(arrays, list):
        return [from_np(value) for value in arrays]
    elif isinstance(arrays, tuple):
        return tuple(from_np(value) for value in arrays)
    raise ValueError(f"`{arrays.__class__.__name__}` not supported")
