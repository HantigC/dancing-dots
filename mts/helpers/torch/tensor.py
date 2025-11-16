import torch

from .types import TensorCollectionType


def to(
    tensor_collection: TensorCollectionType,
    dtype=None,
    device=None,
) -> TensorCollectionType:
    # TODO: Maybe use a stack for recursive calls. Only if needed
    if isinstance(tensor_collection, torch.Tensor):
        return tensor_collection.to(device=device, dtype=dtype)
    elif isinstance(tensor_collection, dict):
        return {
            key: to(value, dtype, device) for key, value in tensor_collection.items()
        }
    elif isinstance(tensor_collection, list):
        return [to(value, dtype, device) for value in tensor_collection]
    elif isinstance(tensor_collection, tuple):
        return tuple(to(value, dtype, device) for value in tensor_collection)
    raise ValueError(f"`{tensor_collection.__class__.__name__}` not supported")
