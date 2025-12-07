from typing import Hashable, Iterable, TypeVar

import numpy as np
import torch
T = TypeVar("T", np.ndarray, torch.Tensor)
CollectionType = dict[Hashable, T] | Iterable[T] | T

TensorCollectionType = CollectionType[torch.Tensor]
NPArrayCollectionType = CollectionType[np.ndarray]
