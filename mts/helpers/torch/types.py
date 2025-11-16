from typing import Hashable, Iterable

import torch

TensorCollectionType = (
    dict[Hashable, torch.Tensor] | Iterable[torch.Tensor] | torch.Tensor
)
