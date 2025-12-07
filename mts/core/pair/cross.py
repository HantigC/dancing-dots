import numpy as np
import torch

from mts.core.types import PairType
from mts.helpers.torch.tensor import from_np, to_2d


def compute_cross_pairs(
    image_embeddings: list[np.ndarray]
    | np.ndarray
    | torch.Tensor
    | list[torch.Tensor,],
    cutoff_th=0.6,  # should be strict
    distance_th: float = 1000,
    min_pairs=20,
) -> list[PairType]:
    image_embeddings = from_np(image_embeddings)
    image_embeddings = to_2d(image_embeddings)

    dm = torch.cdist(image_embeddings, image_embeddings, p=2).detach().cpu().numpy()
    mask = dm <= cutoff_th

    num_embeddings = len(image_embeddings)
    ar = np.arange(num_embeddings)
    matching_list: list[PairType] = []

    for st_idx in range(num_embeddings - 1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < distance_th:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))

    matching_list = sorted(list(set(matching_list)))
    return matching_list
