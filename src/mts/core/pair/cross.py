import cv2
import numpy as np
import torch

from mts.core.types import DistancedTriple, PairType
from mts.helpers.torch.tensor import from_np, to_2d


def compute_cross_pairs(
    image_embeddings: list[np.ndarray]
    | np.ndarray
    | torch.Tensor
    | list[torch.Tensor,],
    cutoff_th=0.6,  # should be strict
    distance_th: float = 1000,
    min_pairs=20,
    max_pairs_per_image: int | None = None,
    max_pairs: int | None = None,
) -> list[PairType]:
    image_embeddings = from_np(image_embeddings)
    image_embeddings = to_2d(image_embeddings)

    dm = torch.cdist(image_embeddings, image_embeddings, p=2).detach().cpu().numpy()
    mask = dm <= cutoff_th

    num_embeddings = len(image_embeddings)
    ar = np.arange(num_embeddings)

    matching_dict: dict[PairType[int], DistancedTriple] = {}
    for st_idx in range(num_embeddings):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]
        if max_pairs_per_image is not None and len(to_match) > max_pairs_per_image:
            to_match = np.argsort(dm[st_idx])[: max_pairs_per_image + 1]
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < distance_th:
                st, nd = sorted((st_idx, idx.item()))
                matching_dict[st, nd] = DistancedTriple(st, nd, dm[st_idx, idx])

    matching_list = sorted(list(matching_dict.values()))
    if max_pairs is not None:
        matching_list = matching_list[:max_pairs]
    return matching_list


def compute_knn_pairs(
    image_embeddings: list[np.ndarray] | np.ndarray | torch.Tensor | list[torch.Tensor],
    k: int = 20,
    distance_th: float = 1000,
    max_pairs_per_image: int | None = None,
    max_pairs: int | None = None,
) -> list[DistancedTriple]:
    image_embeddings = from_np(image_embeddings)
    image_embeddings = to_2d(image_embeddings)
    embeddings_np = image_embeddings.detach().cpu().numpy().astype(np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    # k+1 because the self-match (distance=0) is always returned as the nearest
    knn_matches = matcher.knnMatch(embeddings_np, embeddings_np, k=k + 1)

    matching_dict: dict[tuple[int, int], DistancedTriple] = {}
    for st_idx, matches in enumerate(knn_matches):
        kept = [m for m in matches if m.trainIdx != st_idx and m.distance <= distance_th]
        if max_pairs_per_image is not None:
            kept = kept[:max_pairs_per_image]
        for m in kept:
            nd_idx = m.trainIdx
            st, nd = sorted((st_idx, nd_idx))
            key = (st, nd)
            if key not in matching_dict or m.distance < matching_dict[key].distance:
                matching_dict[key] = DistancedTriple(st, nd, m.distance)

    matching_list = sorted(matching_dict.values(), key=lambda x: x.distance)
    if max_pairs is not None:
        matching_list = matching_list[:max_pairs]
    return matching_list
