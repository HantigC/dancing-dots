from __future__ import annotations

import cv2
import numpy as np
import torch

from .base import BaseMatcher


class NearestNeighborMatcher(BaseMatcher):
    def __init__(self, ratio_threshold: float = 0.75) -> None:
        self.ratio_threshold = ratio_threshold
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match(
        self,
        st_kp: torch.Tensor,
        nd_kp: torch.Tensor,
        st_descriptors: torch.Tensor,
        nd_descriptors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        st_desc = st_descriptors.cpu().numpy().astype(np.float32)
        nd_desc = nd_descriptors.cpu().numpy().astype(np.float32)

        knn_matches = self.matcher.knnMatch(st_desc, nd_desc, k=2)

        good = [m for m, n in knn_matches if m.distance < self.ratio_threshold * n.distance]

        if len(good) == 0:
            return torch.empty(0), torch.empty(0, 2, dtype=torch.long)

        match_dists = torch.tensor([m.distance for m in good], dtype=torch.float32)
        match_idxs = torch.tensor([[m.queryIdx, m.trainIdx] for m in good], dtype=torch.long)

        return match_dists, match_idxs


def match_descriptors(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    ratio_threshold: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(
        descriptors1.astype(np.float32),
        descriptors2.astype(np.float32),
        k=2,
    )
    good = [m for m, n in knn_matches if m.distance < ratio_threshold * n.distance]
    if len(good) == 0:
        return np.empty(0, dtype=np.float32), np.empty((0, 2), dtype=np.int64)
    dists = np.array([m.distance for m in good], dtype=np.float32)
    idxs = np.array([[m.queryIdx, m.trainIdx] for m in good], dtype=np.int64)
    return dists, idxs
