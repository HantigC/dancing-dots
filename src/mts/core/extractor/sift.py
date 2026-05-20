from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm.auto import tqdm

from mts.core.extractor.base import BaseExtractor


class SiftExtractor(BaseExtractor):
    def __init__(self, num_features) -> None:
        super().__init__()
        self.sift = cv2.SIFT_create(nfeatures=num_features)

    def extract(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_np = image.squeeze().cpu().numpy()
        if img_np.ndim == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np = img_np.astype(np.uint8)
        if img_np.ndim == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        kpts = torch.tensor(
            [[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=torch.float32
        )
        descs = torch.tensor(descriptors, dtype=torch.float32)

        return kpts, descs


def extract_sift_features(
    images: list[tuple[np.ndarray, Path | str]],
    num_features: int = 0,
) -> list[tuple[Path | str, np.ndarray, np.ndarray]]:
    sift = cv2.SIFT_create(nfeatures=num_features)
    results = []
    for image, filepath in tqdm(images):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        results.append((filepath, kpts, descriptors))
    return results
