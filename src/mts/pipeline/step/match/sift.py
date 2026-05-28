import logging
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.base import BasePipelineStep
from mts.pipeline.step.pair.sift import extract_sift_features, match_descriptors

LOGGER = logging.getLogger(__name__)


class SiftKeypointMatchStep(BasePipelineStep):
    def __init__(
        self,
        num_features: int = 2048,
        ratio_threshold: float = 0.75,
        min_matches: int = 30,
        reuse: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.ratio_threshold = ratio_threshold
        self.min_matches = min_matches
        self.reuse = reuse

    def run(
        self,
        *,
        image_repository: BaseImageRepository,
        input: Any = None,
        state: dict[str, Any] = None,
    ) -> Any:
        self._extract_keypoints(image_repository)
        self._match_pairs(image_repository)

    def _pending_keypoint_ids(self, image_repository: BaseImageRepository) -> list:
        all_ids = list(image_repository.image_ids())
        if not self.reuse:
            return all_ids
        return [
            img_id for img_id in all_ids
            if image_repository.get_keypoints(img_id, name="sift") is None
        ]

    def _extract_keypoints(self, image_repository: BaseImageRepository) -> None:
        LOGGER.info("Extracting SIFT keypoints and descriptors...")
        all_ids = list(image_repository.image_ids())
        image_ids = self._pending_keypoint_ids(image_repository)
        if not image_ids:
            LOGGER.info("All keypoints already computed, skipping.")
            return
        LOGGER.info("Computing keypoints for %d / %d images...", len(image_ids), len(all_ids))
        images = (image_repository.load_image(image_id) for image_id in image_ids)
        kpts_descriptors = extract_sift_features(
            images,
            num_features=self.num_features,
            tqdm_kwargs={"desc": "Extract SIFT features"},
        )
        for image_id, (kpts, descriptors) in zip(image_ids, kpts_descriptors):
            image_repository.add_keypoints(image_id, kpts, name="sift")
            image_repository.add_descriptors(image_id, descriptors, name="sift")

    def _match_pairs(self, image_repository: BaseImageRepository) -> None:
        LOGGER.info("Matching SIFT descriptors across pairs...")
        pairs = image_repository.get_pairs()
        for idx1, idx2 in tqdm(pairs, desc="Match SIFT pairs"):
            if self.reuse and image_repository.get_matches(idx1, idx2, name="sift") is not None:
                continue
            desc1 = image_repository.get_descriptors(idx1, name="sift")
            desc2 = image_repository.get_descriptors(idx2, name="sift")
            if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
                continue
            _dists, idxs = match_descriptors(desc1, desc2, self.ratio_threshold)
            if len(idxs) >= self.min_matches:
                image_repository.add_matches(idx1, idx2, idxs, name="sift")
