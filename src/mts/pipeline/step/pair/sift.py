import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from mts.core.embedder.base import extract_embeddings_from_images
from mts.core.embedder.dinov2 import DinoV2GlobalDescriptors
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.pair.cross import compute_cross_pairs
from mts.core.scene_graph.nx import mst_pair_distanced_triple
from mts.core.types import DistancedTriple, ImageId, PairType, PathLike
from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.base import BasePipelineStep
from mts.pipeline.step.pair.mast3r import MstPairTriple

LOGGER = logging.getLogger(__name__)


def extract_sift_features(
    images: list[tuple[np.ndarray, Path | str]],
    num_features: int = 0,
    tqdm_kwargs: dict | None = None,
) -> list[tuple[PathLike, np.ndarray, np.ndarray]]:
    tqdm_kwargs = tqdm_kwargs or {}
    sift = cv2.SIFT_create(nfeatures=num_features)
    results = []
    for image in tqdm(images, **tqdm_kwargs):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        results.append((kpts, descriptors))
    return results


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


def filter_validated_paris(
    pairs: list[DistancedTriple],
    kpts_descriptors: list[tuple[np.ndarray, np.ndarray]],
    images_sizes: dict[int, tuple[int, int]],
) -> list[DistancedTriple]:
    filtered_pairs = []
    for triple in tqdm(pairs):
        st_idx, nd_idx = triple.st, triple.nd
        kpts1, descriptors1 = kpts_descriptors[st_idx]
        kpts2, descriptors2 = kpts_descriptors[nd_idx]
        dist, matches = match_descriptors(
            descriptors1=descriptors1,
            descriptors2=descriptors2,
            ratio_threshold=0.8,
        )
        inlier_matches = validate_kps_matches(
            kpts1[matches[:, 0]],
            kpts2[matches[:, 1]],
            images_sizes[st_idx],
            images_sizes[nd_idx],
        )
        if len(inlier_matches) > 30:
            filtered_pairs.append(DistancedTriple(st_idx, nd_idx, triple.distance))
    return filtered_pairs


class SiftDistanceParer(BasePipelineStep):
    def __init__(
        self,
        dinov2_global_descriptor: DinoV2GlobalDescriptors,
        upper_threshold: float = 1.01,
    ) -> None:
        super().__init__()
        self.dinov2_global_descriptor = dinov2_global_descriptor
        self.upper_threshold = upper_threshold

    def run(
        self,
        *,
        image_repository: BaseImageRepository,
        input: Any = None,
        state: dict[str, Any] = None,
    ) -> Any:
        LOGGER.info("Compute pairs...")
        pairs = self._compute_pairs(image_repository)
        LOGGER.info("Write possible pairs to repository...")
        image_repository.add_pairs(pairs.possible_pairs)

        LOGGER.info("Write starting pairs to repository...")
        image_repository.store("starting_pairs", pairs.possible_pairs)

        LOGGER.info("Add starting pairs to state...")
        state["starting_pairs"] = pairs.mst_pairs

    def _extract_initial_triples(
        self,
        image_repository: BaseImageRepository,
        images_ids: list[ImageId],
    ) -> list[DistancedTriple]:

        dinov2_embeddings = extract_embeddings_from_images(
            self.dinov2_global_descriptor,
            (image_repository.load_image(image_id) for image_id in images_ids),
        )

        pairs = compute_cross_pairs(
            dinov2_embeddings,
            cutoff_th=0.25,
            distance_th=2,
            min_pairs=1,
        )
        return pairs

    def _extract_sift_features(
        self,
        image_repository: BaseImageRepository,
        images_ids: list[ImageId],
        pairs: list[DistancedTriple],
    ) -> list[DistancedTriple]:
        kpts_descriptors = extract_sift_features(
            (image_repository.load_image(image_id) for image_id in images_ids),
            2024,
        )
        images_sizes = {
            num: image_repository.load_image(image_id).shape[:2]
            for num, image_id in enumerate(images_ids)
        }
        filtered_pairs = filter_validated_paris(pairs, kpts_descriptors, images_sizes)
        return filtered_pairs

    @property
    def device(self):
        return self.dinov2_global_descriptor.device

    def to(self, device=None, **kwargs):
        self.dinov2_global_descriptor.to(device=device, **kwargs)

    def _extract_starting_pairs(
        self,
        distance_triples: list[DistancedTriple],
        image_ids: list[int],
    ) -> list[PairType[int]]:
        mst = mst_pair_distanced_triple(
            distance_triples,
            image_ids,
            self.upper_threshold,
        )
        mst_pairs = []
        for st_image_id, nd_image_id in mst.edges:
            st_image_id, nd_image_id = sorted((st_image_id, nd_image_id))
            mst_pairs.append((st_image_id, nd_image_id))
        return mst_pairs

    def _extract_possible_pairs(
        self,
        filtered_triples: list[DistancedTriple],
        image_ids: list[int],
    ) -> list[PairType[int]]:
        possible_pairs = []
        for distance_triple in filtered_triples:
            st_idx, nd_idx = distance_triple.st, distance_triple.nd
            st_image_id = image_ids[st_idx]
            nd_image_id = image_ids[nd_idx]
            st_image_id, nd_image_id = sorted((st_image_id, nd_image_id))
            possible_pairs.append((st_image_id, nd_image_id))
        return possible_pairs

    def _compute_pairs(self, image_repository: BaseImageRepository) -> MstPairTriple:
        filepaths_as_str_to_ids_map = {
            str(image_repository.get_filepath(image_id)): image_id
            for image_id in image_repository.image_ids()
        }
        num_to_ids_map = {
            num: image_id for num, image_id in enumerate(image_repository.image_ids())
        }

        filepaths_as_str = list(filepaths_as_str_to_ids_map)
        image_ids = [
            filepaths_as_str_to_ids_map[filepath_str]
            for filepath_str in filepaths_as_str
        ]
        initial_triples = self._extract_initial_triples(image_repository, image_ids)
        filtered_triples = self._extract_sift_features(
            image_repository,
            image_ids,
            initial_triples,
        )

        mst_pairs = self._extract_starting_pairs(
            filtered_triples,
            num_to_ids_map,
        )
        possible_pairs = self._extract_possible_pairs(
            filtered_triples,
            image_ids,
        )

        return MstPairTriple(
            mst_pairs,
            possible_pairs,
            filepaths_as_str_to_ids_map,
        )
