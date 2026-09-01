import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from mts.core.embedder.base import BaseEmbedder, extract_embeddings_from_images
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.pair.cross import compute_cross_pairs
from mts.core.scene_graph.nx import mst_pair_distanced_triple
from mts.core.types import DistancedTriple, ImageId, PairType, PathLike
from mts.pipeline.repository.base import BaseImageRepository, SceneScopedImageRepository
from mts.pipeline.step.base import PerSceneStep
from mts.pipeline.step.pair.common import extract_possible_pairs
from mts.pipeline.step.pair.mast3r import MstPairTriple
from mts.utils.torchx import to_torch_format

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
) -> tuple[list[DistancedTriple], dict[tuple[int, int], np.ndarray]]:
    filtered_pairs = []
    validated_matches: dict[tuple[int, int], np.ndarray] = {}
    for triple in tqdm(pairs, desc="Validate pairs with SIFT"):
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
            validated_matches[(st_idx, nd_idx)] = matches
    LOGGER.info(
        "Validated pairs: %d / %d passed SIFT inlier filter",
        len(filtered_pairs),
        len(pairs),
    )
    return filtered_pairs, validated_matches


class SiftDistanceParer(PerSceneStep):
    def __init__(
        self,
        global_descriptor: BaseEmbedder,
        upper_threshold: float = 1.01,
        cutoff_th: float = 0.25,
        distance_th: float = 2.0,
        min_pairs: int = 1,
        max_pairs_per_image: int | None = None,
        max_pairs: int | None = None,
        top_n: int | None = None,
        save_keypoints: bool = False,
        save_descriptors: bool = False,
        save_matches: bool = False,
    ) -> None:
        super().__init__()
        self.global_descriptor = global_descriptor
        self.upper_threshold = upper_threshold
        self.cutoff_th = cutoff_th
        self.distance_th = distance_th
        self.min_pairs = min_pairs
        self.max_pairs_per_image = max_pairs_per_image
        self.max_pairs = max_pairs
        self.top_n = top_n
        self.save_keypoints = save_keypoints
        self.save_descriptors = save_descriptors
        self.save_matches = save_matches

    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> Any:
        LOGGER.info("Compute pairs...")
        pairs = self._compute_pairs(image_repository)
        LOGGER.info("Write possible pairs to repository...")
        image_repository.add_pairs(pairs.possible_pairs)

        LOGGER.info("Write starting pairs to repository...")
        image_repository.store("starting_pairs", pairs.possible_pairs)

        LOGGER.info("Add starting pairs to state...")
        scene_state["starting_pairs"] = pairs.mst_pairs

    def _extract_initial_triples(
        self,
        image_repository: BaseImageRepository,
        images_ids: list[ImageId],
    ) -> list[DistancedTriple]:
        LOGGER.info(
            "Extracting global embeddings '%s' for %d images...",
            self.global_descriptor.__class__.__name__,
            len(images_ids),
        )
        embeddings = extract_embeddings_from_images(
            self.global_descriptor,
            (
                to_torch_format(image_repository.load_image(image_id))
                for image_id in images_ids
            ),
            tqdm_kwargs=dict(total=len(list(image_repository.image_ids()))),
        )

        pairs = compute_cross_pairs(
            embeddings,
            cutoff_th=self.cutoff_th,
            distance_th=self.distance_th,
            min_pairs=self.min_pairs,
            max_pairs_per_image=self.max_pairs_per_image,
            max_pairs=self.max_pairs,
        )
        LOGGER.info("Cross-pairs: %d candidate pairs", len(pairs))
        return pairs

    def _extract_sift_features(
        self,
        image_repository: BaseImageRepository,
        images_ids: list[ImageId],
        pairs: list[DistancedTriple],
    ) -> list[DistancedTriple]:
        LOGGER.info("Extracting SIFT features for %d images...", len(images_ids))
        kpts_descriptors = extract_sift_features(
            (image_repository.load_image(image_id) for image_id in images_ids),
            2024,
            tqdm_kwargs={"desc": "Extract SIFT features"},
        )
        if self.save_keypoints or self.save_descriptors:
            for image_id, (kpts, descriptors) in zip(images_ids, kpts_descriptors):
                if self.save_keypoints:
                    image_repository.add_keypoints(image_id, kpts, name="sift")
                if self.save_descriptors:
                    image_repository.add_descriptors(image_id, descriptors, name="sift")

        images_sizes = {
            num: image_repository.load_image(image_id).shape[:2]
            for num, image_id in enumerate(images_ids)
        }
        filtered_pairs, validated_matches = filter_validated_paris(
            pairs, kpts_descriptors, images_sizes
        )
        if self.save_matches:
            for (st_idx, nd_idx), matches in validated_matches.items():
                st_image_id = images_ids[st_idx]
                nd_image_id = images_ids[nd_idx]
                image_repository.add_matches(
                    st_image_id, nd_image_id, matches, name="sift"
                )

        return filtered_pairs

    @property
    def device(self):
        return self.global_descriptor.device

    def to(self, device=None, **kwargs):
        self.global_descriptor.to(device=device, **kwargs)

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
        if self.top_n is not None:
            filtered_triples = sorted(filtered_triples, key=lambda t: t.distance)[
                : self.top_n
            ]
            LOGGER.info("top_n=%d: kept %d pairs", self.top_n, len(filtered_triples))

        mst_pairs = self._extract_starting_pairs(
            filtered_triples,
            num_to_ids_map,
        )
        possible_pairs = extract_possible_pairs(
            filtered_triples,
            image_ids,
        )
        LOGGER.info(
            "Pairs computed: %d MST pairs, %d possible pairs",
            len(mst_pairs),
            len(possible_pairs),
        )
        return MstPairTriple(
            mst_pairs,
            possible_pairs,
            filepaths_as_str_to_ids_map,
        )
