import logging
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from mts.core.embedder.base import BaseEmbedder, extract_embeddings_from_images
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.pair.cross import compute_knn_pairs
from mts.core.scene_graph.nx import mst_pair_distanced_triple
from mts.core.types import DistancedTriple, ImageId, PairType
from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.base import BasePipelineStep
from mts.pipeline.step.pair.common import extract_possible_pairs
from mts.pipeline.step.pair.mast3r import MstPairTriple
from mts.utils.torchx import to_torch_format

LOGGER = logging.getLogger(__name__)


def extract_cv_features(
    images,
    detector: cv2.Feature2D,
    tqdm_kwargs: dict | None = None,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    tqdm_kwargs = tqdm_kwargs or {}
    results = []
    for image in tqdm(images, **tqdm_kwargs):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        kpts = (
            np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
            if keypoints
            else np.empty((0, 2), dtype=np.float32)
        )
        results.append((kpts, descriptors))
    return results


def match_descriptors_cv(
    descriptors1: np.ndarray | None,
    descriptors2: np.ndarray | None,
    norm_type: int,
    ratio_threshold: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        descriptors1 is None
        or descriptors2 is None
        or len(descriptors1) < 2
        or len(descriptors2) < 2
    ):
        return np.empty(0, dtype=np.float32), np.empty((0, 2), dtype=np.int64)
    matcher = cv2.BFMatcher(norm_type)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good = [m for m, n in knn_matches if m.distance < ratio_threshold * n.distance]
    if not good:
        return np.empty(0, dtype=np.float32), np.empty((0, 2), dtype=np.int64)
    dists = np.array([m.distance for m in good], dtype=np.float32)
    idxs = np.array([[m.queryIdx, m.trainIdx] for m in good], dtype=np.int64)
    return dists, idxs


def filter_validated_pairs_cv(
    pairs: list[DistancedTriple],
    kpts_descriptors: list[tuple[np.ndarray, np.ndarray | None]],
    images_sizes: dict[int, tuple[int, int]],
    norm_type: int,
    descriptor_name: str = "cv",
    ratio_threshold: float = 0.75,
    min_inliers: int = 30,
) -> tuple[list[DistancedTriple], dict[tuple[int, int], np.ndarray]]:
    filtered_pairs = []
    validated_matches: dict[tuple[int, int], np.ndarray] = {}
    for triple in tqdm(pairs, desc=f"Validate pairs with {descriptor_name}"):
        st_idx, nd_idx = triple.st, triple.nd
        kpts1, descriptors1 = kpts_descriptors[st_idx]
        kpts2, descriptors2 = kpts_descriptors[nd_idx]
        _, matches = match_descriptors_cv(
            descriptors1,
            descriptors2,
            norm_type,
            ratio_threshold,
        )
        if len(matches) == 0:
            continue
        inlier_matches = validate_kps_matches(
            kpts1[matches[:, 0]],
            kpts2[matches[:, 1]],
            images_sizes[st_idx],
            images_sizes[nd_idx],
        )
        if len(inlier_matches) > min_inliers:
            filtered_pairs.append(DistancedTriple(st_idx, nd_idx, triple.distance))
            validated_matches[(st_idx, nd_idx)] = matches
    LOGGER.info(
        "Validated pairs: %d / %d passed %s inlier filter",
        len(filtered_pairs),
        len(pairs),
        descriptor_name,
    )
    return filtered_pairs, validated_matches


class CVDistanceParer(BasePipelineStep):
    def __init__(
        self,
        global_descriptor: BaseEmbedder,
        detector: cv2.Feature2D,
        norm_type: int,
        descriptor_name: str,
        upper_threshold: float = 1.01,
        k: int = 20,
        distance_th: float = 1000,
        max_pairs_per_image: int | None = None,
        max_pairs: int | None = None,
        top_n: int | None = None,
        ratio_threshold: float = 0.75,
        min_inliers: int = 30,
        save_keypoints: bool = False,
        save_descriptors: bool = False,
        save_matches: bool = False,
        **_,
    ) -> None:
        super().__init__()
        self.global_descriptor = global_descriptor
        self.detector = detector
        self.norm_type = norm_type
        self.descriptor_name = descriptor_name
        self.upper_threshold = upper_threshold
        self.k = k
        self.distance_th = distance_th
        self.max_pairs_per_image = max_pairs_per_image
        self.max_pairs = max_pairs
        self.top_n = top_n
        self.ratio_threshold = ratio_threshold
        self.min_inliers = min_inliers
        self.save_keypoints = save_keypoints
        self.save_descriptors = save_descriptors
        self.save_matches = save_matches

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
        pairs = compute_knn_pairs(
            embeddings,
            k=self.k,
            distance_th=self.distance_th,
            max_pairs_per_image=self.max_pairs_per_image,
            max_pairs=self.max_pairs,
        )
        LOGGER.info("Cross-pairs: %d candidate pairs", len(pairs))
        return pairs

    def _extract_cv_features(
        self,
        image_repository: BaseImageRepository,
        images_ids: list[ImageId],
        pairs: list[DistancedTriple],
    ) -> list[DistancedTriple]:
        LOGGER.info(
            "Extracting %s features for %d images...",
            self.descriptor_name,
            len(images_ids),
        )
        kpts_descriptors = extract_cv_features(
            (image_repository.load_image(image_id) for image_id in images_ids),
            self.detector,
            tqdm_kwargs={"desc": f"Extract {self.descriptor_name} features"},
        )
        name = self.descriptor_name.lower()
        if self.save_keypoints or self.save_descriptors:
            for image_id, (kpts, descriptors) in zip(images_ids, kpts_descriptors):
                if self.save_keypoints:
                    image_repository.add_keypoints(image_id, kpts, name=name)
                if self.save_descriptors:
                    image_repository.add_descriptors(image_id, descriptors, name=name)

        images_sizes = {
            num: image_repository.load_image(image_id).shape[:2]
            for num, image_id in enumerate(images_ids)
        }
        filtered_pairs, validated_matches = filter_validated_pairs_cv(
            pairs,
            kpts_descriptors,
            images_sizes,
            norm_type=self.norm_type,
            descriptor_name=self.descriptor_name,
            ratio_threshold=self.ratio_threshold,
            min_inliers=self.min_inliers,
        )
        if self.save_matches:
            for (st_idx, nd_idx), matches in validated_matches.items():
                image_repository.add_matches(
                    images_ids[st_idx], images_ids[nd_idx], matches, name=name
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
            distance_triples, image_ids, self.upper_threshold
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
        filtered_triples = self._extract_cv_features(
            image_repository, image_ids, initial_triples
        )
        if self.top_n is not None:
            filtered_triples = sorted(filtered_triples, key=lambda t: t.distance)[
                : self.top_n
            ]
            LOGGER.info("top_n=%d: kept %d pairs", self.top_n, len(filtered_triples))
        mst_pairs = self._extract_starting_pairs(filtered_triples, num_to_ids_map)
        possible_pairs = extract_possible_pairs(filtered_triples, image_ids)
        LOGGER.info(
            "Pairs computed: %d MST pairs, %d possible pairs",
            len(mst_pairs),
            len(possible_pairs),
        )
        return MstPairTriple(mst_pairs, possible_pairs, filepaths_as_str_to_ids_map)


def make_sift_distance_parer(
    global_descriptor: BaseEmbedder, num_features: int = 2048, **kwargs
) -> CVDistanceParer:
    return CVDistanceParer(
        global_descriptor=global_descriptor,
        detector=cv2.SIFT_create(nfeatures=num_features),
        norm_type=cv2.NORM_L2,
        descriptor_name="SIFT",
        **kwargs,
    )


def make_orb_distance_parer(
    global_descriptor: BaseEmbedder, num_features: int = 2048, **kwargs
) -> CVDistanceParer:
    return CVDistanceParer(
        global_descriptor=global_descriptor,
        detector=cv2.ORB_create(nfeatures=num_features),
        norm_type=cv2.NORM_HAMMING,
        descriptor_name="ORB",
        **kwargs,
    )


def make_brisk_distance_parer(
    global_descriptor: BaseEmbedder,
    thresh: int = 30,
    octaves: int = 3,
    **kwargs,
) -> CVDistanceParer:
    return CVDistanceParer(
        global_descriptor=global_descriptor,
        detector=cv2.BRISK_create(thresh=thresh, octaves=octaves),
        norm_type=cv2.NORM_HAMMING,
        descriptor_name="BRISK",
        **kwargs,
    )


def make_akaze_distance_parer(
    global_descriptor: BaseEmbedder,
    threshold: float = 0.001,
    n_octaves: int = 4,
    **kwargs,
) -> CVDistanceParer:
    return CVDistanceParer(
        global_descriptor=global_descriptor,
        detector=cv2.AKAZE_create(threshold=threshold, nOctaves=n_octaves),
        norm_type=cv2.NORM_HAMMING,
        descriptor_name="AKAZE",
        **kwargs,
    )


def make_kaze_distance_parer(
    global_descriptor: BaseEmbedder,
    threshold: float = 0.001,
    n_octaves: int = 4,
    **kwargs,
) -> CVDistanceParer:
    return CVDistanceParer(
        global_descriptor=global_descriptor,
        detector=cv2.KAZE_create(threshold=threshold, nOctaves=n_octaves),
        norm_type=cv2.NORM_L2,
        descriptor_name="KAZE",
        **kwargs,
    )


class FastWithDescriptor:
    # FAST detects only; pair with an external descriptor extractor
    def __init__(self, fast: cv2.FastFeatureDetector, descriptor: cv2.Feature2D):
        self._fast = fast
        self._descriptor = descriptor

    def detectAndCompute(self, image, mask):
        keypoints = self._fast.detect(image, mask)
        return self._descriptor.compute(image, keypoints)


def make_fast_distance_parer(
    global_descriptor: BaseEmbedder,
    threshold: int = 10,
    nonmax_suppression: bool = True,
    num_features: int = 2048,
    **kwargs,
) -> CVDistanceParer:
    fast = cv2.FastFeatureDetector_create(
        threshold=threshold, nonmaxSuppression=nonmax_suppression
    )
    orb = cv2.ORB_create(nfeatures=num_features)
    return CVDistanceParer(
        global_descriptor=global_descriptor,
        detector=FastWithDescriptor(fast, orb),
        norm_type=cv2.NORM_HAMMING,
        descriptor_name="FAST",
        **kwargs,
    )
