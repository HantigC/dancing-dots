import itertools as it
import logging
from typing import Any

from mts.core.pair.cross import compute_cross_pairs
from mts.core.types import ImageId, PairType
from mts.pipeline.repository.base import BaseImageRepository, SceneScopedImageRepository
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import PerSceneStep
from mts.pipeline.step.pair.common import extract_possible_pairs

LOGGER = logging.getLogger(__name__)


class CrossEmbeddingParerStep(PerSceneStep):
    def __init__(
        self,
        min_images: int = 20,
        cutoff_th: float = 0.6,
        distance_th: float = 1000,
        min_pairs: int = 20,
        max_pairs_per_image: int | None = None,
        max_pairs: int | None = None,
    ) -> None:
        super().__init__()
        self.min_images = min_images
        self.cutoff_th = cutoff_th
        self.distance_th = distance_th
        self.min_pairs = min_pairs
        self.max_pairs_per_image = max_pairs_per_image
        self.max_pairs = max_pairs

    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> None:
        LOGGER.info("Compute pairs for scene '%s'...", scene)
        pairs = self._compute_pairs(image_repository, scene)
        LOGGER.info("Computed %d pairs for scene '%s'", len(pairs), scene)
        # every pair is within `scene`; add_pairs raises if that is ever
        # violated, so cross-scene pairs can never be persisted here.
        image_repository.add_pairs(pairs)

    def _compute_pairs(
        self, image_repository: BaseImageRepository, scene: str
    ) -> list[PairType[ImageId]]:
        image_ids = list(image_repository.image_ids(scene=scene))
        if len(image_ids) < self.min_images:
            return list(it.combinations(image_ids, 2))
        image_embeddings = [
            image_repository.get_global_descriptor(
                image_id,
            )
            for image_id in image_ids
        ]
        pairs = compute_cross_pairs(
            image_embeddings,
            self.cutoff_th,
            self.distance_th,
            self.min_pairs,
            max_pairs_per_image=self.max_pairs_per_image,
            max_pairs=self.max_pairs,
        )
        possible_pairs = extract_possible_pairs(pairs, image_ids)
        return possible_pairs
