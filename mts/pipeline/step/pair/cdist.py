import itertools as it
import logging
from typing import Sequence

from mts.core.pair.cross import compute_cross_pairs
from mts.core.types import ImageId, PairType
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import BasePipelineStep

LOGGER = logging.getLogger(__name__)


class CrossEmbeddingParer(BasePipelineStep):
    def __init__(
        self,
        image_repository: ImageRepository,
        min_images: int = 20,
        cutoff_th: float = 0.6,
        distance_th: float = 1000,
        min_pairs: int = 20,
    ) -> None:
        self.image_repository = image_repository
        self.min_images = min_images
        self.cutoff_th = cutoff_th
        self.distance_th = distance_th
        self.min_pairs = min_pairs

    def run(self) -> None:
        LOGGER.info("Compute pairs...")
        pairs = self._compute_pairs(self.image_repository.image_ids())
        LOGGER.info("Computed %d pairs...", len(pairs))
        self.image_repository.add_pairs(pairs)

    def _compute_pairs(self, image_ids: Sequence[ImageId]) -> list[PairType[ImageId]]:
        if self.image_repository.images_num() < self.min_images:
            return list(it.combinations(image_ids, 2))
        image_embeddings = [
            self.image_repository.get_global_descriptor(image_id)
            for image_id in image_ids
        ]
        pairs = compute_cross_pairs(
            image_embeddings,
            self.cutoff_th,
            self.distance_th,
            self.min_pairs,
        )
        return pairs
