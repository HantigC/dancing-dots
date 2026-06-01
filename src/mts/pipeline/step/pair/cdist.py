import itertools as it
import logging

from mts.core.pair.cross import compute_cross_pairs
from mts.core.types import ImageId, PairType
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import BasePipelineStep, use_image_repository
from mts.pipeline.step.pair.common import extract_possible_pairs

LOGGER = logging.getLogger(__name__)


class CrossEmbeddingParerStep(BasePipelineStep):
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

    @use_image_repository
    def run(self, image_repository: ImageRepository) -> None:
        LOGGER.info("Compute pairs...")
        pairs = self._compute_pairs(image_repository)
        LOGGER.info("Computed %d pairs...", len(pairs))
        image_repository.add_pairs(pairs)

    def _compute_pairs(
        self, image_repository: ImageRepository
    ) -> list[PairType[ImageId]]:
        if image_repository.images_num() < self.min_images:
            return list(it.combinations(image_repository.image_ids(), 2))
        image_ids = list(image_repository.image_ids())
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
