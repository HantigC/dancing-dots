import itertools as it
import logging
from typing import Any

from mts.core.pair.cross import compute_knn_pairs
from mts.core.scene_graph.nx import mst_pair_distanced_triple
from mts.core.types import ImageId, PairType
from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.base import BasePipelineStep
from mts.pipeline.step.pair.common import extract_possible_pairs

LOGGER = logging.getLogger(__name__)


class KnnEmbeddingParerStep(BasePipelineStep):
    def __init__(
        self,
        min_images: int = 20,
        k: int = 20,
        min_distance_th: float = 0.0,
        distance_th: float = 1000,
        max_pairs_per_image: int | None = None,
        max_pairs: int | None = None,
    ) -> None:
        super().__init__()
        self.min_images = min_images
        self.k = k
        self.min_distance_th = min_distance_th
        self.distance_th = distance_th
        self.max_pairs_per_image = max_pairs_per_image
        self.max_pairs = max_pairs

    def run(
        self,
        *,
        image_repository: BaseImageRepository,
        input: Any = None,
        state: dict[str, Any] = None,
    ) -> None:
        LOGGER.info("Compute pairs...")
        image_ids = list(image_repository.image_ids())

        if len(image_ids) < self.min_images:
            pairs = list(it.combinations(image_ids, 2))
            image_repository.add_pairs(pairs)
            state["starting_pairs"] = pairs
            return

        image_embeddings = [
            image_repository.get_global_descriptor(image_id)
            for image_id in image_ids
        ]
        triples = compute_knn_pairs(
            image_embeddings,
            self.k,
            self.distance_th,
            max_pairs_per_image=self.max_pairs_per_image,
            max_pairs=self.max_pairs,
        )
        triples = [t for t in triples if t.distance >= self.min_distance_th]

        possible_pairs = extract_possible_pairs(triples, image_ids)
        LOGGER.info("Computed %d pairs", len(possible_pairs))
        image_repository.add_pairs(possible_pairs)

        mst = mst_pair_distanced_triple(triples, image_ids, self.distance_th)
        mst_pairs = [
            tuple(sorted((st, nd))) for st, nd in mst.edges
        ]
        LOGGER.info("MST starting pairs: %d", len(mst_pairs))
        state["starting_pairs"] = mst_pairs
