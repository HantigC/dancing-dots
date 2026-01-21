from __future__ import annotations

import logging

import torch
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.processor import Retriever

from mts.core.model.mast3r.io import load_model
from mts.core.types import PairType, PathLike
from mts.pipeline.repository.inmemeory import ImageId, ImageRepository
from mts.pipeline.step.base import BasePipelineStep, use_image_repository

LOGGER = logging.getLogger(__name__)


class Mast3rParer(BasePipelineStep):
    def __init__(
        self,
        retriever: Retriever,
        scene_graph: str,
    ) -> None:
        super().__init__()
        self.retriever = retriever
        self.scene_graph = scene_graph

    @use_image_repository
    def run(self, image_repository: ImageRepository) -> None:
        LOGGER.info("Compute pairs...")
        pairs = self._compute_pairs(image_repository)
        LOGGER.info("Computed %d pairs...", len(pairs))
        image_repository.add_pairs(pairs)

    @property
    def device(self):
        return self.retriever.device

    def to(self, device=None, **kwargs):
        self.retriever.model.to(device, **kwargs)
        self.retriever.device = device

    def _compute_pairs(
        self, image_repository: ImageRepository
    ) -> list[PairType[ImageId]]:
        images_filepaths = [str(image_filepath) for image_filepath in image_repository.image_filepaths()]
        with torch.no_grad():
            similarity_matrix_np = self.retriever(images_filepaths)

        raw_pairs = make_pairs(
            images_filepaths,
            self.scene_graph,
            prefilter=None,
            symmetrize=True,
            sim_mat=similarity_matrix_np,
        )

        pairs = list(
            set(
                (
                    image_repository.get_image_id(st_image_filepath),
                    image_repository.get_image_id(nd_image_filepath),
                )
                for st_image_filepath, nd_image_filepath in raw_pairs
            )
        )
        return pairs

    @classmethod
    def from_checkpoints(
        cls,
        model_checkpoint: PathLike,
        retrieval_checkpoint: PathLike,
        scene_graph: str, 
    ) -> Mast3rParer:
        mast3r_model = load_model(model_checkpoint, torch.device("cpu"))
        retriever = Retriever(
            retrieval_checkpoint,
            backbone=mast3r_model,
            device=torch.device("cpu"),
        )
        return cls(retriever, scene_graph)
