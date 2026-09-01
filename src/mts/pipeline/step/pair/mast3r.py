from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np
import torch
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.processor import Retriever

from mts.core.model.mast3r.io import load_model
from mts.core.scene_graph.nx import mst_from_distance_matrix
from mts.core.types import PairType, PathLike
from mts.pipeline.repository.base import BaseImageRepository, SceneScopedImageRepository
from mts.pipeline.repository.inmemeory import ImageId, ImageRepository
from mts.pipeline.step.base import PerSceneStep
from mts.utils.pair import from_distance_matrix

LOGGER = logging.getLogger(__name__)


class Mast3rParer(PerSceneStep):
    def __init__(
        self,
        retriever: Retriever,
        scene_graph: str,
    ) -> None:
        super().__init__()
        self.retriever = retriever
        self.scene_graph = scene_graph

    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> None:
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
        images_filepaths = [
            str(image_filepath) for image_filepath in image_repository.image_filepaths()
        ]
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


class MstPairTriple(NamedTuple):
    mst_pairs: list[PairType[int]]
    possible_pairs: list[PairType[int]]
    filepaths_to_id_map: dict[str, int]


class Mast3rDistanceParer(PerSceneStep):
    def __init__(
        self,
        retriever: Retriever,
    ) -> None:
        super().__init__()
        self.retriever = retriever

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
        image_repository.store("starting_pairs", pairs.mst_pairs)

        LOGGER.info("Add starting pairs to state...")
        scene_state["starting_pairs"] = pairs.mst_pairs

    @property
    def device(self):
        return self.retriever.device

    def to(self, device=None, **kwargs):
        self.retriever.model.to(device, **kwargs)
        self.retriever.device = device

    def _compute_distance_matrix(self, filepaths_as_str: list[str]) -> np.ndarray:
        LOGGER.info("Computing distance matrix...")

        with torch.no_grad():
            similarity_matrix = self.retriever(filepaths_as_str)

            distance_matrix = 1 - similarity_matrix
            distance_matrix = (distance_matrix - distance_matrix.min()) / (
                distance_matrix.max() - distance_matrix.min()
            )
        return distance_matrix

    def _extract_starting_pairs(
        self,
        distance_matrix: np.ndarray,
        filepaths_as_str_to_ids_map: dict[str, int],
    ) -> list[PairType[int]]:
        mst = mst_from_distance_matrix(
            distance_matrix,
            list(filepaths_as_str_to_ids_map),
        )

        mst_pairs = []
        for st_node, nd_node in mst.edges:
            st_image_id, nd_image_id = sorted(
                (
                    filepaths_as_str_to_ids_map[st_node],
                    filepaths_as_str_to_ids_map[nd_node],
                )
            )
            mst_pairs.append((st_image_id, nd_image_id))
        return mst_pairs

    def _extract_possible_pairs(
        self,
        distance_matrix: np.ndarray,
        filepaths_as_str_to_ids_map: dict[str, int],
        filepaths_as_str: list[str],
    ) -> list[PairType[int]]:
        possible_pairs = []
        for st_idx, nd_idx in from_distance_matrix(distance_matrix):
            st_filepath_str = filepaths_as_str[st_idx]
            nd_filepath_str = filepaths_as_str[nd_idx]
            st_image_id, nd_image_id = (
                filepaths_as_str_to_ids_map[st_filepath_str],
                filepaths_as_str_to_ids_map[nd_filepath_str],
            )
            possible_pairs.append((st_image_id, nd_image_id))
        return possible_pairs

    def _compute_pairs(self, image_repository: BaseImageRepository) -> MstPairTriple:
        filepaths_as_str_to_ids_map = {
            str(image_repository.get_filepath(image_id)): image_id
            for image_id in image_repository.image_ids()
        }

        filepaths_as_str = list(filepaths_as_str_to_ids_map)

        distance_matrix = self._compute_distance_matrix(filepaths_as_str)
        mst_pairs = self._extract_starting_pairs(
            distance_matrix,
            filepaths_as_str_to_ids_map,
        )
        possible_pairs = self._extract_possible_pairs(
            distance_matrix,
            filepaths_as_str_to_ids_map,
            filepaths_as_str,
        )

        return MstPairTriple(
            mst_pairs,
            possible_pairs,
            filepaths_as_str_to_ids_map,
        )

    @classmethod
    def from_checkpoints(
        cls,
        model_checkpoint: PathLike,
        retrieval_checkpoint: PathLike,
    ) -> Mast3rParer:
        mast3r_model = load_model(model_checkpoint, torch.device("cpu"))
        retriever = Retriever(
            retrieval_checkpoint,
            backbone=mast3r_model,
            device=torch.device("cpu"),
        )
        return cls(retriever)
