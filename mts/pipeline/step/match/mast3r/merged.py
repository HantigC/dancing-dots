from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from mast3r.model import AsymmetricMASt3R

from mts.core.matching.dense.mast3r import extract_dense_keypoints
from mts.core.matching.dense.merge.round import merge_matches
from mts.core.model.mast3r.io import load_model
from mts.core.scene_graph.model import Image, MatchKind, TwoViewEdge
from mts.core.scene_graph.nx import extract_matches
from mts.core.scene_graph.transient import grow_from_pairs
from mts.core.types import ImageId, PairType, PathLike
from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.extract.kp.base import BasePipelineStep

LOGGER = logging.getLogger(__name__)


class Mast3rMatchPipelineStep(BasePipelineStep):
    def __init__(
        self,
        mast3r_model: AsymmetricMASt3R,
        min_pairs: int = 15,
        match_conf_th: float = 1.001,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.mast3r_model = mast3r_model
        self.min_pairs = min_pairs
        self.match_conf_th = match_conf_th
        self.verbose = verbose

    def run(
        self,
        *,
        image_repository: BaseImageRepository,
        input: Any = None,
        state: dict[str, Any] = None,
    ) -> Any:
        starting_pairs: list[PairType[ImageId]] = state["starting_pairs"]
        keypoints_map, matches_map = self._compute_matches(
            image_repository,
            starting_pairs,
        )
        self._save_matches_and_kpts(
            keypoints_map,
            matches_map,
            image_repository,
        )

    def _save_matches_and_kpts(
        self,
        keypoints_map,
        matches_map,
        image_repository,
    ) -> None:
        for image_filepath, keypoints in keypoints_map.items():
            image_id = image_repository.get_image_id(Path(image_filepath))
            image_repository.add_keypoints(image_id, keypoints)

        for (st_image_filepath, nd_image_filepath), matches in matches_map.items():
            st_image_id = image_repository.get_image_id(Path(st_image_filepath))
            nd_image_id = image_repository.get_image_id(Path(nd_image_filepath))
            image_repository.add_matches(st_image_id, nd_image_id, matches)

    def _compute_matches(
        self,
        image_repository: BaseImageRepository,
        mst_pairs: list[PairType[ImageId]],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, dict[str, np.ndarray]],
    ]:
        scene_graph = self._init_graph_from_mst(
            image_repository,
            mst_pairs,
        )
        possible_pairs = [
            (
                str(image_repository.get_filepath(st_id)),
                str(image_repository.get_filepath(nd_id)),
            )
            for st_id, nd_id in image_repository.get_pairs()
        ]
        grow_from_pairs(
            scene_graph,
            possible_pairs,
            self._match_two_images,
        )

        matches_dict = extract_matches(scene_graph)

        global_keypoints, global_matches = merge_matches(matches_dict)
        return global_keypoints, global_matches

    def _match_two_images(
        self,
        st_filepath: str,
        nd_filepath: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        matches_map = extract_dense_keypoints(
            self.mast3r_model,
            [(0, 1)],
            [st_filepath, nd_filepath],
            device=self.device,
        )
        try_first = st_filepath
        try_second = nd_filepath

        if len(matches_map) == 0:
            return np.array([]), np.array([])

        try:
            matches_mmap = matches_map[try_first]
        except KeyError:
            matches_mmap = matches_map[try_second]
            try_first, try_second = try_second, try_first

        if len(matches_mmap) == 0:
            return np.array([]), np.array([])

        matches = matches_mmap[try_second]
        st_kpts, nd_kpts = np.split(matches, 2, axis=1)
        return st_kpts, nd_kpts

    def _init_graph_from_mst(
        self,
        image_repository: BaseImageRepository,
        mst_pairs: list[PairType[int]],
    ) -> nx.Graph:
        image_id_to_num = {}
        filepaths_as_str = []
        for num, image_id in enumerate(image_repository.image_ids()):
            image_id_to_num[image_id] = num
            filepaths_as_str.append(str(image_repository.get_filepath(image_id)))

        mst_paris_indices = [
            (image_id_to_num[st_image_id], image_id_to_num[nd_image_id])
            for st_image_id, nd_image_id in mst_pairs
        ]
        matches_map = extract_dense_keypoints(
            self.mast3r_model,
            mst_paris_indices,
            filepaths_as_str,
            device=self.device,
        )

        scene_graph = nx.Graph().to_undirected()
        for image_id, image in image_repository.iterate_over_images():
            height, width = image.shape[:2]
            image = Image(
                height=height,
                width=width,
            )
            scene_graph.add_node(
                image_repository.get_filepath(image_id),
                image=image,
            )

        for st_filepath, matched_filepaths_map in matches_map.items():
            for nd_filepath, kpts in matched_filepaths_map.items():
                st_kpts, nd_kpts = np.split(kpts, 2, axis=1)
                scene_graph.add_edge(
                    st_filepath,
                    nd_filepath,
                    two_view=TwoViewEdge(
                        st_filepath=st_filepath,
                        nd_filepath=nd_filepath,
                        kpts_for={
                            st_filepath: st_kpts,
                            nd_filepath: nd_kpts,
                        },
                        match_kind=MatchKind.MATCHED,
                        num_matches=len(kpts),
                    ),
                    weight=len(kpts),
                )
        return scene_graph

    @classmethod
    def from_checkpoint(
        cls, mast3r_model_checkpoint: PathLike, **kwargs
    ) -> Mast3rMatchPipelineStep:
        mast3r_model = load_model(mast3r_model_checkpoint, torch.device("cpu"))
        return cls(mast3r_model, **kwargs)
