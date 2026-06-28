from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import torch
from mast3r.model import AsymmetricMASt3R

from mts.core.matching.dense.mast3r import extract_dense_keypoints
from mts.core.matching.dense.merge.round import merge_matches
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.model.mast3r.io import load_model
from mts.core.scene_graph.model import Image, MatchKind, TwoViewEdge
from mts.core.scene_graph.nx import extract_matches
from mts.core.types import ImageId, PairType, PathLike
from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.extract.kp.base import BasePipelineStep

LOGGER = logging.getLogger(__name__)
GrowCallable = Callable[
    [
        nx.Graph,
        list[tuple[str, str]],
        Callable[[str, str], np.ndarray],
    ],
    None,
]


class Mast3rMatchPipelineStep(BasePipelineStep):
    def __init__(
        self,
        mast3r_model: AsymmetricMASt3R,
        grow_graph: GrowCallable,
        verbose: bool = True,
        match_conf_th: float = 0.5,
        pixel_tol: int = 0,
    ) -> None:
        super().__init__()
        self.mast3r_model = mast3r_model
        self.verbose = verbose
        self.grow_graph = grow_graph

        self.match_conf_th = match_conf_th
        self.pixel_tol = pixel_tol

    def run(
        self,
        *,
        image_repository: BaseImageRepository,
        input: Any = None,
        state: dict[str, Any] = None,
    ) -> Any:
        starting_pairs: list[PairType[ImageId]] = state["starting_pairs"]
        keypoints_map, matches_map, match_kind_map = self._compute_matches(
            image_repository,
            starting_pairs,
        )
        self._save_matches_and_kpts(
            keypoints_map,
            matches_map,
            match_kind_map,
            image_repository,
        )

    def _compute_matches(
        self,
        image_repository: BaseImageRepository,
        mst_pairs: list[PairType[ImageId]],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[tuple[str, str], np.ndarray],
        dict[tuple[str, str], MatchKind],
    ]:
        scene_graph = self._create_graph(image_repository, mst_pairs)
        matches_dict, match_kind_map = extract_matches(scene_graph)
        global_keypoints, global_matches = merge_matches(matches_dict)
        return global_keypoints, global_matches, match_kind_map

    def _create_graph(
        self,
        image_repository: BaseImageRepository,
        mst_pairs: list[PairType[ImageId]],
    ) -> nx.Graph:
        scene_graph, filepath_to_hw = self._init_graph_from_mst(
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

        self.grow_graph(
            scene_graph,
            possible_pairs,
            lambda st, nd: self._match_two_images(
                st, nd, filepath_to_hw[st], filepath_to_hw[nd]
            ),
        )
        return scene_graph

    def _save_matches_and_kpts(
        self,
        keypoints_map: dict[str, np.ndarray],
        matches_map: dict[tuple[str, str], np.ndarray],
        match_kind_map: dict[tuple[str, str], MatchKind],
        image_repository: BaseImageRepository,
    ) -> None:
        for image_filepath, keypoints in keypoints_map.items():
            image_id = image_repository.get_image_id(Path(image_filepath))
            image_repository.add_keypoints(image_id, keypoints, name="mast3r")

        for (st_image_filepath, nd_image_filepath), matches in matches_map.items():
            st_image_id = image_repository.get_image_id(Path(st_image_filepath))
            nd_image_id = image_repository.get_image_id(Path(nd_image_filepath))
            image_repository.add_matches(
                st_image_id, nd_image_id, matches, name="mast3r"
            )
            kind = match_kind_map.get((st_image_filepath, nd_image_filepath))
            if kind is not None:
                image_repository.upsert_match_metadata(
                    st_image_id, nd_image_id, match_kind=kind.value
                )

    def _match_two_images(
        self,
        st_filepath: str,
        nd_filepath: str,
        st_hw: tuple[int, int],
        nd_hw: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:

        try:
            matches_map = extract_dense_keypoints(
                self.mast3r_model,
                [(0, 1)],
                [st_filepath, nd_filepath],
                device=self.device,
                tqdm_kwargs=dict(disable=True),
                match_conf_th=self.match_conf_th,
                pixel_tol=self.pixel_tol,
                batch_size=1,
            )
        except Exception:
            LOGGER.exception("Trouble with extracting the dense keypoints")
            matches_map = {}

        try_first = st_filepath
        try_second = nd_filepath

        if len(matches_map) == 0:
            return np.array([]), np.array([])

        try:
            matches_mmap = matches_map[try_first]
        except KeyError:
            matches_mmap = matches_map[try_second]
            try_first, try_second = try_second, try_first
            st_hw, nd_hw = nd_hw, st_hw

        if len(matches_mmap) == 0:
            return np.array([]), np.array([])

        matches = matches_mmap[try_second]
        st_kpts, nd_kpts = np.split(matches, 2, axis=1)

        try:
            inlier_matches = validate_kps_matches(st_kpts, nd_kpts, st_hw, nd_hw)
        except Exception:
            LOGGER.exception(
                "Could not validate matches for %s <-> %s",
                st_filepath,
                nd_filepath,
            )
            inlier_matches = np.empty((0, 2), dtype=np.int32)

        if len(inlier_matches) == 0:
            return np.array([]), np.array([])

        return st_kpts[inlier_matches[:, 0]], nd_kpts[inlier_matches[:, 1]]

    def _init_graph_from_mst(
        self,
        image_repository: BaseImageRepository,
        mst_pairs: list[PairType[int]],
    ) -> tuple[nx.Graph, dict[str, tuple[int, int]]]:
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
            match_conf_th=self.match_conf_th,
            pixel_tol=self.pixel_tol,
            batch_size=1,
        )

        scene_graph = nx.Graph().to_undirected()
        filepath_to_hw = {}
        for image_id, image in image_repository.iterate_over_images():
            height, width = image.shape[:2]
            filepath = image_repository.get_filepath(image_id)
            filepath_to_hw[str(filepath)] = (height, width)
            image = Image(
                height=height,
                width=width,
            )
            scene_graph.add_node(
                filepath,
                image=image,
            )

        for st_filepath, matched_filepaths_map in matches_map.items():
            for nd_filepath, kpts in matched_filepaths_map.items():
                st_kpts, nd_kpts = np.split(kpts, 2, axis=1)
                try:
                    inlier_matches = validate_kps_matches(
                        st_kpts,
                        nd_kpts,
                        filepath_to_hw[st_filepath],
                        filepath_to_hw[nd_filepath],
                    )
                except Exception:
                    LOGGER.exception(
                        "Could not validate matches for %s <-> %s",
                        st_filepath,
                        nd_filepath,
                    )
                    inlier_matches = np.empty((0, 2), dtype=np.int32)

                if len(inlier_matches) == 0:
                    continue

                st_kpts = st_kpts[inlier_matches[:, 0]]
                nd_kpts = nd_kpts[inlier_matches[:, 1]]
                num_inliers = len(inlier_matches)
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
                        num_matches=num_inliers,
                    ),
                    weight=num_inliers,
                    mst=True,
                )
        return scene_graph, filepath_to_hw

    @classmethod
    def from_checkpoint(
        cls,
        mast3r_model_checkpoint: PathLike,
        grow_graph: GrowCallable,
        device=torch.device("cpu"),
        **kwargs,
    ) -> Mast3rMatchPipelineStep:
        mast3r_model = load_model(mast3r_model_checkpoint, device=device)
        return cls(
            mast3r_model,
            grow_graph,
            **kwargs,
        )
