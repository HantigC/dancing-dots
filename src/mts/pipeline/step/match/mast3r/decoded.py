from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import torch

from mts.core.matching.dense.merge.round import merge_matches
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.scene_graph.model import Image, MatchKind, TwoViewEdge
from mts.core.scene_graph.nx import extract_matches
from mts.core.types import ImageId, PairType
from mts.helpers.torch.tensor import from_np
from mts.helpers.torch.tensor import to as to_device
from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.base import BasePipelineStep
from mts.pipeline.step.match.mast3r.encode_decode import (
    DescriptorsGrid,
    extract_sparse_matches,
)

LOGGER = logging.getLogger(__name__)

GrowCallable = Callable[
    [
        nx.Graph,
        list[tuple[str, str]],
        Callable[[str, str], np.ndarray],
    ],
    None,
]

PruneCallable = Callable[[nx.Graph], nx.Graph]


class Mast3rDecodedMatchPipelineStep(BasePipelineStep):
    """Builds a scene graph purely from MASt3R pair decodings that
    `Mast3rEncodeDecodeStep` already persisted to the image repository --
    it never runs the MASt3R model itself. For every pair it needs, it
    loads the cached decoded descriptor/confidence grids, extracts sparse
    correspondences and geometrically validates them. Pairs with no
    cached decoding (skipped by `Mast3rEncodeDecodeStep`'s `min_matches`
    threshold) contribute no edge, same as if matching had failed.
    """

    def __init__(
        self,
        grow_graph: GrowCallable,
        prune_connections: PruneCallable | None = None,
        decodings_name: str = "mast3r",
        kernel_size: int = 7,
        min_conf: float = 1.01,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.grow_graph = grow_graph
        self.prune_connections = prune_connections
        self.decodings_name = decodings_name
        self.kernel_size = kernel_size
        self.min_conf = min_conf
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self._device = torch.device(device)
        return super().to(device, *args, **kwargs)

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
        if self.prune_connections is not None:
            scene_graph = self.prune_connections(scene_graph)
        matches_dict, match_kind_map = extract_matches(scene_graph)
        global_keypoints, global_matches = merge_matches(matches_dict)
        return global_keypoints, global_matches, match_kind_map

    def _create_graph(
        self,
        image_repository: BaseImageRepository,
        mst_pairs: list[PairType[ImageId]],
    ) -> nx.Graph:
        scene_graph = self._init_graph_from_mst(image_repository, mst_pairs)
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
            lambda st, nd: self._match_two_images(st, nd, image_repository),
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
        image_repository: BaseImageRepository,
    ) -> tuple[np.ndarray, np.ndarray]:
        st_id = image_repository.get_image_id(Path(st_filepath))
        nd_id = image_repository.get_image_id(Path(nd_filepath))
        return self._decoded_pair_matches(image_repository, st_id, nd_id)

    def _decoded_pair_matches(
        self,
        image_repository: BaseImageRepository,
        st_id: ImageId,
        nd_id: ImageId,
    ) -> tuple[np.ndarray, np.ndarray]:
        decoded = image_repository.load_pair(st_id, nd_id, name=self.decodings_name)
        if decoded is None:
            return np.array([]), np.array([])

        decoded = to_device(from_np(decoded), device=self._device)
        st_features = decoded["st_features"]
        nd_features = decoded["nd_features"]

        st_grid = DescriptorsGrid.from_tuple(st_features["conf"], st_features["desc"])
        nd_grid = DescriptorsGrid.from_tuple(nd_features["conf"], nd_features["desc"])

        st_hw = image_repository.get_size_hw(st_id)
        nd_hw = image_repository.get_size_hw(nd_id)

        st_kpts, nd_kpts = extract_sparse_matches(
            st_grid,
            nd_grid,
            device=self._device,
            kernel_size=self.kernel_size,
            min_conf=self.min_conf,
            st_original_size=st_hw,
            nd_original_size=nd_hw,
        )
        if len(st_kpts) == 0:
            return np.array([]), np.array([])

        try:
            inlier_matches = validate_kps_matches(st_kpts, nd_kpts, st_hw, nd_hw)
        except Exception:
            LOGGER.exception(
                "Could not validate decoded matches for %s <-> %s", st_id, nd_id
            )
            return np.array([]), np.array([])

        if len(inlier_matches) == 0:
            return np.array([]), np.array([])

        return st_kpts[inlier_matches[:, 0]], nd_kpts[inlier_matches[:, 1]]

    def _init_graph_from_mst(
        self,
        image_repository: BaseImageRepository,
        mst_pairs: list[PairType[ImageId]],
    ) -> nx.Graph:
        scene_graph = nx.Graph().to_undirected()
        for image_id in image_repository.image_ids():
            height, width = image_repository.get_size_hw(image_id)
            filepath = str(image_repository.get_filepath(image_id))
            scene_graph.add_node(
                filepath,
                image=Image(height=height, width=width),
            )

        for st_id, nd_id in mst_pairs:
            st_kpts, nd_kpts = self._decoded_pair_matches(image_repository, st_id, nd_id)
            if len(st_kpts) == 0:
                continue

            st_filepath = str(image_repository.get_filepath(st_id))
            nd_filepath = str(image_repository.get_filepath(nd_id))
            num_inliers = len(st_kpts)
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
        return scene_graph
