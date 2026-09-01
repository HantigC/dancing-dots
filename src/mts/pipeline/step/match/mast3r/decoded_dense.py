from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import torch

from mts.core.matching.dense.mast3r import extract_dense_kpts
from mts.core.matching.dense.merge.round import merge_matches
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.scene_graph.model import Image, MatchKind
from mts.core.scene_graph.nx import extract_matches
from mts.core.types import ImageId
from mts.helpers.torch.tensor import from_np
from mts.helpers.torch.tensor import to as to_device
from mts.pipeline.repository.base import BaseImageRepository, SceneScopedImageRepository
from mts.pipeline.step.base import PerSceneStep

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


class Mast3rDecodedDenseMatchPipelineStep(PerSceneStep):
    """Builds a scene graph purely from MASt3R pair decodings that
    `Mast3rEncodeDecodeStep` already persisted to the image repository --
    it never runs the MASt3R model itself. It grows the graph over every
    pair with a persisted decoding (pairs skipped by
    `Mast3rEncodeDecodeStep`'s `min_matches` threshold are absent), loading
    the cached decoded feature dicts and extracting/validating dense
    correspondences (via `extract_dense_kpts`) for each, instead of the
    grid-based sparse extraction `Mast3rDecodedMatchPipelineStep` uses.
    """

    def __init__(
        self,
        grow_graph: GrowCallable,
        prune_connections: PruneCallable | None = None,
        decodings_name: str = "mast3r",
        match_conf_th: float = 0.5,
        min_pairs: int = 50,
        pixel_tol: int = 0,
        top_k_matches: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.grow_graph = grow_graph
        self.prune_connections = prune_connections
        self.decodings_name = decodings_name
        self.match_conf_th = match_conf_th
        self.min_pairs = min_pairs
        self.pixel_tol = pixel_tol
        self.top_k_matches = top_k_matches
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self._device = torch.device(device)
        return super().to(device, *args, **kwargs)

    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> Any:
        keypoints_map, matches_map, match_kind_map = self._compute_matches(
            image_repository,
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
    ) -> tuple[
        dict[str, np.ndarray],
        dict[tuple[str, str], np.ndarray],
        dict[tuple[str, str], MatchKind],
    ]:
        scene_graph = self._create_graph(image_repository)
        if self.prune_connections is not None:
            scene_graph = self.prune_connections(scene_graph)
        matches_dict, match_kind_map = extract_matches(scene_graph)
        global_keypoints, global_matches = merge_matches(matches_dict)
        return global_keypoints, global_matches, match_kind_map

    def _create_graph(
        self,
        image_repository: BaseImageRepository,
    ) -> nx.Graph:
        scene_graph = nx.Graph().to_undirected()
        for image_id in image_repository.image_ids():
            height, width = image_repository.get_size_hw(image_id)
            filepath = str(image_repository.get_filepath(image_id))
            scene_graph.add_node(
                filepath,
                image=Image(height=height, width=width),
            )

        possible_pairs = [
            (
                str(image_repository.get_filepath(st_id)),
                str(image_repository.get_filepath(nd_id)),
            )
            for st_id, nd_id in image_repository.get_stored_pairs(self.decodings_name)
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
        decoded, direction = image_repository.load_pair(st_id, nd_id, name=self.decodings_name)
        if decoded is None:
            return np.array([]), np.array([])

        decoded = to_device(from_np(decoded), device=self._device)
        if direction == (nd_id, st_id):
            st_features, nd_features = decoded["nd_features"], decoded["st_features"]
        else:
            st_features, nd_features = decoded["st_features"], decoded["nd_features"]

        st_features = dict(st_features)
        nd_features = dict(nd_features)
        st_features["true_shape"] = st_features["true_shape"].squeeze()
        nd_features["true_shape"] = nd_features["true_shape"].squeeze()

        st_hw = image_repository.get_size_hw(st_id)
        nd_hw = image_repository.get_size_hw(nd_id)

        try:
            st_kpts, nd_kpts = extract_dense_kpts(
                to_device(st_features, device=torch.device("cpu")),
                to_device(nd_features, device=torch.device("cpu")),
                st_hw,
                nd_hw,
                self.match_conf_th,
                self.min_pairs,
                self._device,
                top_k=self.top_k_matches,
                pixel_tol=self.pixel_tol,
            )
        except Exception:
            LOGGER.exception(
                "Could not extract dense keypoints for %s <-> %s", st_id, nd_id
            )
            return np.array([]), np.array([])

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
