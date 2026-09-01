from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from dust3r.utils.image import load_images
from tqdm.auto import tqdm

from mts.core.matching.dense.mast3r import (
    EncodedImageFeaturesDict,
    Mast3rTwoStep,
    NdFeatures,
    StFeatures,
)
from mts.core.matching.dense.merge.round import merge_matches
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.model.mast3r.io import load_model
from mts.core.scene_graph.model import Image, MatchKind
from mts.core.scene_graph.nx import extract_matches
from mts.core.types import ImageId, PathLike
from mts.helpers.torch.tensor import from_np, to_numpy
from mts.helpers.torch.tensor import to as to_device
from mts.pipeline.repository.base import BaseImageRepository, SceneScopedImageRepository
from mts.pipeline.step.base import PerSceneStep
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


class Mast3rDecodedMatchPipelineStep(PerSceneStep):
    """Builds a scene graph purely from MASt3R pair decodings that
    `Mast3rEncodeDecodeStep` already persisted to the image repository --
    it grows the graph over every pair `Mast3rEncodeDecodeStep` validated
    (recorded under `validated_pairs_name`), loading the cached decoded
    descriptor/confidence grids and extracting/validating sparse
    correspondences for each. If a validated pair's decoding was never
    persisted (e.g. `Mast3rEncodeDecodeStep` was run with
    `store_pairs=False`), and a `mast3r_two_step` model was configured, it
    is recomputed from the pair's per-image encodings, the same way
    `Mast3rEncodeDecodeStep` decodes a pair.

    Before any pair is decoded, `run` calls `_prepare_encodings` once to
    populate an in-memory encoding cache for every repository image:
    persisted encodings are loaded from the repository, and every
    remaining image is batch-loaded and encoded with MASt3R in a single
    pass (the same way `Mast3rEncodeDecodeStep._encode_all_images` does),
    persisting the result back only when `store_encodings=True`. Pair
    decoding then only ever reads from that cache -- it never loads or
    encodes an image on the fly. The cache holds tensors on `device` only
    when `cache_encodings_on_device=True`, otherwise it holds them on CPU
    (moved to `device` on each lookup) so the cache doesn't pin every
    image's encoding in GPU/MPS memory.
    """

    def __init__(
        self,
        grow_graph: GrowCallable,
        prune_connections: PruneCallable | None = None,
        mast3r_two_step: Mast3rTwoStep | None = None,
        decodings_name: str = "mast3r",
        encodings_name: str = "mast3r-encoding",
        image_size: int = 512,
        kernel_size: int = 7,
        min_conf: float = 1.01,
        device: str | torch.device = "cpu",
        validated_pairs_name: str = "validated-mast3r-pairs",
        store_encodings: bool = False,
        cache_encodings_on_device: bool = False,
    ) -> None:
        super().__init__()
        self.grow_graph = grow_graph
        self.prune_connections = prune_connections
        self.mast3r_two_step = mast3r_two_step
        self.decodings_name = decodings_name
        self.encodings_name = encodings_name
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.min_conf = min_conf
        self._device = torch.device(device)
        self.validated_pairs_name = validated_pairs_name
        self.store_encodings = store_encodings
        self.cache_encodings_on_device = cache_encodings_on_device
        self._encoding_cache: dict[ImageId, EncodedImageFeaturesDict] = {}

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self._device = torch.device(device)
        return super().to(device, *args, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        mast3r_model_checkpoint: PathLike,
        grow_graph: GrowCallable,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> Mast3rDecodedMatchPipelineStep:
        mast3r_model = load_model(mast3r_model_checkpoint, device=device)
        mast3r_two_step = Mast3rTwoStep(mast3r_model)
        return cls(
            grow_graph,
            mast3r_two_step=mast3r_two_step,
            device=device,
            **kwargs,
        )

    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> Any:
        try:
            self._prepare_encodings(image_repository)
            keypoints_map, matches_map, match_kind_map = self._compute_matches(
                image_repository,
            )
            self._save_matches_and_kpts(
                keypoints_map,
                matches_map,
                match_kind_map,
                image_repository,
            )
        finally:
            # cache is keyed by globally-unique image ids so it stays correct
            # across scenes; clearing bounds memory over a many-scene run.
            self._encoding_cache.clear()

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

        validated_pairs = image_repository.load(self.validated_pairs_name)
        if not validated_pairs:
            LOGGER.info(
                "No validated pairs found under %r; falling back to all repository pairs",
                self.validated_pairs_name,
            )
            validated_pairs = image_repository.get_pairs()
        possible_pairs = [
            (
                str(image_repository.get_filepath(st_id)),
                str(image_repository.get_filepath(nd_id)),
            )
            for st_id, nd_id in validated_pairs
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
            image_repository.add_matches(st_image_id, nd_image_id, matches, name="mast3r")
            kind = match_kind_map.get((st_image_filepath, nd_image_filepath))
            if kind is not None:
                image_repository.upsert_match_metadata(st_image_id, nd_image_id, match_kind=kind.value)

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
            st_features, nd_features = self._recompute_pair_features(image_repository, st_id, nd_id)
        else:
            decoded = to_device(from_np(decoded), device=self._device)
            if direction == (nd_id, st_id):
                st_features, nd_features = decoded["nd_features"], decoded["st_features"]
            else:
                st_features, nd_features = decoded["st_features"], decoded["nd_features"]

        if st_features is None or nd_features is None:
            return np.array([]), np.array([])

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

        st_kpts = st_kpts[..., ::-1]
        nd_kpts = nd_kpts[..., ::-1]
        try:
            inlier_matches = validate_kps_matches(st_kpts, nd_kpts, st_hw, nd_hw)
        except Exception:
            LOGGER.exception("Could not validate decoded matches for %s <-> %s", st_id, nd_id)
            return np.array([]), np.array([])

        if len(inlier_matches) == 0:
            return np.array([]), np.array([])

        return st_kpts[inlier_matches[:, 0]], nd_kpts[inlier_matches[:, 1]]

    def _recompute_pair_features(
        self,
        image_repository: BaseImageRepository,
        st_id: ImageId,
        nd_id: ImageId,
    ) -> tuple[StFeatures | None, NdFeatures | None]:
        if self.mast3r_two_step is None:
            LOGGER.warning(
                "No persisted decoding for %s <-> %s and no MASt3R model configured to recompute it",
                st_id,
                nd_id,
            )
            return None, None

        st_encoded = self._get_encoding(image_repository, st_id)
        nd_encoded = self._get_encoding(image_repository, nd_id)
        if st_encoded is None or nd_encoded is None:
            LOGGER.warning(
                "Could not obtain a MASt3R encoding for %s <-> %s; cannot recompute decoding",
                st_id,
                nd_id,
            )
            return None, None

        try:
            with torch.inference_mode():
                decoded = self.mast3r_two_step.decode_feature_pairs(st_encoded, nd_encoded)
        except Exception:
            LOGGER.exception("Could not recompute decoding for %s <-> %s", st_id, nd_id)
            return None, None

        return decoded["st_features"], decoded["nd_features"]

    def _prepare_encodings(self, image_repository: BaseImageRepository) -> None:
        """Populates `_encoding_cache` for every repository image before
        any pair is decoded: images with a persisted encoding are loaded
        from the repository, everything else is batch-loaded and encoded
        with MASt3R in a single pass (persisted back when
        `store_encodings=True`)."""
        image_ids = list(image_repository.image_ids())
        missing_ids: list[ImageId] = []
        for image_id in image_ids:
            if image_id in self._encoding_cache:
                continue
            encoded_np = image_repository.load(f"{self.encodings_name}_{image_id}")
            if encoded_np is None:
                missing_ids.append(image_id)
                continue
            self._cache_encoding(image_id, from_np(encoded_np))

        if not missing_ids:
            return

        if self.mast3r_two_step is None:
            LOGGER.warning(
                "No persisted encoding for %d image(s) and no MASt3R model configured to compute them",
                len(missing_ids),
            )
            return

        LOGGER.info("Encoding %d image(s) with MASt3R...", len(missing_ids))
        filepaths = [str(image_repository.get_filepath(image_id)) for image_id in missing_ids]
        try:
            images = load_images(filepaths, size=self.image_size, verbose=False)
        except Exception:
            LOGGER.exception("Could not load images for MASt3R encoding")
            return

        for image_id, image in zip(missing_ids, tqdm(images, desc="Encoding images")):
            encoded = self._encode_loaded_image(image_id, image)
            if encoded is None:
                continue

            if self.store_encodings:
                image_repository.store(f"{self.encodings_name}_{image_id}", to_numpy(encoded))

            self._cache_encoding(image_id, to_device(encoded, device=torch.device("cpu")))

    def _encode_loaded_image(
        self,
        image_id: ImageId,
        image: dict[str, Any],
    ) -> EncodedImageFeaturesDict | None:
        ignore_keys = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}
        for name in image.keys():
            if name in ignore_keys:
                continue
            image[name] = image[name].to(self._device, non_blocking=True)

        img = image["img"]
        true_shape = image.get("true_shape")
        if true_shape is not None:
            if isinstance(true_shape, np.ndarray):
                true_shape = torch.from_numpy(true_shape)
        else:
            true_shape = torch.tensor(img.shape[-2:])[None].repeat(img.shape[0], 1)

        try:
            with torch.inference_mode():
                encoded_image_dict = self.mast3r_two_step.encode_image(
                    {"image": img, "true_shape": true_shape},
                )
        except Exception:
            LOGGER.exception("Could not encode image %s with MASt3R", image_id)
            return None

        return EncodedImageFeaturesDict.from_add_shape(encoded_image_dict, true_shape)

    def _cache_encoding(
        self,
        image_id: ImageId,
        cpu_encoded: EncodedImageFeaturesDict,
    ) -> None:
        if self.cache_encodings_on_device:
            self._encoding_cache[image_id] = to_device(cpu_encoded, device=self._device)
        else:
            self._encoding_cache[image_id] = cpu_encoded

    def _get_encoding(
        self,
        image_repository: BaseImageRepository,
        image_id: ImageId,
    ) -> EncodedImageFeaturesDict | None:
        cached = self._encoding_cache.get(image_id)
        if cached is None:
            LOGGER.warning(
                "No cached MASt3R encoding for %s; _prepare_encodings should have "
                "populated it upfront",
                image_id,
            )
            return None

        if self.cache_encodings_on_device:
            return cached
        return to_device(cached, device=self._device)
