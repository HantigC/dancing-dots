import logging
from pathlib import Path
from typing import Any, Callable, TypedDict

import networkx as nx
import numpy as np
import torch
from dust3r.utils.device import collate_with_cat
from dust3r.utils.image import load_images
from tqdm.auto import tqdm

from mts.core.matching.dense.mast3r import (
    DecodedImagePairDict,
    EncodedImageDict,
    EncodedImageFeaturesDict,
    Mast3rTwoStep,
    extract_dense_kpts,
)
from mts.core.matching.dense.merge.round import merge_matches
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.model.mast3r.io import load_model
from mts.core.scene_graph.model import Image, MatchKind, TwoViewEdge
from mts.core.scene_graph.nx import extract_matches
from mts.core.types import ImageId, Pairs, PairType, PathLike
from mts.helpers.torch.tensor import to
from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.base import BasePipelineStep

LOGGER = logging.getLogger(__name__)
GrowCallable = Callable[
    [
        nx.Graph,
        list[tuple[str, str]],
        Callable[[str, str], np.ndarray],
    ],
    None,
]


class CollatedBatch(TypedDict):
    st_features_batch: EncodedImageFeaturesDict
    nd_features_batch: EncodedImageFeaturesDict

    st_shapes_batch: torch.Tensor
    nd_shapes_batch: torch.Tensor


def collect_features(
    batch_of_pair_nums: list[int],
    pairs,
    feature_map,
    shapes_map,
) -> CollatedBatch:
    st_features = []
    nd_features = []

    st_shapes = []
    nd_shapes = []

    for st_image_id, nd_image_id in [pairs[idx] for idx in batch_of_pair_nums]:
        st_features.append(feature_map[st_image_id])
        nd_features.append(feature_map[nd_image_id])

        st_shapes.append(shapes_map[st_image_id])
        nd_shapes.append(shapes_map[nd_image_id])

    st_features_batch = collate_with_cat(st_features)
    nd_features_batch = collate_with_cat(nd_features)

    st_shapes_batch = collate_with_cat(st_shapes)
    nd_shapes_batch = collate_with_cat(nd_shapes)
    collated_features = {
        "st_features_batch": st_features_batch,
        "nd_features_batch": nd_features_batch,
        "st_shapes_batch": st_shapes_batch,
        "nd_shapes_batch": nd_shapes_batch,
    }
    return collated_features


def unbatch(
    features: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    batch_size = next(iter(features.values())).shape[0]
    batch_list = [{} for _ in range(batch_size)]

    for k, vs in features.items():
        for idx, v in enumerate(vs):
            batch_list[idx][k] = v
    return batch_list


EncodedImageMap = dict[ImageId, EncodedImageDict]
ImageShapesMap = dict[ImageId, torch.Tensor]


class Mast3rMatchPipelineStep(BasePipelineStep):
    def __init__(
        self,
        mast3r_two_step: Mast3rTwoStep,
        grow_graph: GrowCallable,
        verbose: bool = True,
        image_size: int = 512,
        min_pairs: int = 50,
        match_conf_th: float = 0.5,
        pixel_tol: int = 0,
        top_k_matches: int | None = None
    ) -> None:
        super().__init__()
        self.mast3r_two_step = mast3r_two_step
        self.verbose = verbose
        self.grow_graph = grow_graph
        self.match_conf_th = match_conf_th
        self.min_pairs = min_pairs
        self.pixel_tol = pixel_tol
        self.image_size = image_size
        self.top_k_matches = top_k_matches

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

    def _encode_image(self, image) -> EncodedImageDict:
        ignore_keys = set(
            [
                "depthmap",
                "dataset",
                "label",
                "instance",
                "idx",
                "true_shape",
                "rng",
            ]
        )
        for name in image.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            image[name] = image[name].to(self.device, non_blocking=True)

        img1 = image["img"]
        b = img1.shape[0]
        true_shape = image.get("true_shape")
        if true_shape is not None:
            if isinstance(true_shape, np.ndarray):
                true_shape = torch.from_numpy(true_shape)
        else:
            true_shape = (torch.tensor(img1.shape[-2:])[None].repeat(b, 1),)

        st_encoded_image_features = self.mast3r_two_step.encode_image(
            {
                "image": img1,
                "true_shape": true_shape,
            }
        )
        return st_encoded_image_features, true_shape

    def _encode_images(
        self,
        image_repository: BaseImageRepository,
    ) -> tuple[EncodedImageMap, ImageShapesMap]:
        LOGGER.info("Encode images...")
        image_ids = list(image_repository.image_ids())
        image_filepaths = [
            image_repository.get_filepath(
                image_id,
            )
            for image_id in image_ids
        ]
        images = load_images(
            image_filepaths,
            size=self.image_size,
            verbose=False,
        )

        feature_map = {}
        shapes_map = {}
        with torch.no_grad():
            for image_id, image in zip(image_ids, tqdm(images)):
                encoded_image_feature, true_shape = self._encode_image(image)
                feature_map[image_id] = to(
                    encoded_image_feature, device=torch.device("cpu")
                )
                shapes_map[image_id] = true_shape

        return feature_map, shapes_map

    def _decode_pair(
        self,
        st_encoded_image_features: EncodedImageDict,
        nd_encoded_image_features: EncodedImageDict,
        st_true_shape: torch.Tensor,
        nd_true_shape: torch.Tensor,
    ) -> DecodedImagePairDict:
        decoded_feature_pairs = self.mast3r_two_step.decode_feature_pairs(
            to(
                EncodedImageFeaturesDict.from_add_shape(
                    st_encoded_image_features, st_true_shape
                ),
                device=self.device,
            ),
            to(
                EncodedImageFeaturesDict.from_add_shape(
                    nd_encoded_image_features, nd_true_shape
                ),
                device=self.device,
            ),
        )

        return decoded_feature_pairs

    def _extract_dense_kpts(
        self,
        decoded_feature_pairs: DecodedImagePairDict,
        st_original_size: tuple[int, int],
        nd_original_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        decoded_feature_pairs["st_features"]["true_shape"] = decoded_feature_pairs[
            "st_features"
        ]["true_shape"].squeeze()
        decoded_feature_pairs["nd_features"]["true_shape"] = decoded_feature_pairs[
            "nd_features"
        ]["true_shape"].squeeze()
        try:
            st_kpts, nd_kpts = extract_dense_kpts(
                to(decoded_feature_pairs["st_features"], device=torch.device("cpu")),
                to(decoded_feature_pairs["nd_features"], device=torch.device("cpu")),
                st_original_size,
                nd_original_size,
                self.match_conf_th,
                self.min_pairs,
                self.device,
                top_k=self.top_k_matches,
                pixel_tol=self.pixel_tol,
            )
        except Exception:
            LOGGER.exception("Trouble with extracting the dense keypoints")
            st_kpts, nd_kpts = np.array([]), np.array([])

        return st_kpts, nd_kpts

    def _compute_pair_matched_kpts(
        self,
        st_encoded_image_features: EncodedImageDict,
        nd_encoded_image_features: EncodedImageDict,
        st_true_shape: torch.Tensor,
        nd_true_shape: torch.Tensor,
        st_original_size: tuple[int, int],
        nd_original_size: tuple[int, int],
        validate: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:

        decoded_feature_pairs = self._decode_pair(
            st_encoded_image_features,
            nd_encoded_image_features,
            st_true_shape,
            nd_true_shape,
        )
        decoded_feature_pairs["st_features"]["true_shape"] = decoded_feature_pairs[
            "st_features"
        ]["true_shape"].squeeze()
        decoded_feature_pairs["nd_features"]["true_shape"] = decoded_feature_pairs[
            "nd_features"
        ]["true_shape"].squeeze()
        try:
            st_kpts, nd_kpts = extract_dense_kpts(
                to(decoded_feature_pairs["st_features"], device=torch.device("cpu")),
                to(decoded_feature_pairs["nd_features"], device=torch.device("cpu")),
                st_original_size,
                nd_original_size,
                self.match_conf_th,
                self.min_pairs,
                self.device,
                top_k=self.top_k_matches,
                pixel_tol=self.pixel_tol,
            )
        except Exception:
            LOGGER.exception("Trouble with extracting the dense keypoints")
            st_kpts, nd_kpts = np.array([]), np.array([])

        if len(st_kpts) == 0:
            return st_kpts, nd_kpts

        if validate:
            try:
                inlier_matches = validate_kps_matches(
                    st_kpts,
                    nd_kpts,
                    st_original_size,
                    nd_original_size,
                )
            except Exception:
                LOGGER.exception("Not able to validate")
                st_kpts = np.empty((0, 2), np.float32)
                nd_kpts = np.empty((0, 2), np.float32)
            else:
                st_kpts = st_kpts[inlier_matches[:, 0]]
                nd_kpts = nd_kpts[inlier_matches[:, 1]]

        return st_kpts, nd_kpts

    def _compute_matched_kpts(
        self,
        feature_map: EncodedImageMap,
        shapes_map: ImageShapesMap,
        pairs: Pairs[int],
        original_sizes_map: dict[ImageId, tuple[int, int]],
        validate: bool = True,
    ) -> dict[PairType[ImageId], DecodedImagePairDict]:
        dense_kps = {}
        with torch.no_grad():
            for st_image_id, nd_image_id in tqdm(pairs):
                st_encoded_image_features = feature_map[st_image_id]
                nd_encoded_image_features = feature_map[nd_image_id]

                st_true_shape = shapes_map[st_image_id]
                nd_true_shape = shapes_map[nd_image_id]
                st_original_size = original_sizes_map[st_image_id]
                nd_original_size = original_sizes_map[nd_image_id]
                st_kpts, nd_kpts = self._compute_pair_matched_kpts(
                    st_encoded_image_features,
                    nd_encoded_image_features,
                    st_true_shape,
                    nd_true_shape,
                    st_original_size,
                    nd_original_size,
                    validate,
                )
                if len(st_kpts) == 0:
                    continue
                dense_kps[st_image_id, nd_image_id] = st_kpts, nd_kpts
        return dense_kps

    def _decode_batched_pairs(
        self,
        feature_map: EncodedImageMap,
        shapes_map: ImageShapesMap,
        pairs: Pairs[int],
        original_sizes_map: dict[ImageId, tuple[int, int]],
    ) -> dict[PairType[ImageId], DecodedImagePairDict]:
        decoded_repr = {}

        with torch.no_grad():
            for st_image_id, nd_image_id in tqdm(pairs):
                st_encoded_image_features = feature_map[st_image_id]
                nd_encoded_image_features = feature_map[nd_image_id]

                st_true_shape = shapes_map[st_image_id]
                nd_true_shape = shapes_map[nd_image_id]
                decoded_repr[st_image_id, nd_image_id] = self._decode_pair(
                    st_encoded_image_features,
                    nd_encoded_image_features,
                    st_true_shape,
                    nd_true_shape,
                )

        return decoded_repr

    def _compute_matches(
        self,
        image_repository: BaseImageRepository,
        mst_pairs: list[PairType[ImageId]],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[tuple[str, str], np.ndarray],
        dict[tuple[str, str], MatchKind],
    ]:
        features_map, shapes_map = self._encode_images(image_repository)
        scene_graph = self._create_graph(
            features_map,
            shapes_map,
            image_repository,
            mst_pairs,
        )
        matches_dict, match_kind_map = extract_matches(scene_graph)
        global_keypoints, global_matches = merge_matches(matches_dict)
        return global_keypoints, global_matches, match_kind_map

    def _create_graph(
        self,
        features_map: EncodedImageMap,
        shapes_map: ImageShapesMap,
        image_repository: BaseImageRepository,
        mst_pairs: list[PairType[ImageId]],
    ) -> nx.Graph:
        scene_graph, filepath_to_hw = self._init_graph_from_mst(
            features_map,
            shapes_map,
            image_repository,
            mst_pairs,
        )
        possible_pairs = []
        for st_id, nd_id in image_repository.get_pairs():
            st_filepath = str(image_repository.get_filepath(st_id))
            nd_filepath = str(image_repository.get_filepath(nd_id))
            if not scene_graph.has_edge(st_filepath, nd_filepath):
                possible_pairs.append((st_filepath, nd_filepath))

        self.grow_graph(
            scene_graph,
            possible_pairs,
            lambda st_filepath, nd_filepath: self._match_two_images(
                st_filepath,
                nd_filepath,
                features_map,
                shapes_map,
                image_repository,
                validate=True,
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
        features_map: EncodedImageMap,
        shapes_map: ImageShapesMap,
        image_repository: BaseImageRepository,
        validate: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        st_image_id = image_repository.get_image_id(st_filepath)
        nd_image_id = image_repository.get_image_id(nd_filepath)
        st_original_size = image_repository.get_size_hw(st_image_id)
        nd_original_size = image_repository.get_size_hw(st_image_id)

        st_true_shape = shapes_map[st_image_id]
        nd_true_shape = shapes_map[nd_image_id]
        st_encoded_image_features = features_map[st_image_id]
        nd_encoded_image_features = features_map[nd_image_id]

        st_kpts, nd_kpts = self._compute_pair_matched_kpts(
            st_encoded_image_features,
            nd_encoded_image_features,
            st_true_shape,
            nd_true_shape,
            st_original_size,
            nd_original_size,
            validate,
        )
        return st_kpts, nd_kpts

    def _init_graph_from_mst(
        self,
        features_map: EncodedImageMap,
        shapes_map: ImageShapesMap,
        image_repository: BaseImageRepository,
        mst_pairs: list[PairType[ImageId]],
    ) -> tuple[nx.Graph, dict[str, tuple[int, int]]]:
        image_id_to_num = {}
        filepaths_as_str = []
        original_size_map = {}
        for num, image_id in enumerate(image_repository.image_ids()):
            image_id_to_num[image_id] = num
            filepaths_as_str.append(str(image_repository.get_filepath(image_id)))
            original_size_map[image_id] = image_repository.get_size_hw(image_id)

        matches_map = self._compute_matched_kpts(
            features_map,
            shapes_map,
            mst_pairs,
            original_size_map,
            validate=True,
        )

        scene_graph = nx.Graph().to_undirected()
        filepath_to_hw = {}
        for image_id in image_repository.image_ids():
            height, width = image_repository.get_size_hw(image_id)
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

        for (st_image_id, nd_image_id), kpts in matches_map.items():
            st_kpts, nd_kpts = kpts
            st_filepath = image_repository.get_filepath(st_image_id)
            nd_filepath = image_repository.get_filepath(nd_image_id)
            st_kpts, nd_kpts = kpts

            if len(st_kpts) == 0:
                continue
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
            )
        return scene_graph, filepath_to_hw

    @classmethod
    def from_checkpoint(
        cls,
        mast3r_model_checkpoint: PathLike,
        grow_graph: GrowCallable,
        device=torch.device("cpu"),
        **kwargs,
    ) -> "Mast3rMatchPipelineStep":
        mast3r_model = load_model(mast3r_model_checkpoint, device=device)
        mast3r_two_step = Mast3rTwoStep(mast3r_model)
        return cls(
            mast3r_two_step,
            grow_graph,
            **kwargs,
        )
