import itertools as it
import logging
from pathlib import Path
from typing import Any, Callable, TypedDict

import more_itertools as mit
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
    extract_dense_keypoints,
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
from mts.utils.iterate import group_by

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
        image_size: int = 500,
    ) -> None:
        super().__init__()
        self.mast3r_two_step = mast3r_two_step
        self.verbose = verbose
        self.grow_graph = grow_graph
        self.image_size = image_size

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

    def _encode_images(
        self,
        image_repository: BaseImageRepository,
    ) -> tuple[EncodedImageMap, ImageShapesMap]:
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
        with torch.no_grad():
            for image_id, image in zip(image_ids, tqdm(images)):
                for name in image.keys():  # pseudo_focal
                    if name in ignore_keys:
                        continue
                    image[name] = image[name].to(self.device, non_blocking=True)

                img1 = image["img"]
                B = img1.shape[0]
                true_shape = image.get("true_shape")
                if true_shape is not None:
                    if isinstance(true_shape, np.ndarray):
                        true_shape = torch.from_numpy(true_shape)
                else:
                    true_shape = (torch.tensor(img1.shape[-2:])[None].repeat(B, 1),)

                st_encoded_image_features = self.mast3r_two_step.encode_image(
                    {
                        "image": img1,
                        "true_shape": true_shape,
                    }
                )
                feature_map[image_id] = st_encoded_image_features
                shapes_map[image_id] = true_shape
        return feature_map, shapes_map

    def _decode_image_pair(
        self,
        feature_map: EncodedImageMap,
        shapes_map: ImageShapesMap,
        image_repository: BaseImageRepository,
    ):

        with torch.no_grad():
            decoded_feature_pairs = self.mast3r_two_step.decode_feature_pairs(
                EncodedImageFeaturesDict.from_add_shape(
                    batched["st_features_batch"],
                    batched["st_shapes_batch"],
                ),
                EncodedImageFeaturesDict.from_add_shape(
                    batched["nd_features_batch"],
                    batched["nd_shapes_batch"],
                ),
            )
            decoded_feature_pairs_list.append(decoded_feature_pairs)

            original_sizes = [
                (
                    image_repository.get_size_hw(pairs[idx][0]),
                    image_repository.get_size_hw(pairs[idx][1]),
                )
                for idx in batch_idx
            ]
            st_features_list = to(
                unbatch(decoded_feature_pairs["st_features"]),
                device=torch.device("cpu"),
            )
            nd_features_list = to(
                unbatch(decoded_feature_pairs["nd_features"]),
                device=torch.device("cpu"),
            )

            for pair_idx, (
                st_original_size,
                nd_original_size,
            ), st_features, nd_features in zip(
                batch_idx, original_sizes, st_features_list, nd_features_list
            ):
                st_features["true_shape"] = st_features["true_shape"].numpy().tolist()
                nd_features["true_shape"] = nd_features["true_shape"].numpy().tolist()
                dense_kps[tuple(pairs[pair_idx])] = extract_dense_kpts(
                    st_features,
                    nd_features,
                    st_original_size,
                    nd_original_size,
                    1.01,
                    50,
                    self.mps_device,
                )

    def _decode_batched_pairs(
        self,
        feature_map: EncodedImageMap,
        shapes_map: ImageShapesMap,
        image_repository: BaseImageRepository,
        pairs: Pairs[int],
        batch_size: int = 1,
    ) -> dict[PairType[ImageId], DecodedImagePairDict]:
        batch_pair_idxs = self._compute_batches_idxs_pairs(
            feature_map, pairs, batch_size
        )
        decoded_feature_pairs_list = []
        dense_kps = {}

        with torch.no_grad():
            for batch_idx in tqdm(batch_pair_idxs):
                batched = collect_features(
                    batch_idx,
                    pairs,
                    feature_map,
                    shapes_map,
                )
                decoded_feature_pairs = self.mast3r_two_step.decode_feature_pairs(
                    EncodedImageFeaturesDict.from_add_shape(
                        batched["st_features_batch"],
                        batched["st_shapes_batch"],
                    ),
                    EncodedImageFeaturesDict.from_add_shape(
                        batched["nd_features_batch"],
                        batched["nd_shapes_batch"],
                    ),
                )
                decoded_feature_pairs_list.append(decoded_feature_pairs)

                original_sizes = [
                    (
                        image_repository.get_size_hw(pairs[idx][0]),
                        image_repository.get_size_hw(pairs[idx][1]),
                    )
                    for idx in batch_idx
                ]
                st_features_list = to(
                    unbatch(decoded_feature_pairs["st_features"]),
                    device=torch.device("cpu"),
                )
                nd_features_list = to(
                    unbatch(decoded_feature_pairs["nd_features"]),
                    device=torch.device("cpu"),
                )

                for pair_idx, (
                    st_original_size,
                    nd_original_size,
                ), st_features, nd_features in zip(
                    batch_idx, original_sizes, st_features_list, nd_features_list
                ):
                    st_features["true_shape"] = (
                        st_features["true_shape"].numpy().tolist()
                    )
                    nd_features["true_shape"] = (
                        nd_features["true_shape"].numpy().tolist()
                    )
                    dense_kps[tuple(pairs[pair_idx])] = extract_dense_kpts(
                        st_features,
                        nd_features,
                        st_original_size,
                        nd_original_size,
                        1.01,
                        50,
                        self.mps_device,
                    )

        return decoded_feature_pairs_list

    def _compute_batches_idxs_pairs(
        self,
        shapes_map: dict[ImageId, torch.Tuple],
        pairs: list[PairType[ImageId]],
        batch_size: int = 1,
    ) -> list[list[int]]:

        size_pair_idxs = (
            [
                (
                    (
                        tuple([*shapes_map[st_image_id].numpy().squeeze().tolist()]),
                        tuple([*shapes_map[nd_image_id].numpy().squeeze().tolist()]),
                    ),
                    pair_num,
                )
                for pair_num, (st_image_id, nd_image_id) in enumerate(pairs)
            ],
        )
        sizes_group = group_by(
            size_pair_idxs,
            key=lambda x: x[0],
            value=lambda x: x[1],
        )

        mixed_bached_gen = (
            mit.chunked(
                pairs_idxs,
                batch_size,
            )
            for pairs_idxs in sizes_group.values()
        )

        batch_pair_idxs = list(it.chain.from_iterable(mixed_bached_gen))
        return batch_pair_idxs

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

        self.grow_graph(
            scene_graph,
            possible_pairs,
            lambda st, nd: self._match_two_images(
                st, nd, features_map, shapes_map, image_repository
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
    ) -> tuple[np.ndarray, np.ndarray]:
        st_image_id = image_repository.get_image_id(st_filepath)
        nd_image_id = image_repository.get_image_id(nd_filepath)

        st_true_shape = shapes_map[st_image_id]
        nd_true_shape = shapes_map[nd_image_id]
        st_encoded_image_features = features_map[st_image_id]
        nd_encoded_image_features = features_map[nd_image_id]

        decoded_feature_pairs = self.mast3r_two_step.decode_feature_pairs(
            EncodedImageFeaturesDict.from_add_shape(
                st_encoded_image_features, st_true_shape
            ),
            EncodedImageFeaturesDict.from_add_shape(
                nd_encoded_image_features, nd_true_shape
            ),
        )

        decoded_feature_pairs["st_features"]["true_shape"] = (
            decoded_feature_pairs["st_features"]["true_shape"].numpy().tolist()
        )
        decoded_feature_pairs["nd_features"]["true_shape"] = (
            decoded_feature_pairs["nd_features"]["true_shape"].numpy().tolist()
        )
        st_original_size = image_repository.get_size_hw(st_image_id)
        nd_original_size = image_repository.get_size_hw(nd_image_id)

        try:
            st_kpts, nd_kpts = extract_dense_kpts(
                decoded_feature_pairs["st_features"],
                decoded_feature_pairs["nd_features"],
                st_original_size,
                nd_original_size,
                1.01,
                50,
                self.device,
            )
        except Exception:
            LOGGER.exception("Trouble with extracting the dense keypoints")
            st_kpts, nd_kpts = np.array([]), np.array([])

        try:
            inlier_matches = validate_kps_matches(
                st_kpts,
                nd_kpts,
                st_original_size,
                nd_original_size,
            )
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
        features_map: EncodedImageMap,
        shapes_map: ImageShapesMap,
        image_repository: BaseImageRepository,
        mst_pairs: list[PairType[ImageId]],
        batch_size: int = 1,
    ) -> tuple[nx.Graph, dict[str, tuple[int, int]]]:
        image_id_to_num = {}
        filepaths_as_str = []
        for num, image_id in enumerate(image_repository.image_ids()):
            image_id_to_num[image_id] = num
            filepaths_as_str.append(str(image_repository.get_filepath(image_id)))

        matches_map = self._decode_batched_pairs(
            features_map,
            shapes_map,
            image_repository,
            mst_pairs,
            batch_size,
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
            )
        return scene_graph, filepath_to_hw

    @classmethod
    def from_checkpoint(
        cls, mast3r_model_checkpoint: PathLike, grow_graph: GrowCallable, **kwargs
    ) -> "Mast3rMatchPipelineStep":
        mast3r_model = load_model(mast3r_model_checkpoint, torch.device("cpu"))
        mast3r_two_step = Mast3rTwoStep(mast3r_model)
        return cls(
            mast3r_two_step,
            grow_graph,
            **kwargs,
        )
