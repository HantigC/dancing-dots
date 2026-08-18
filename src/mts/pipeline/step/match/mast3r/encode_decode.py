import logging
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from dust3r.utils.image import load_images
from mast3r.fast_nn import cdistMatcher
from tqdm.auto import tqdm

from mts.core.matching.dense.mast3r import (
    DecodedImagePairDict,
    EncodedImageDict,
    EncodedImageFeaturesDict,
    Mast3rTwoStep,
)
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.model.mast3r.io import load_model
from mts.core.model.mast3r.transform import transform_keypoints_to_original
from mts.core.types import ImageId, PathLike
from mts.helpers.torch.tensor import to, to_numpy
from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.base import BasePipelineStep, use_image_repository

LOGGER = logging.getLogger(__name__)

EncodedImageMap = dict[ImageId, EncodedImageDict]
ImageShapesMap = dict[ImageId, torch.Tensor]


class DescriptorsGrid(TypedDict):
    conf: torch.Tensor
    desc: torch.Tensor

    @classmethod
    def from_tuple(cls, conf, desc) -> "DescriptorsGrid":
        if desc.ndim == 4:
            if desc.shape[0] != 1:
                raise ValueError(f"descriptors should have batch size of 1, not {desc.shape[0]}")
            desc = desc.squeeze(0)

        if conf.ndim == 3:
            if conf.shape[0] != 1:
                raise ValueError(f"confidence should have batch size of 1, not {conf.shape[0]}")
            conf = conf.squeeze(0)

        return {
            "conf": conf,
            "desc": desc,
        }


class KeypointsDescriptors(TypedDict):
    desc: torch.Tensor
    coords: torch.Tensor | np.ndarray

    @classmethod
    def from_tuple(
        cls,
        desc: torch.Tensor,
        coords: torch.Tensor | np.ndarray,
    ) -> "KeypointsDescriptors":
        return {
            "desc": desc,
            "coords": coords,
        }


def match_descriptors(
    st_kpts_descriptors: KeypointsDescriptors,
    nd_kpts_descriptors: KeypointsDescriptors,
    device: str | torch.device,
    **matcher_kw,
) -> tuple[np.ndarray, np.ndarray]:
    st_desc = st_kpts_descriptors["desc"]
    nd_desc = nd_kpts_descriptors["desc"]

    st_coords = st_kpts_descriptors["coords"]
    nd_coords = nd_kpts_descriptors["coords"]

    st_tree = cdistMatcher(st_desc, device=device)
    nd_tree = cdistMatcher(nd_desc, device=device)

    xy1 = np.int32(np.arange(0, len(st_desc)))
    xy2 = np.full_like(xy1, -1)

    old_xy1 = xy1.copy()
    old_xy2 = xy2.copy()

    notyet = np.ones(len(xy1), dtype=bool)

    niter = 0
    max_iter = 1
    while notyet.any():
        _, xy2[notyet] = to_numpy(
            nd_tree.query(st_desc[xy1[notyet]], **matcher_kw),
        )
        _, xy1[notyet] = to_numpy(
            st_tree.query(nd_desc[xy2[notyet]], **matcher_kw),
        )
        notyet &= old_xy1 != xy1
        niter += 1
        if niter >= max_iter:
            break
        old_xy1[:] = xy1
        old_xy2[:] = xy2

    st_matched_kpts = st_coords[xy1].cpu().numpy()
    nd_matched_kpts = nd_coords[xy2].cpu().numpy()

    _, st_dedup_indices = np.unique(
        st_matched_kpts,
        axis=0,
        return_index=True,
    )
    _, nd_dedup_indices = np.unique(
        nd_matched_kpts,
        axis=0,
        return_index=True,
    )

    deduped_indices = np.intersect1d(
        st_dedup_indices,
        nd_dedup_indices,
    )

    nd_deduped_kpts = nd_matched_kpts[deduped_indices]
    st_deduped_kpts = st_matched_kpts[deduped_indices]
    return st_deduped_kpts, nd_deduped_kpts


ImageSizeHW = tuple[int, int]


def make_coord_grid_as(grid: torch.Tensor):
    h, w = grid.shape
    coord_grid = torch.stack(
        torch.meshgrid(
            torch.arange(h),
            torch.arange(w),
        )
    ).permute(1, 2, 0)

    return coord_grid


def get_coords(
    conf: torch.Tensor,
    kernel_size: int = 5,
    min_conf: float = 1.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, indices = F.max_pool2d(
        conf.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        return_indices=True,
    )
    h, w = conf.shape
    mask = out > min_conf
    indices = indices[mask]

    row = indices // w
    col = indices % w

    rows, cols = torch.stack(
        [
            row,
            col,
        ]
    )
    return rows, cols


def extract_sparse_matches(
    st_features: DescriptorsGrid,
    nd_features: DescriptorsGrid,
    device: str | torch.device | None = None,
    kernel_size: int = 7,
    min_conf: float = 1.01,
    st_original_size: ImageSizeHW | None = None,
    nd_original_size: ImageSizeHW | None = None,
    **matcher_kw,
) -> tuple[np.ndarray, np.ndarray]:
    st_desc = st_features["desc"]
    nd_desc = nd_features["desc"]

    st_conf = st_features["conf"]
    nd_conf = nd_features["conf"]
    device = device or st_desc.device

    st_coord_grid = make_coord_grid_as(st_conf).to(
        device=device,
    )
    nd_coord_grid = make_coord_grid_as(nd_conf).to(
        device=device,
    )

    st_rows, st_cols = get_coords(
        st_conf,
        kernel_size=kernel_size,
        min_conf=min_conf,
    )

    if len(st_rows) == 0 and len(st_cols) == 0:
        return (
            np.array([]),
            np.array([]),
        )
    nd_rows, nd_cols = get_coords(
        nd_conf,
        kernel_size=kernel_size,
        min_conf=min_conf,
    )

    if len(nd_rows) == 0 and len(nd_cols) == 0:
        return np.array([]), np.array([])

    selected_st_desc = st_desc[st_rows, st_cols]
    selected_st_coords = st_coord_grid[st_rows, st_cols]

    selected_nd_desc = nd_desc[nd_rows, nd_cols]
    selected_nd_coords = nd_coord_grid[nd_rows, nd_cols]

    st_kpts_descriptors = KeypointsDescriptors.from_tuple(
        selected_st_desc,
        selected_st_coords,
    )

    nd_kpts_descriptors = KeypointsDescriptors.from_tuple(
        selected_nd_desc,
        selected_nd_coords,
    )
    st_matched_kpts, nd_matched_kpts = match_descriptors(
        st_kpts_descriptors,
        nd_kpts_descriptors,
        device=device,
        **matcher_kw,
    )
    if st_original_size is not None and nd_original_size is not None:
        st_matched_kpts = transform_keypoints_to_original(
            st_matched_kpts,
            st_original_size,
        )
        nd_matched_kpts = transform_keypoints_to_original(nd_matched_kpts, nd_original_size)
    return st_matched_kpts, nd_matched_kpts


class Mast3rEncodeDecodeStep(BasePipelineStep):
    """Encodes every image and decodes every repository pair with MASt3R,
    persisting both to the image repository. Every encoding is persisted;
    a pair's decoding is only persisted if its two-view-validated sparse
    matches (via `extract_sparse_matches` + `validate_kps_matches`) meet
    `min_matches`. Does not extract keypoints/matches for reconstruction
    or grow a scene graph -- purely a compute-and-cache step.
    """

    def __init__(
        self,
        mast3r_two_step: Mast3rTwoStep,
        image_size: int = 512,
        kernel_size: int = 7,
        encodings_name: str = "mast3r-encoding",
        decodings_name: str = "mast3r",
        min_matches: int = 50,
        store_pairs: bool = False,
        validated_pairs_name: str = "validated-mast3r-pairs",
    ) -> None:
        super().__init__()
        self.mast3r_two_step = mast3r_two_step
        self.image_size = image_size
        self.encodings_name = encodings_name
        self.decodings_name = decodings_name
        self.min_matches = min_matches
        self.kernel_size = kernel_size
        self.store_pairs = store_pairs
        self.validated_pairs_name = validated_pairs_name

    @use_image_repository
    def run(self, image_repository: BaseImageRepository) -> None:
        LOGGER.info("Encoding all images with MASt3R...")
        encoded_map, shapes_map = self._encode_all_images(image_repository)
        pairs = image_repository.get_pairs()
        LOGGER.info("Decoding %d repository pairs with MASt3R...", len(pairs))
        self._decode_all_pairs(image_repository, encoded_map, shapes_map, pairs)
        LOGGER.info("Mast3rEncodeDecodeStep finished")

    def _encode_image(self, image) -> tuple[EncodedImageDict, torch.Tensor]:
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
        for name in image.keys():
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

        encoded_image_features = self.mast3r_two_step.encode_image(
            {
                "image": img1,
                "true_shape": true_shape,
            }
        )
        return encoded_image_features, true_shape

    def _encode_all_images(
        self,
        image_repository: BaseImageRepository,
    ) -> tuple[EncodedImageMap, ImageShapesMap]:
        image_ids = list(image_repository.image_ids())
        image_filepaths = [str(image_repository.get_filepath(image_id)) for image_id in image_ids]
        images = load_images(
            image_filepaths,
            size=self.image_size,
            verbose=False,
        )

        encoded_map = {}
        shapes_map = {}
        with torch.no_grad():
            for image_id, image in zip(image_ids, tqdm(images, desc="Encoding images")):
                encoded_image_features, true_shape = self._encode_image(image)
                encoded_cpu = to(encoded_image_features, device=torch.device("cpu"))
                encoded_map[image_id] = encoded_cpu
                shapes_map[image_id] = true_shape
                self._persist_encoding(image_repository, image_id, encoded_cpu, true_shape)

        return encoded_map, shapes_map

    def _persist_encoding(
        self,
        image_repository: BaseImageRepository,
        image_id: ImageId,
        encoded: EncodedImageDict,
        true_shape: torch.Tensor,
    ) -> None:
        encoded_np = to_numpy(encoded)
        encoded_np["true_shape"] = to_numpy(true_shape)
        image_repository.store(f"{self.encodings_name}_{image_id}", encoded_np)

    def _decode_pair(
        self,
        st_encoded_image_features: EncodedImageDict,
        nd_encoded_image_features: EncodedImageDict,
        st_true_shape: torch.Tensor,
        nd_true_shape: torch.Tensor,
    ):
        return self.mast3r_two_step.decode_feature_pairs(
            to(
                EncodedImageFeaturesDict.from_add_shape(st_encoded_image_features, st_true_shape),
                device=self.device,
            ),
            to(
                EncodedImageFeaturesDict.from_add_shape(nd_encoded_image_features, nd_true_shape),
                device=self.device,
            ),
        )

    def _decode_all_pairs(
        self,
        image_repository: BaseImageRepository,
        encoded_map: EncodedImageMap,
        shapes_map: ImageShapesMap,
        pairs,
    ) -> None:
        validated_pairs = []
        with torch.no_grad():
            for st_id, nd_id in tqdm(pairs, desc="Decoding pairs"):
                decoded = self._decode_pair(
                    encoded_map[st_id],
                    encoded_map[nd_id],
                    shapes_map[st_id],
                    shapes_map[nd_id],
                )
                num_matches = self._count_validated_matches(image_repository, st_id, nd_id, decoded)
                if num_matches < self.min_matches:
                    LOGGER.debug(
                        "Skipping decoding for pair (%s, %s): %d validated matches < min_matches=%d",
                        st_id,
                        nd_id,
                        num_matches,
                        self.min_matches,
                    )
                    continue

                validated_pairs.append((st_id, nd_id))

                if not self.store_pairs:
                    continue

                decoded_np = to_numpy(decoded)
                image_repository.store_pair(
                    st_id,
                    nd_id,
                    self.decodings_name,
                    decoded_np,
                )

        image_repository.store(self.validated_pairs_name, validated_pairs)

    def _count_validated_matches(
        self,
        image_repository: BaseImageRepository,
        st_id: ImageId,
        nd_id: ImageId,
        decoded: DecodedImagePairDict,
    ) -> int:
        st_features = decoded["st_features"]
        nd_features = decoded["nd_features"]

        st_grid = DescriptorsGrid.from_tuple(st_features["desc_conf"], st_features["desc"])
        nd_grid = DescriptorsGrid.from_tuple(nd_features["desc_conf"], nd_features["desc"])

        st_original_size = image_repository.get_size_hw(st_id)
        nd_original_size = image_repository.get_size_hw(nd_id)

        st_kpts, nd_kpts = extract_sparse_matches(
            st_grid,
            nd_grid,
            device=self.device,
            kernel_size=self.kernel_size,
            st_original_size=st_original_size,
            nd_original_size=nd_original_size,
        )
        if len(st_kpts) == 0:
            return 0

        try:
            st_kpts = st_kpts[..., ::-1]
            nd_kpts = nd_kpts[..., ::-1]
            inlier_matches = validate_kps_matches(
                st_kpts,
                nd_kpts,
                st_original_size,
                nd_original_size,
            )
        except Exception:
            LOGGER.exception("Two-view validation failed for pair (%s, %s)", st_id, nd_id)
            return 0

        return len(inlier_matches)

    @classmethod
    def from_checkpoint(
        cls,
        mast3r_model_checkpoint: PathLike,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> "Mast3rEncodeDecodeStep":
        mast3r_model = load_model(mast3r_model_checkpoint, device=device)
        mast3r_two_step = Mast3rTwoStep(mast3r_model)
        return cls(mast3r_two_step, **kwargs)
