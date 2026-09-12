from __future__ import annotations

import gc
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from dust3r.utils.image import load_images
from tqdm.auto import tqdm

from mts.core.matching.dense.fast_nn import dense_extract
from mts.core.matching.dense.mast3r import EncodedImageFeaturesDict, Mast3rTwoStep
from mts.core.matching.dense.merge.round import merge_matches
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.model.mast3r.io import load_model
from mts.core.scene_graph.model import MatchKind
from mts.core.types import ImageId, PathLike
from mts.helpers.torch.tensor import to_numpy
from mts.pipeline.repository.base import BaseImageRepository, SceneScopedImageRepository
from mts.pipeline.step.base import PerSceneStep

LOGGER = logging.getLogger(__name__)

_IGNORE_IMAGE_KEYS = {
    "depthmap",
    "dataset",
    "label",
    "instance",
    "idx",
    "true_shape",
    "rng",
}

EncodingCache = dict[ImageId, EncodedImageFeaturesDict]


class Mast3rFastSingleMatchPipelineStep(PerSceneStep):
    """Per-scene, single-GPU MASt3R matcher (encode-once / decode-per-pair).

    Single-device counterpart to ``Mast3rFastMatchPipelineStep``: runs the
    MASt3R encoder once per image, keeps every encoding resident on
    ``self.device`` for the lifetime of the scene, then walks the pair list
    in order -- decoding + running the all-torch dense reciprocal-NN match
    extraction from :mod:`mts.core.matching.dense.fast_nn`, one pair at a
    time. There is no replica list and no windowing: it's a plain sequential
    loop over a single ``Mast3rTwoStep``, so it moves with a normal
    ``OnDeviceRunner`` in the config like any other step.

    Writes per-image keypoints under ``keypoints_name`` and ``(M, 2)`` index
    matches under ``matches_name`` (both default to ``"mast3r"``), so it is a
    drop-in for ``ColmapReconstructionStep``. Requires a prior pairing step
    (consumes ``image_repository.get_pairs()``); needs no prior
    keypoint-extraction step.
    """

    def __init__(
        self,
        mast3r_two_step: Mast3rTwoStep,
        keypoints_name: str = "mast3r",
        matches_name: str = "mast3r",
        image_size: int = 512,
        match_conf_th: float = 1.01,
        min_pairs: int = 200,
        subsample: int = 16,
        pixel_tol: int = 0,
        max_iter: int = 1,
        top_k_matches: int | None = None,
        validate: bool = True,
        validate_max_error: float = 2.0,
        search_subsample: int | None = None,
    ) -> None:
        super().__init__()
        self.mast3r_two_step = mast3r_two_step

        self.keypoints_name = keypoints_name
        self.matches_name = matches_name
        self.image_size = image_size
        self.match_conf_th = match_conf_th
        self.min_pairs = min_pairs
        self.subsample = subsample
        self.pixel_tol = pixel_tol
        self.max_iter = max_iter
        self.top_k_matches = top_k_matches
        self.validate = validate
        self.validate_max_error = validate_max_error
        self.search_subsample = search_subsample

    @classmethod
    def from_checkpoint(
        cls,
        mast3r_model_checkpoint: PathLike,
        **kwargs,
    ) -> "Mast3rFastSingleMatchPipelineStep":
        # loaded onto the CPU; the config's ``OnDeviceRunner`` moves the whole
        # step (and this submodule with it) to the target device before
        # ``.run()``, same as every other single-device step in this repo.
        mast3r_two_step = Mast3rTwoStep(
            load_model(mast3r_model_checkpoint, device=torch.device("cpu"))
        )
        return cls(mast3r_two_step, **kwargs)

    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> Any:
        pairs = list(image_repository.get_pairs())
        if not pairs:
            LOGGER.warning(
                "Mast3rFastSingleMatchPipelineStep: scene '%s' has no pairs, skipping",
                scene,
            )
            return

        LOGGER.info(
            "Mast3rFastSingleMatchPipelineStep: scene '%s' -- %d pairs on %s",
            scene,
            len(pairs),
            self.device,
        )
        encoding_cache = self._encode_images(image_repository)
        original_sizes = {
            image_id: image_repository.get_size_hw(image_id)
            for image_id in encoding_cache
        }
        filepath_to_id = {
            str(image_repository.get_filepath(image_id)): image_id
            for image_id in encoding_cache
        }
        try:
            keypoints_map, matches_map = self._compute_matches(
                image_repository, pairs, encoding_cache, original_sizes
            )
            self._save_matches_and_kpts(
                keypoints_map, matches_map, filepath_to_id, image_repository
            )
        finally:
            encoding_cache.clear()
            gc.collect()

    def _encode_images(self, image_repository: BaseImageRepository) -> EncodingCache:
        LOGGER.info("Mast3rFastSingleMatchPipelineStep: encoding images...")
        image_ids = list(image_repository.image_ids())
        filepaths = [
            str(image_repository.get_filepath(image_id)) for image_id in image_ids
        ]
        images = load_images(filepaths, size=self.image_size, verbose=False)

        self.mast3r_two_step.eval()

        # a single cache, held on ``self.device`` for the whole scene -- each
        # image is encoded once here and reused for every pair it appears in.
        encoding_cache: EncodingCache = {}
        with torch.inference_mode():
            for image_id, image in zip(image_ids, tqdm(images, desc="Encoding images")):
                encoded = self._encode_image(image)
                if encoded is None:
                    continue
                encoding_cache[image_id] = encoded
        return encoding_cache

    def _encode_image(self, image: dict[str, Any]) -> EncodedImageFeaturesDict | None:
        device = self.device
        for name in image.keys():
            if name in _IGNORE_IMAGE_KEYS:
                continue
            image[name] = image[name].to(device, non_blocking=True)

        img = image["img"]
        true_shape = image.get("true_shape")
        if true_shape is not None:
            if isinstance(true_shape, np.ndarray):
                true_shape = torch.from_numpy(true_shape)
        else:
            true_shape = torch.tensor(img.shape[-2:])[None].repeat(img.shape[0], 1)

        try:
            encoded_image_dict = self.mast3r_two_step.encode_image(
                {"image": img, "true_shape": true_shape}
            )
        except Exception:
            LOGGER.exception(
                "Mast3rFastSingleMatchPipelineStep: could not encode image with MASt3R"
            )
            return None

        return EncodedImageFeaturesDict.from_add_shape(encoded_image_dict, true_shape)

    def _compute_matches(
        self,
        image_repository: BaseImageRepository,
        pairs: list[tuple[ImageId, ImageId]],
        encoding_cache: EncodingCache,
        original_sizes: dict[ImageId, tuple[int, int]],
    ) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]:
        out_match: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        device = self.device
        self.mast3r_two_step.eval()

        with torch.inference_mode():
            for st_id, nd_id in tqdm(
                pairs, desc="Mast3rFastSingleMatchPipelineStep matching"
            ):
                if st_id not in encoding_cache or nd_id not in encoding_cache:
                    continue

                decoded = self._decode_pair(
                    encoding_cache[st_id], encoding_cache[nd_id]
                )
                kpts_on_device = self._extract_pair(
                    decoded, device, original_sizes[st_id], original_sizes[nd_id]
                )
                if kpts_on_device is None:
                    continue

                st_kpts, nd_kpts = self._finalize_pair(kpts_on_device)
                if len(st_kpts) < self.min_pairs:
                    continue

                st_fp = str(image_repository.get_filepath(st_id))
                nd_fp = str(image_repository.get_filepath(nd_id))
                out_match[st_fp][nd_fp] = np.concatenate([st_kpts, nd_kpts], axis=1)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if self.validate:
            out_match = self._validate_matches(
                image_repository, out_match, original_sizes
            )

        return merge_matches(out_match)

    def _decode_pair(
        self,
        st_encoded: EncodedImageFeaturesDict,
        nd_encoded: EncodedImageFeaturesDict,
    ):
        decoded = self.mast3r_two_step.decode_feature_pairs(st_encoded, nd_encoded)
        decoded["st_features"]["true_shape"] = decoded["st_features"][
            "true_shape"
        ].squeeze()
        decoded["nd_features"]["true_shape"] = decoded["nd_features"][
            "true_shape"
        ].squeeze()
        return decoded

    def _extract_pair(
        self,
        decoded,
        device: torch.device,
        st_original_size: tuple[int, int],
        nd_original_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            return dense_extract(
                decoded,
                st_original_size,
                nd_original_size,
                device=device,
                match_conf_th=self.match_conf_th,
                min_pairs=self.min_pairs,
                subsample=self.subsample,
                pixel_tol=self.pixel_tol,
                max_iter=self.max_iter,
                size_param=self.image_size,
                top_k=self.top_k_matches,
                search_subsample=self.search_subsample,
            )
        except Exception:
            LOGGER.exception(
                "Mast3rFastSingleMatchPipelineStep: trouble extracting dense keypoints"
            )
            return None

    def _finalize_pair(
        self,
        kpts_on_device: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
        st_kpts = to_numpy(kpts_on_device[0])
        nd_kpts = to_numpy(kpts_on_device[1])
        if st_kpts.size == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        return st_kpts, nd_kpts

    def _validate_matches(
        self,
        image_repository: BaseImageRepository,
        out_match: dict[str, dict[str, np.ndarray]],
        original_sizes: dict[ImageId, tuple[int, int]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Geometry-verify every raw pair's matches before they are merged.

        ``out_match`` maps ``st_fp -> nd_fp -> (M, 4)`` arrays of paired
        ``[st_xy | nd_xy]`` coordinates; each pair is RANSAC-filtered and pairs
        left below ``min_pairs`` are dropped.
        """
        fp_to_size = {
            str(image_repository.get_filepath(image_id)): size
            for image_id, size in original_sizes.items()
        }
        total_pairs = sum(len(nd_map) for nd_map in out_match.values())
        LOGGER.info(
            "Mast3rFastSingleMatchPipelineStep: validation started -- %d pairs",
            total_pairs,
        )
        validated: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        with tqdm(
            total=total_pairs, desc="Mast3rFastSingleMatchPipelineStep validating"
        ) as tbar:
            for st_fp, nd_map in out_match.items():
                for nd_fp, arr in nd_map.items():
                    if arr.size == 0:
                        tbar.update(1)
                        continue
                    st_kpts = np.ascontiguousarray(arr[:, :2])
                    nd_kpts = np.ascontiguousarray(arr[:, 2:])
                    try:
                        inliers = validate_kps_matches(
                            st_kpts,
                            nd_kpts,
                            fp_to_size[st_fp],
                            fp_to_size[nd_fp],
                            max_error=self.validate_max_error,
                        )
                    except Exception:
                        LOGGER.exception(
                            "Mast3rFastSingleMatchPipelineStep: not able to validate matches"
                        )
                        tbar.update(1)
                        continue
                    kept = arr[inliers[:, 0]]
                    if len(kept) >= self.min_pairs:
                        validated[st_fp][nd_fp] = kept
                    tbar.update(1)
        validated_pairs = sum(len(nd_map) for nd_map in validated.values())
        LOGGER.info(
            "Mast3rFastSingleMatchPipelineStep: validation ended -- %d/%d pairs kept",
            validated_pairs,
            total_pairs,
        )
        return validated

    def _save_matches_and_kpts(
        self,
        keypoints_map: dict[str, np.ndarray],
        matches_map: dict[tuple[str, str], np.ndarray],
        filepath_to_id: dict[str, ImageId],
        image_repository: BaseImageRepository,
    ) -> None:
        for image_filepath, keypoints in keypoints_map.items():
            image_id = filepath_to_id[image_filepath]
            image_repository.add_keypoints(
                image_id, keypoints, name=self.keypoints_name
            )

        for (st_image_filepath, nd_image_filepath), matches in matches_map.items():
            st_image_id = filepath_to_id[st_image_filepath]
            nd_image_id = filepath_to_id[nd_image_filepath]
            image_repository.add_matches(
                st_image_id, nd_image_id, matches, name=self.matches_name
            )
            image_repository.upsert_match_metadata(
                st_image_id, nd_image_id, match_kind=MatchKind.MATCHED.value
            )
