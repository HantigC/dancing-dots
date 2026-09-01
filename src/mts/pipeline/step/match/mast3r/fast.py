from __future__ import annotations

import copy
import gc
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from dust3r.utils.image import load_images
from torch import nn
from tqdm.auto import tqdm

from mts.core.matching.dense.fast_nn import dense_extract
from mts.core.matching.dense.mast3r import EncodedImageFeaturesDict, Mast3rTwoStep
from mts.core.matching.dense.merge.round import merge_matches
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.model.mast3r.io import load_model
from mts.core.scene_graph.model import MatchKind
from mts.core.types import ImageId, PathLike
from mts.helpers.torch.tensor import to, to_numpy
from mts.pipeline.repository.base import BaseImageRepository, SceneScopedImageRepository
from mts.pipeline.step.base import PerSceneStep

LOGGER = logging.getLogger(__name__)


def _rebind_landscape_heads(mast3r_model) -> None:
    """Re-point the ``head{1,2}`` landscape wrappers at this model's own heads.

    ``AsymmetricCroCo3DStereo.set_downstream_head`` stores ``head1``/``head2`` as
    plain closures produced by ``transpose_to_landscape(downstream_head{1,2})``.
    ``copy.deepcopy`` copies functions by reference, so a cloned replica's
    wrappers keep calling the *source* model's heads (and therefore its device).
    Rebuilding the wrappers after a deepcopy makes each replica self-contained.
    """
    from dust3r.utils.misc import transpose_to_landscape

    for i in (1, 2):
        head_module = getattr(mast3r_model, f"downstream_head{i}", None)
        if head_module is None:
            continue
        wrapper = getattr(mast3r_model, f"head{i}", None)
        activate = getattr(wrapper, "__name__", "wrapper_yes") == "wrapper_yes"
        setattr(
            mast3r_model,
            f"head{i}",
            transpose_to_landscape(head_module, activate=activate),
        )


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


class Mast3rFastMatchPipelineStep(PerSceneStep):
    """Per-scene, multi-GPU MASt3R matcher (encode-once / decode-per-pair).

    Runs the MASt3R encoder once per image (on the first device), caches the
    encodings, then for every pair already stored in the repository runs only
    the decoder + head followed by the all-torch dense reciprocal-NN match
    extraction from :mod:`mts.core.matching.dense.fast_nn`.

    A single ``Mast3rTwoStep`` is passed in and cloned to one replica per
    entry in ``devices`` (e.g. ``["cuda:0", "cuda:1"]``); one encoding-cache
    copy is held per device. The pair list is walked ``len(devices)`` at a
    time: every replica's ``decode_feature_pairs`` for the window is launched
    back-to-back (async on CUDA, so the GPUs run concurrently), then the dense
    match extraction is enqueued for each, and only afterwards are the results
    pulled to the host; each CUDA device is synchronised once at the very end.
    With a single device this is a plain sequential loop. Because the step
    pins each replica to its device itself, **do not wrap it in
    ``OnDeviceRunner``** -- pass ``devices`` in the config.

    Writes per-image keypoints under ``keypoints_name`` and ``(M, 2)`` index
    matches under ``matches_name`` (both default to ``"mast3r"``), so it is a
    drop-in for ``ColmapReconstructionStep``. Requires a prior pairing step
    (consumes ``image_repository.get_pairs()``); needs no prior
    keypoint-extraction step.
    """

    def __init__(
        self,
        mast3r_two_step: Mast3rTwoStep | list[Mast3rTwoStep],
        devices: list[str] | str | None = None,
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
    ) -> None:
        super().__init__()
        if devices is None:
            devices = ["cpu"]
        elif isinstance(devices, str):
            devices = [devices]
        else:
            devices = list(devices)

        self._devices = [torch.device(d) for d in devices]

        # one replica per device. A list of models (one already built per
        # device, e.g. from ``from_checkpoint``) is used as-is; a single model
        # is cloned for the extra devices. ``copy.deepcopy`` leaves the DPT
        # ``head{1,2}`` landscape wrappers bound to the source model, so rebind
        # them on every clone -- otherwise the head runs on the wrong device.
        if isinstance(mast3r_two_step, (list, tuple)):
            replicas = list(mast3r_two_step)
            if len(replicas) != len(self._devices):
                raise ValueError(
                    f"got {len(replicas)} models for {len(self._devices)} devices"
                )
        else:
            replicas = [mast3r_two_step]
            for _ in range(len(self._devices) - 1):
                clone = copy.deepcopy(mast3r_two_step)
                _rebind_landscape_heads(clone.mast3r_model)
                replicas.append(clone)
        self.replicas = nn.ModuleList(replicas)
        for replica, device in zip(self.replicas, self._devices):
            replica.to(device)

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

    @property
    def device(self) -> torch.device:
        return self._devices[0]

    def to(self, device=None, *args, **kwargs):
        # Each replica is pinned to its own device in __init__; a blanket
        # device move would collapse a multi-device run onto one device, so
        # ignore the device here (dtype/other args still apply per replica).
        if device is not None:
            LOGGER.debug(
                "Mast3rFastMatchPipelineStep.to(%s) ignored; replicas stay on %s",
                device,
                [str(d) for d in self._devices],
            )
        for replica in self.replicas:
            nn.Module.to(replica, *args, **kwargs)
        return self

    @classmethod
    def from_checkpoint(
        cls,
        mast3r_model_checkpoint: PathLike,
        devices: list[str] | str = ("cuda:0",),
        **kwargs,
    ) -> "Mast3rFastMatchPipelineStep":
        if devices is None:
            device_list = ["cpu"]
        elif isinstance(devices, str):
            device_list = [devices]
        else:
            device_list = list(devices)

        # Load a fresh model straight onto each device instead of deepcopying
        # one -- a deepcopy keeps the DPT head wrappers bound to the source
        # model's device (see ``_rebind_landscape_heads``).
        two_steps = [
            Mast3rTwoStep(
                load_model(mast3r_model_checkpoint, device=torch.device(device))
            )
            for device in device_list
        ]
        return cls(two_steps, devices=device_list, **kwargs)

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
                "Mast3rFastMatchPipelineStep: scene '%s' has no pairs, skipping", scene
            )
            return

        LOGGER.info(
            "Mast3rFastMatchPipelineStep: scene '%s' -- %d pairs across %d device(s) %s",
            scene,
            len(pairs),
            len(self._devices),
            [str(d) for d in self._devices],
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
        # one cache copy per device (each pinned to that device). ``true_shape``
        # is left on the host: the decoder head reads it via ``.cpu()`` and the
        # dense extractor moves it itself, so keeping it off-GPU avoids needless
        # transfers and ``.numpy()`` surprises downstream.
        device_caches = [
            {
                iid: {
                    key: (value if key == "true_shape" else value.to(device))
                    for key, value in enc.items()
                }
                for iid, enc in encoding_cache.items()
            }
            for device in self._devices
        ]
        try:
            keypoints_map, matches_map = self._compute_matches(
                image_repository, pairs, device_caches, original_sizes
            )
            self._save_matches_and_kpts(
                keypoints_map, matches_map, filepath_to_id, image_repository
            )
        finally:
            encoding_cache.clear()
            for cache in device_caches:
                cache.clear()
            gc.collect()

    def _encode_images(self, image_repository: BaseImageRepository) -> EncodingCache:
        LOGGER.info("Mast3rFastMatchPipelineStep: encoding images...")
        image_ids = list(image_repository.image_ids())
        filepaths = [
            str(image_repository.get_filepath(image_id)) for image_id in image_ids
        ]
        images = load_images(filepaths, size=self.image_size, verbose=False)

        replica = self.replicas[0]
        device = self._devices[0]
        replica.eval()

        encoding_cache: EncodingCache = {}
        with torch.inference_mode():
            for image_id, image in zip(image_ids, tqdm(images, desc="Encoding images")):
                encoded = self._encode_image(replica, device, image)
                if encoded is None:
                    continue
                encoding_cache[image_id] = to(encoded, device=torch.device("cpu"))
        return encoding_cache

    def _encode_image(
        self, replica: Mast3rTwoStep, device: torch.device, image: dict[str, Any]
    ) -> EncodedImageFeaturesDict | None:
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
            encoded_image_dict = replica.encode_image(
                {"image": img, "true_shape": true_shape}
            )
        except Exception:
            LOGGER.exception(
                "Mast3rFastMatchPipelineStep: could not encode image with MASt3R"
            )
            return None

        return EncodedImageFeaturesDict.from_add_shape(encoded_image_dict, true_shape)

    def _compute_matches(
        self,
        image_repository: BaseImageRepository,
        pairs: list[tuple[ImageId, ImageId]],
        device_caches: list[EncodingCache],
        original_sizes: dict[ImageId, tuple[int, int]],
    ) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]:
        out_match: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        n = len(self._devices)
        for replica in self.replicas:
            replica.eval()

        with torch.inference_mode():
            with tqdm(
                total=len(pairs), desc="Mast3rFastMatchPipelineStep matching"
            ) as tbar:
                for i in range(0, len(pairs), n):
                    window = pairs[i : i + n]

                    # 1. launch every replica's decoder for this window back to
                    #    back -- on CUDA these return immediately, so the GPUs
                    #    work concurrently.
                    decoded = [None] * len(window)
                    for slot, (st_id, nd_id) in enumerate(window):
                        cache = device_caches[slot]
                        if st_id not in cache or nd_id not in cache:
                            continue
                        decoded[slot] = self._decode_pair(
                            self.replicas[slot],
                            cache[st_id],
                            cache[nd_id],
                        )

                    # 2. enqueue the dense match extraction for each slot
                    #    (still device-resident, kernels not yet awaited).
                    kpts_on_device = [None] * len(window)
                    for slot, (st_id, nd_id) in enumerate(window):
                        if decoded[slot] is None:
                            continue
                        kpts_on_device[slot] = self._extract_pair(
                            decoded[slot],
                            self._devices[slot],
                            original_sizes[st_id],
                            original_sizes[nd_id],
                        )

                    # 3. pull to host, validate, collect.
                    for slot, (st_id, nd_id) in enumerate(window):
                        if kpts_on_device[slot] is None:
                            continue
                        st_kpts, nd_kpts = self._finalize_pair(
                            kpts_on_device[slot],
                            original_sizes[st_id],
                            original_sizes[nd_id],
                        )
                        if len(st_kpts) < self.min_pairs:
                            continue
                        st_fp = str(image_repository.get_filepath(st_id))
                        nd_fp = str(image_repository.get_filepath(nd_id))
                        out_match[st_fp][nd_fp] = np.concatenate(
                            [st_kpts, nd_kpts], axis=1
                        )

                    tbar.update(len(window))

        # sync once at the end, not per pair
        for device in self._devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        return merge_matches(out_match)

    def _decode_pair(
        self,
        replica: Mast3rTwoStep,
        st_encoded: EncodedImageFeaturesDict,
        nd_encoded: EncodedImageFeaturesDict,
    ):
        # ``st_encoded`` / ``nd_encoded`` already live on ``replica``'s device
        # (one cache copy per device, built in ``run_scene``).
        decoded = replica.decode_feature_pairs(st_encoded, nd_encoded)
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
                top_k=self.top_k_matches,
            )
        except Exception:
            LOGGER.exception(
                "Mast3rFastMatchPipelineStep: trouble extracting dense keypoints"
            )
            return None

    def _finalize_pair(
        self,
        kpts_on_device: tuple[torch.Tensor, torch.Tensor],
        st_original_size: tuple[int, int],
        nd_original_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        st_kpts = to_numpy(kpts_on_device[0])
        nd_kpts = to_numpy(kpts_on_device[1])
        if st_kpts.size == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        if self.validate:
            try:
                inliers = validate_kps_matches(
                    st_kpts, nd_kpts, st_original_size, nd_original_size
                )
            except Exception:
                LOGGER.exception(
                    "Mast3rFastMatchPipelineStep: not able to validate matches"
                )
                return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
            st_kpts = st_kpts[inliers[:, 0]]
            nd_kpts = nd_kpts[inliers[:, 1]]

        return st_kpts, nd_kpts

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
