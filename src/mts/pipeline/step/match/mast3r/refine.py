from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mast3r.model import AsymmetricMASt3R

from mts.core.matching.dense.merge.round import merge_matches
from mts.core.model.mast3r.io import load_model
from mts.core.model.mast3r.matching import fine_match
from mts.pipeline.repository.base import SceneScopedImageRepository
from mts.pipeline.step.base import PerSceneStep

LOGGER = logging.getLogger(__name__)


class Mast3rRefineMatchPipelineStep(PerSceneStep):
    """Refine an existing set of matches with MASt3R coarse-to-fine matching.

    Reads the keypoints + matches produced by a previous matching step
    (``source_keypoints_name`` / ``source_matches_name``), reconstructs the
    matched pixel coordinates for every registered pair, and feeds them to
    :func:`mts.core.model.mast3r.matching.fine_match` as seeds. ``fine_match``
    runs only the crop-selection + fine-refinement half of MASt3R
    coarse-to-fine: the seeds pick which overlapping crop pairs to refine, and
    a fresh, higher-resolution dense match is computed inside those crops.

    The refined correspondences (which do *not* contain the input seeds) are
    de-duplicated back into per-image keypoints + index matches with
    :func:`merge_matches` and written under ``target_keypoints_name`` /
    ``target_matches_name`` so downstream steps (e.g. COLMAP reconstruction)
    can point at the refined name.
    """

    def __init__(
        self,
        mast3r_model: AsymmetricMASt3R,
        source_keypoints_name: str = "mast3r",
        source_matches_name: str = "mast3r",
        target_keypoints_name: str = "mast3r_refined",
        target_matches_name: str = "mast3r_refined",
        min_pairs: int = 15,
        max_image_size: int | None = None,
        pixel_tol: int = 5,
        subsample: int = 8,
        max_batch_size: int = 48,
        overlap: float = 0.5,
        conf_thr: float | None = None,
        bucket_size: int = 1,
        round_digits: int = 1,
    ) -> None:
        super().__init__()
        self.mast3r_model = mast3r_model
        self.source_keypoints_name = source_keypoints_name
        self.source_matches_name = source_matches_name
        self.target_keypoints_name = target_keypoints_name
        self.target_matches_name = target_matches_name
        self.min_pairs = min_pairs
        self.max_image_size = max_image_size
        self.pixel_tol = pixel_tol
        self.subsample = subsample
        self.max_batch_size = max_batch_size
        self.overlap = overlap
        self.conf_thr = conf_thr
        self.bucket_size = bucket_size
        self.round_digits = round_digits

    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> Any:
        pairs = image_repository.get_pairs()
        LOGGER.info(
            "Refining '%s' matches over %d pairs (scene '%s')",
            self.source_matches_name,
            len(pairs),
            scene,
        )

        out_match: dict[str, dict[str, np.ndarray]] = {}
        refined_pairs = 0
        for st_image_id, nd_image_id in pairs:
            seed_matches = image_repository.get_matches(
                st_image_id, nd_image_id, name=self.source_matches_name
            )
            if seed_matches is None or len(seed_matches) < self.min_pairs:
                continue

            st_keypoints = image_repository.get_keypoints(
                st_image_id, name=self.source_keypoints_name
            )
            nd_keypoints = image_repository.get_keypoints(
                nd_image_id, name=self.source_keypoints_name
            )
            if st_keypoints is None or nd_keypoints is None:
                continue

            matches_im0 = st_keypoints[seed_matches[:, 0]]
            matches_im1 = nd_keypoints[seed_matches[:, 1]]

            st_filepath = str(image_repository.get_filepath(st_image_id))
            nd_filepath = str(image_repository.get_filepath(nd_image_id))

            with torch.inference_mode():
                refined = fine_match(
                    st_filepath,
                    nd_filepath,
                    matches_im0,
                    matches_im1,
                    self.mast3r_model,
                    device=self.device,
                    max_image_size=self.max_image_size,
                    pixel_tol=self.pixel_tol,
                    subsample=self.subsample,
                    max_batch_size=self.max_batch_size,
                    overlap=self.overlap,
                    conf_thr=self.conf_thr,
                )
            gc.collect()

            if len(refined) < self.min_pairs:
                continue

            out_match.setdefault(st_filepath, {})[nd_filepath] = np.concatenate(
                [refined.matches_im0, refined.matches_im1], axis=1
            )
            refined_pairs += 1

        if not out_match:
            LOGGER.warning(
                "No pairs refined for scene '%s' -- nothing written under '%s'",
                scene,
                self.target_matches_name,
            )
            return

        global_keypoints, global_matches = merge_matches(
            out_match,
            bucket_size=self.bucket_size,
            round_digits=self.round_digits,
        )

        for image_filepath, keypoints in global_keypoints.items():
            image_id = image_repository.get_image_id(Path(image_filepath))
            image_repository.add_keypoints(
                image_id, keypoints, name=self.target_keypoints_name
            )

        for (st_filepath, nd_filepath), matches in global_matches.items():
            st_image_id = image_repository.get_image_id(Path(st_filepath))
            nd_image_id = image_repository.get_image_id(Path(nd_filepath))
            image_repository.add_matches(
                st_image_id, nd_image_id, matches, name=self.target_matches_name
            )

        LOGGER.info(
            "Refined %d/%d pairs for scene '%s' -> '%s' (%d images, %d match pairs)",
            refined_pairs,
            len(pairs),
            scene,
            self.target_matches_name,
            len(global_keypoints),
            len(global_matches),
        )

    @classmethod
    def from_checkpoint(
        cls, mast3r_model_checkpoint: str, **kwargs
    ) -> Mast3rRefineMatchPipelineStep:
        mast3r_model = load_model(mast3r_model_checkpoint, torch.device("cpu"))
        return cls(mast3r_model, **kwargs)
