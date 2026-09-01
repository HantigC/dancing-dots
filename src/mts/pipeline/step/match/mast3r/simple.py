from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from mast3r.model import AsymmetricMASt3R

from mts.core.matching.dense.mast3r import match_pairs
from mts.pipeline.repository.base import SceneScopedImageRepository
from mts.core.model.mast3r.io import load_model
from mts.pipeline.step.base import PerSceneStep


class Mast3rMatchPipelineStep(PerSceneStep):
    def __init__(
        self,
        mast3r_model: AsymmetricMASt3R,
        min_pairs: int = 15,
        match_conf_th: float = 1.001,
    ) -> None:
        super().__init__()
        self.mast3r_model = mast3r_model
        self.min_pairs = min_pairs
        self.match_conf_th = match_conf_th

    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> Any:
        images_filepaths = [
            str(image_filepath) for image_filepath in image_repository.image_filepaths()
        ]

        filename_to_idx = {
            image_filepath: idx for idx, image_filepath in enumerate(images_filepaths)
        }

        indexed_pairs = [
            (
                filename_to_idx[str(image_repository.get_filepath(st_image_id))],
                filename_to_idx[str(image_repository.get_filepath(nd_image_id))],
            )
            for st_image_id, nd_image_id in image_repository.get_pairs()
        ]
        keypoints_map, matches_map = match_pairs(
            self.mast3r_model,
            indexed_pairs,
            images_filepaths,
            self.min_pairs,
            match_conf_th=self.match_conf_th,
            device=self.device,
        )
        for image_filepath, keypoints in keypoints_map.items():
            image_id = image_repository.get_image_id(Path(image_filepath))
            image_repository.add_keypoints(image_id, keypoints, name="mast3r")

        for (st_image_filepath, nd_image_filepath), matches in matches_map.items():
            st_image_id = image_repository.get_image_id(Path(st_image_filepath))
            nd_image_id = image_repository.get_image_id(Path(nd_image_filepath))
            image_repository.add_matches(st_image_id, nd_image_id, matches, name="mast3r")

    @classmethod
    def from_checkpoint(
        cls, mast3r_model_checkpoint: str, **kwargs
    ) -> Mast3rMatchPipelineStep:
        mast3r_model = load_model(mast3r_model_checkpoint, torch.device("cpu"))
        return cls(mast3r_model, **kwargs)
