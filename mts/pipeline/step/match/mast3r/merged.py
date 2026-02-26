
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from mast3r.model import AsymmetricMASt3R

from mts.core.matcher.dense.mast3r import match_pairs
from mts.core.model.mast3r.io import load_model
from mts.core.repository.base import BaseImageRepository
from mts.core.scene_graph.model import Image, MatchKind, TwoViewEdge
from mts.pipeline.step.extract.kp.base import BasePipelineStep


class Mast3rMatchPipelineStep(BasePipelineStep):
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

    def run(
        self,
        *,
        image_repository: BaseImageRepository,
        input: Any = None,
        state: dict[str, Any] = None,
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
            image_repository.add_keypoints(image_id, keypoints)

        for (st_image_filepath, nd_image_filepath), matches in matches_map.items():
            st_image_id = image_repository.get_image_id(Path(st_image_filepath))
            nd_image_id = image_repository.get_image_id(Path(nd_image_filepath))
            image_repository.add_matches(st_image_id, nd_image_id, matches)

    def create_mts_graph(self, image_repository: BaseImageRepository):
        scene_graph = nx.Graph().to_undirected()
        for image_id, image in image_repository.iterate_over_images():
            height, width = image.shape[:2]
            image = Image(
                height=height,
                width=width,
            )
            scene_graph.add_node(
                image_repository.get_filepath(image_id),
                image=image,
            )

        for st_filepath, matched_filepaths_map in matches_map.items():
            for nd_filepath, kpts in matched_filepaths_map.items():
                st_kpts, nd_kpts = np.split(kpts, 2,axis=1)
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
                        num_matches=len(kpts)
                    ),
                    weight=len(kpts),
                )

    @classmethod
    def from_checkpoint(
        cls, mast3r_model_checkpoint: str, **kwargs
    ) -> Mast3rMatchPipelineStep:
        mast3r_model = load_model(mast3r_model_checkpoint, torch.device("cpu"))
        return cls(mast3r_model, **kwargs)
