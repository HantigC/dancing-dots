import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pycolmap

from mts.core.geometry.rigid3d import Rigid3D
from mts.core.types import StateType
from mts.helpers.colmap.database import COLMAPDatabase
from mts.helpers.colmap.h5_to_db import CameraModel
from mts.pipeline.repository.base import AlreadyExistsException, BaseImageRepository
from mts.pipeline.repository.export.colmap import export_to_colmap
from mts.pipeline.step.base import BasePipelineStep, use_image_repository

LOGGER = logging.getLogger(__name__)


def save_to_repository(
    maps: dict[int, pycolmap.Reconstruction],
    image_repository: BaseImageRepository,
) -> None:
    for map_index, cur_map in maps.items():
        for index, image in cur_map.images.items():
            rigid3d = image.cam_from_world()
            image_id = image_repository.get_image_id(image.name)
            image_repository.add_pose(
                image_id,
                Rigid3D(
                    deepcopy(rigid3d.rotation.matrix()),
                    deepcopy(rigid3d.translation),
                ),
            )
            existing_metadata = image_repository.get_metadata(image_id) or {}
            match_kind = existing_metadata.get("match_kind")
            try:
                image_repository.add_metadata(
                    image_id, cluster=map_index, match_kind=match_kind
                )
            except AlreadyExistsException:
                LOGGER.warning("`%d` already has 'cluster' metadata", image_id)
                image_repository.update_metadata(
                    image_id, cluster=map_index, match_kind=match_kind
                )


SaveToRepoCallable = Callable[
    [
        dict[int, pycolmap.Reconstruction],
        BaseImageRepository,
    ],
    None,
]


def save_sorted_to_repository(
    reconstructions_map: dict[int, pycolmap.Reconstruction],
    image_repository: BaseImageRepository,
) -> None:
    sorted_reconstructions = sorted(
        reconstructions_map.items(),
        key=lambda x: x[1].num_reg_frames(),
    )
    sorted_reconstructions = dict(sorted_reconstructions)
    for map_index, cur_map in sorted_reconstructions.items():
        for index, image in cur_map.images.items():
            rigid3d = image.cam_from_world()
            image_id = image_repository.get_image_id(image.name)
            image_repository.add_pose(
                image_id,
                Rigid3D(
                    deepcopy(rigid3d.rotation.matrix()),
                    deepcopy(rigid3d.translation),
                ),
            )
            existing_metadata = image_repository.get_metadata(image_id) or {}
            match_kind = existing_metadata.get("match_kind")
            try:
                image_repository.add_metadata(
                    image_id, cluster=map_index, match_kind=match_kind
                )
            except AlreadyExistsException:
                LOGGER.warning("`%d` already has 'cluster' metadata", image_id)
                image_repository.update_metadata(
                    image_id, cluster=map_index, match_kind=match_kind
                )


class BaseColmapReconstructionStep(BasePipelineStep, ABC):
    def __init__(
        self,
        save_to_repository: SaveToRepoCallable | None = None,
        single_camera: bool = True,
        camera_model: str = CameraModel.PINHOLE,
        keypoints_name: str = "keypoints",
        descriptors_name: str = "descriptors",
        matches_name: str = "matches",
    ) -> None:
        super().__init__()
        self.single_camera = single_camera
        self.camera_model = camera_model
        self.keypoints_name = keypoints_name
        self.descriptors_name = descriptors_name
        self.matches_name = matches_name
        if save_to_repository is None:
            save_to_repository = save_to_repository
        self._save_to_repository = save_to_repository

    @use_image_repository(params=["state"])
    def run(
        self,
        *,
        image_repository: BaseImageRepository,
        state: StateType,
    ) -> None:
        colmap_dirpath = Path(state["colmap_dirpath"])
        colmap_db_filepath = colmap_dirpath / "colmap.db"
        if not colmap_db_filepath.exists():
            db = export_to_colmap(
                image_repository,
                colmap_db_filepath,
                self.single_camera,
                self.camera_model,
                keypoints_name=self.keypoints_name,
                descriptors_name=self.descriptors_name,
                matches_name=self.matches_name,
            )
        else:
            db = COLMAPDatabase.connect(colmap_db_filepath)
        maps = self._run_colmap(
            image_repository=image_repository,
            input=input,
            db=db,
            state=state,
        )
        self._save_to_repository(maps, image_repository)

    @abstractmethod
    def _run_colmap(
        self,
        *,
        image_repository: BaseImageRepository,
        input: Any | None = None,
        state: StateType | None = None,
    ) -> dict[int, pycolmap.Reconstruction]:
        pass
