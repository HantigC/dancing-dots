
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any

import pycolmap

from mts.core.geometry.rigid3d import Rigid3D
from mts.core.types import StateType
from mts.helpers.colmap.h5_to_db import CameraModel
from mts.pipeline.repository.export.colmap import export_to_colmap
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import BasePipelineStep, use_image_repository


class BaseColmapReconstructionStep(BasePipelineStep, ABC):
    def __init__(
        self,
        single_camera: bool = True,
        camera_model: str = CameraModel.PINHOLE,
    ) -> None:
        super().__init__()
        self.single_camera = single_camera
        self.camera_model = camera_model

    @use_image_repository(params=["state"])
    def run(
        self,
        *,
        image_repository: ImageRepository,
        state: StateType,
    ) -> None:
        colmap_dirpath = Path(state["colmap_dirpath"])
        db = export_to_colmap(
            image_repository,
            colmap_dirpath / "db",
            self.single_camera,
            self.camera_model,
        )
        maps = self._run_colmap(
            image_repository=image_repository,
            input=input,
            db=db,
            state=state,
        )
        self._save_save_to_repository(maps, image_repository)

    @abstractmethod
    def _run_colmap(
        self,
        *,
        image_repository: ImageRepository,
        input: Any | None = None,
        state: StateType | None = None,
    ) -> dict[int, pycolmap.Reconstruction]:
        pass

    def _save_save_to_repository(self, maps, image_repository: ImageRepository):
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
                image_repository.add_metadata(image_id, cluster=map_index)
