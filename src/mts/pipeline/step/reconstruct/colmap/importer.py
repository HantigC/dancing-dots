from __future__ import annotations

from mts.core.types import StateType
from mts.helpers.colmap.database import COLMAPDatabase
from mts.helpers.colmap.h5_to_db import CameraModel
from mts.pipeline.repository.export.colmap import export_to_colmap
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import BasePipelineStep, use_image_repository


class ExportToColmapStep(BasePipelineStep):
    def __init__(
        self,
        single_camera: bool = False,
        camera_model: str = CameraModel.PINHOLE,
    ) -> None:
        super().__init__()
        self.single_camera = single_camera
        self.camera_model = camera_model

    @use_image_repository(params=["state"])
    def run(
        self, image_repository: ImageRepository, state: StateType
    ) -> COLMAPDatabase:
        db = export_to_colmap(
            image_repository,
            state["colmap_db_filepath"],
            self.single_camera,
            self.camera_model,
        )
        return db
