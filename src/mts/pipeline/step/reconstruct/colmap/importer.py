from __future__ import annotations

from pathlib import Path
from typing import Any

from mts.core.types import StateType
from mts.helpers.colmap.database import COLMAPDatabase
from mts.helpers.colmap.h5_to_db import CameraModel
from mts.pipeline.repository.base import SceneScopedImageRepository
from mts.pipeline.repository.export.colmap import export_to_colmap
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import PerSceneStep


class ExportToColmapStep(PerSceneStep):
    def __init__(
        self,
        single_camera: bool = False,
        camera_model: str = CameraModel.PINHOLE,
    ) -> None:
        super().__init__()
        self.single_camera = single_camera
        self.camera_model = camera_model

    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> COLMAPDatabase:
        db_filepath = Path(state["colmap_db_filepath"])
        db_filepath = db_filepath.with_name(
            f"{db_filepath.stem}_{scene}{db_filepath.suffix}"
        )
        db = export_to_colmap(
            image_repository,
            db_filepath,
            self.single_camera,
            self.camera_model,
        )
        return db
