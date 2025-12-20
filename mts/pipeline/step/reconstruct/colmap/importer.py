from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from mts.core.types import PathLike
from mts.helpers.colmap.database import COLMAPDatabase
from mts.helpers.colmap.h5_to_db import CameraModel, create_camera
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import BasePipelineStep, use_no_params


class ColmapImportStep(BasePipelineStep):
    def __init__(
        self,
        db: COLMAPDatabase,
        repository: ImageRepository,
        single_camera: bool = False,
        camera_model: str = CameraModel.PINHOLE,
    ) -> None:
        super().__init__()
        self.db = db
        self.repository = repository
        self.single_camera = single_camera
        self.camera_model = camera_model

    @use_no_params
    def run(
        self,
    ) -> None:
        self.db.create_tables()
        try:
            id_to_db_id = self._add_keypoints()
            self._add_matches(id_to_db_id)
            self.db.commit()
        except BaseException:
            raise
        finally:
            self.db.close()

    def _add_matches(self, id_to_db_id: dict[int, int]) -> None:
        for from_idx, to_idx in tqdm(
            self.repository.get_pairs(),
            total=self.repository.pair_num(),
            desc="Add matches",
        ):
            from_db_id = id_to_db_id[from_idx]
            to_db_id = id_to_db_id[to_idx]
            matches = self.repository.get_matches(from_idx, to_idx)
            if matches is not None:
                self.db.add_matches(from_db_id, to_db_id, matches)

    def _add_keypoints(self) -> dict[int, int]:
        camera_id = None
        id_to_db_id = {}
        for image_id in tqdm(
            self.repository.image_ids(),
            total=self.repository.images_num(),
            desc="Add Keypoints",
        ):
            keypoints = self.repository.get_keypoints(image_id)
            descriptors = self.repository.get_descriptors(image_id)
            image_filepath = self.repository.get_filepath(image_id)

            if camera_id is None or not self.single_camera:
                camera_id = create_camera(
                    self.db,
                    str(image_filepath),
                    self.camera_model,
                )
            db_image_id = self.db.add_image(str(image_filepath), camera_id)
            self.db.add_keypoints(db_image_id, keypoints)
            self.db.add_descriptors(image_id, descriptors)
            id_to_db_id[image_id] = db_image_id

        return id_to_db_id

    @classmethod
    def from_db_filepath(
        cls, db_filepath: PathLike, image_repository: ImageRepository, **kwargs
    ) -> ColmapImportStep:
        db = COLMAPDatabase(db_filepath)
        return cls(db, image_repository, **kwargs)

    @classmethod
    def from_directory(
        cls,
        dirpath: PathLike,
        image_repository: ImageRepository,
        filename: str = "reconstruction.db",
        **kwargs,
    ) -> ColmapImportStep:
        dirpath = Path(dirpath)
        db_filepath = dirpath / filename
        db = COLMAPDatabase(db_filepath)
        return cls(db, image_repository, **kwargs)
