from pathlib import Path
import pycolmap
from mts.core.types import StateType
from mts.helpers.colmap.database import COLMAPDatabase
from mts.pipeline.step.base import use_params
from mts.pipeline.step.reconstruct.colmap.base import BaseColmapReconstructionStep


class ColmapReconstructionStep(BaseColmapReconstructionStep):
    def __init__(self, mapper_options: pycolmap.IncrementalMapperOptions, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mapper_options = mapper_options

    @use_params(params=["state", "db"])
    def _run_colmap(
        self,
        state: StateType,
        db: COLMAPDatabase,
    ) -> None:
        images_dirpath = Path(state["images_dir"])
        colmap_dirpath = Path(state["colmap_dirpath"])
        pycolmap.match_exhaustive(db.database)
        colmap_dirpath.mkdir(exist_ok=True)
        maps = pycolmap.incremental_mapping(
            database_path=db.database,
            image_path=images_dirpath,
            output_path=colmap_dirpath,
            options=self.mapper_options,
        )
        return maps