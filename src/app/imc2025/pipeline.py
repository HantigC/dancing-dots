from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


from mts.core.types import PathLike, StateType
from mts.pipeline.repository import h5 as h5_repo
from mts.pipeline.repository import inmemeory as mem_repo
from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.base import (
    BasePipelineStep,
    from_hydra_config,
    run_pipeline,
)

from .prediction import Prediction

LOGGER = logging.getLogger(__name__)


ALL = "all"


@dataclass
class IMC2025Pipeline:
    project_dirpath: PathLike
    samples: dict[str, list[Prediction]]
    create_repository: Callable[[IMC2025Pipeline], BaseImageRepository]
    create_pipeline: Callable[[Any, BaseImageRepository], list[BasePipelineStep]]
    create_pipeline_input: (
        Callable[[IMC2025Pipeline, BaseImageRepository], Any] | None
    ) = field(default=None)
    create_pipeline_state: (
        Callable[[IMC2025Pipeline, BaseImageRepository], StateType] | None
    ) = field(default=None)
    delete_repo: bool = field(default=False)
    _last_cluster_index: int = field(default=0)
    _repositories_map: dict[str, BaseImageRepository] = field(default_factory=dict)

    def _get_dataset_names(self, datasets_names: list[str] | str = ALL) -> list[str]:
        if isinstance(datasets_names, str) and datasets_names == ALL:
            datasets_names = list(self.samples.keys())
        elif not isinstance(datasets_names, list):
            raise ValueError(
                f"`dataset_names` should be either '{ALL}' or a list of str "
            )
        return datasets_names

    def run(
        self,
        datasets_names: list[str] | str = ALL,
    ):
        datasets_names = self._get_dataset_names(datasets_names)
        t0 = time.monotonic()

        image_repository = self.create_repository(self)
        with image_repository:
            for dataset_name in datasets_names:
                self._repositories_map[dataset_name] = image_repository
            pipeline = self.create_pipeline(image_repository)

            for dataset_name in datasets_names:
                dataset_samples = self.samples[dataset_name]
                LOGGER.info(
                    "Adding %d images for scene '%s'",
                    len(dataset_samples),
                    dataset_name,
                )
                image_repository.add_images(
                    [str(sample.image_filepath) for sample in dataset_samples],
                    scene=dataset_name,
                )

            input = None
            if self.create_pipeline_input is not None:
                input = self.create_pipeline_input(self, image_repository)

            state = {}
            if self.create_pipeline_state is not None:
                state = self.create_pipeline_state(self, image_repository)

            LOGGER.info("Starting the pipeline for %d scene(s)", len(datasets_names))
            try:
                run_pipeline(
                    pipeline,
                    image_repository=image_repository,
                    input=input,
                    state=state,
                )
            except Exception:
                LOGGER.exception(
                    "Pipeline aborted; collecting whatever completed so far"
                )

            for dataset_name in datasets_names:
                self._collect(image_repository, dataset_name)

        if self.delete_repo:
            image_repository.delete_repo()

        elapsed = time.monotonic() - t0
        LOGGER.info(
            "Pipeline finished in %.2f seconds (%.2f minutes)", elapsed, elapsed / 60
        )

    def _collect(
        self,
        repository: BaseImageRepository,
        dataset_name: str,
    ) -> None:
        dataset_samples = self.samples[dataset_name]
        filename_to_prediction = {
            str(prediction.image_filepath): prediction for prediction in dataset_samples
        }
        how_many_clusters = 0
        for image_id in repository.image_ids(scene=dataset_name):
            metadata: dict[str, str] | None = repository.get_metadata(image_id)
            cluster_index = None
            if metadata is not None:
                cluster_index = metadata.get("cluster")
            pose = repository.get_pose(image_id)
            prediction = filename_to_prediction[repository.get_filepath(image_id)]
            if pose is not None:
                prediction.rotation = pose.rotation
                prediction.translation = pose.translation

            if cluster_index is not None:
                how_many_clusters = max(how_many_clusters, cluster_index)
                prediction.cluster_index = cluster_index + self._last_cluster_index
        self._last_cluster_index += how_many_clusters + 1


def _repository_filename(imc2025_pipeline: IMC2025Pipeline) -> str:
    """One shared file for a normal multi-scene run; a per-dataset file when
    the pipeline was handed exactly one dataset (the distributed worker
    path), so concurrent workers sharing an iteration dir don't collide."""
    samples = imc2025_pipeline.samples
    if len(samples) == 1:
        return f"{next(iter(samples))}.h5"
    return "repository.h5"


def create_inmemory_repository(
    imc2025_pipeline: IMC2025Pipeline,
) -> mem_repo.ImageRepository:
    image_repository = mem_repo.ImageRepository()
    image_repository.upsert_repository_metadata(
        dataset_names=list(imc2025_pipeline.samples)
    )
    return image_repository


def create_h5_repository(
    imc2025_pipeline: IMC2025Pipeline,
    delete_on_exit: bool = False,
) -> h5_repo.H5ImageRepository:
    h5_dirpath = Path(imc2025_pipeline.project_dirpath) / "h5_repositories"
    h5_dirpath.mkdir(parents=True, exist_ok=True)
    image_repository = h5_repo.H5ImageRepository.from_filename(
        h5_dirpath,
        _repository_filename(imc2025_pipeline),
        delete_on_exit=delete_on_exit,
    )
    image_repository.upsert_repository_metadata(
        dataset_names=list(imc2025_pipeline.samples)
    )
    return image_repository


def create_pipeline(
    cfg,
    image_repository: BaseImageRepository,
) -> list[BasePipelineStep]:
    pipeline_steps = from_hydra_config(cfg)
    return pipeline_steps


def create_pipeline_state(
    imc2025_pipeline: IMC2025Pipeline,
    image_repository: BaseImageRepository,
) -> StateType:
    project_dirpath = Path(imc2025_pipeline.project_dirpath)
    project_dirpath.mkdir(parents=True, exist_ok=True)
    state = {
        "images_dir": ".",
        "colmap_dirpath": project_dirpath,
    }
    return state
