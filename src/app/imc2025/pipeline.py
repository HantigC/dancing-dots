from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tqdm.auto import tqdm

from mts.core.types import PathLike, StateType
from mts.pipeline.repository import h5 as h5_repo
from mts.pipeline.repository import inmemeory as mem_repo
from mts.pipeline.repository.base import BaseImageRepository
from mts.pipeline.step.base import from_hydra_config, run_pipeline
from mts.pipeline.step.extract.kp.base import BasePipelineStep

from .prediction import Prediction

LOGGER = logging.getLogger(__name__)


ALL = "all"


@dataclass
class IMC2025Pipeline:
    project_dirpath: PathLike
    samples: dict[str, list[Prediction]]
    create_repository: Callable[[str, IMC2025Pipeline], BaseImageRepository]
    create_pipeline: Callable[[str, BaseImageRepository], BasePipelineStep]
    create_pipeline_input: (
        Callable[[IMC2025Pipeline, BaseImageRepository, str], Any] | None
    ) = field(default=None)
    create_pipeline_state: (
        Callable[[IMC2025Pipeline, BaseImageRepository, str], StateType] | None
    ) = field(default=None)
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
        for dataset_name in datasets_names:
            self.run_for(dataset_name)
        elapsed = time.monotonic() - t0
        LOGGER.info("Pipeline finished in %.2f seconds (%.2f minutes)", elapsed, elapsed / 60)

    def run_for(self, dataset_name: str) -> None:
        dataset_samples = self.samples[dataset_name]
        t0 = time.monotonic()

        image_repository = self.create_repository(dataset_name, self)
        with image_repository:
            self._repositories_map[dataset_name] = image_repository
            pipeline = self.create_pipeline(dataset_name, image_repository)

            LOGGER.info("Add `%s` to image_repository", dataset_name)
            for sample in tqdm(dataset_samples, desc="Add images to repository"):
                image_repository.add_image(sample.image_filepath)
            LOGGER.info("Starting the pipeline for `%s`", dataset_name)

            input = None
            if self.create_pipeline_input is not None:
                input = self.create_pipeline_input(self, image_repository, dataset_name)

            state = {}
            if self.create_pipeline_state is not None:
                state = self.create_pipeline_state(self, image_repository, dataset_name)

            run_pipeline(
                pipeline,
                image_repository=image_repository,
                input=input,
                state=state,
            )
            self._collect(image_repository, dataset_samples)
        elapsed = time.monotonic() - t0
        LOGGER.info("`%s` finished in %.2f seconds (%.2f minutes)", dataset_name, elapsed, elapsed / 60)

    def _collect(
        self,
        repository: BaseImageRepository,
        dataset_samples: list[Prediction],
    ) -> None:
        filename_to_prediction = {
            str(prediction.image_filepath): prediction for prediction in dataset_samples
        }
        how_many_clusters = 0
        for image_id in repository.image_ids():
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


def create_inmemory_repository(
    dataset_name: str,
    imc2025_pipeline: IMC2025Pipeline,
) -> mem_repo.ImageRepository:
    image_repository = mem_repo.ImageRepository()
    image_repository.add_repository_metadata(dataset_name=dataset_name)
    return image_repository


def create_h5_repository(
    dataset_name: str,
    imc2025_pipeline: IMC2025Pipeline,
) -> h5_repo.H5ImageRepository:
    dataset_filepath = (
        imc2025_pipeline.project_dirpath / "h5_repositories" 
    )
    dataset_filepath.mkdir(exist_ok=True)
    image_repository = h5_repo.H5ImageRepository.from_filename(
        dataset_filepath,
        f"{dataset_name}.h5",
    )
    image_repository.add_repository_metadata(dataset_name=dataset_name)
    return image_repository


def create_pipeline(
    cfg,
    image_repository: BaseImageRepository,
    dataset_name: str,
) -> list[BasePipelineStep]:
    pipeline_steps = from_hydra_config(cfg)
    return pipeline_steps


def create_pipeline_state(
    imc2025_pipeline: IMC2025Pipeline,
    image_repository: BaseImageRepository,
    dataset_name: str,
) -> StateType:
    dataset_dirpath = Path(imc2025_pipeline.project_dirpath) / dataset_name
    dataset_dirpath.mkdir(exist_ok=True)
    state = {
        "images_dir": ".",
        "colmap_dirpath": dataset_dirpath,
    }
    return state
