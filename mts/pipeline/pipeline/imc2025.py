from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mts.core.types import PathLike, StateType
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import run_pipeline
from mts.pipeline.step.extract.kp.base import BasePipelineStep

LOGGER = logging.getLogger(__name__)


@dataclass
class Prediction:
    image_id: str | None
    dataset: str
    filename: str
    image_filepath: Path
    cluster_index: int | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None


ALL = "all"


@dataclass
class IMC2025Pipeline:
    project_dirpath: PathLike
    samples: dict[str, list[Prediction]]
    create_repository: Callable[[str], ImageRepository]
    create_pipeline: Callable[[str, ImageRepository], BasePipelineStep]
    create_pipeline_input: Callable[[IMC2025Pipeline, ImageRepository, str], Any] | None = field(default=None)
    create_pipeline_state: Callable[[IMC2025Pipeline, ImageRepository, str], StateType] | None = field(default=None)

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
        for dataset_name in datasets_names:
            self.run_for(dataset_name)

    def run_for(self, dataset_name: str) -> None:
        dataset_samples = self.samples[dataset_name]
        image_repository = self.create_repository(dataset_name)
        pipeline = self.create_pipeline(dataset_name, image_repository)

        LOGGER.info("Add {dataset_name} to image_repository")
        for sample in tqdm(dataset_samples):
            image_repository.add_image(sample.image_filepath)
        LOGGER.info("Starting the pipeline for `{dataset_name}`")


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
        LOGGER.info("Ending the pipeline for `{dataset_name}`")
        self._collect(image_repository, dataset_samples)

    def _collect(
        self,
        repository: ImageRepository,
        dataset_samples: list[Prediction],
    ) -> None:
        filename_to_prediction = {
            prediction.image_filepath: prediction for prediction in dataset_samples
        }
        for image_id in repository.image_ids():
            metadata: dict[str, str] = repository.get_metadata(image_id)
            cluster = metadata.get("cluster")
            pose = repository.get_pose(image_id)
            prediction = filename_to_prediction[repository.get_filepath(image_id)]
            prediction.cluster_index = cluster
            if pose is not None:
                prediction.rotation = pose.rotation
                prediction.translation = pose.translation

    @classmethod
    def from_samples_df(
        cls,
        df: pd.DataFrame,
        data_dirpath: PathLike,
        pipeline: BasePipelineStep,
    ) -> IMC2025Pipeline:
        data_dirpath = Path(data_dirpath)
        samples = {}
        for _, row in df.iterrows():
            if row.dataset not in samples:
                samples[row.dataset] = []
            samples[row.dataset].append(
                Prediction(
                    image_id=row.image_id,
                    dataset=row.dataset,
                    filename=row.image,
                    image_filepath=data_dirpath / row.dataset / row.image,
                )
            )
        return cls(samples, pipeline)

    @classmethod
    def from_test_dir(
        cls,
        data_dirpath: PathLike,
        pipeline: BasePipelineStep,
        filename: str = "sample_submission.csv",
    ) -> IMC2025Pipeline:
        data_dirpath = Path(data_dirpath)
        sample_submission_csv = data_dirpath / filename
        df = pd.read_csv(sample_submission_csv)
        return cls.from_samples_df(df, data_dirpath, pipeline)

    @classmethod
    def from_train_dir(
        cls,
        data_dirpath: PathLike,
        pipeline: BasePipelineStep,
        filename: str = "train_labels.csv",
    ) -> IMC2025Pipeline:
        data_dirpath = Path(data_dirpath)
        sample_submission_csv = data_dirpath / filename
        df = pd.read_csv(sample_submission_csv)
        df["image_id"] = df.dataset + "_" + df.image
        return cls.from_samples_df(df, data_dirpath, pipeline)
