import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mts.core.types import PathLike

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


DatasetSamples = dict[str, list[Prediction]]


def _array_to_str(array: np.ndarray) -> str:
    return ";".join([f"{x:.09f}" for x in array])


def _none_to_str(n: int) -> str:
    return ";".join(["nan"] * n)


def to_df(samples: DatasetSamples) -> pd.DataFrame:
    records = []
    for dataset in samples.values():
        for prediction in dataset:
            cluster_name = (
                "outliers"
                if prediction.cluster_index is None
                else f"cluster{prediction.cluster_index}"
            )
            rotation = (
                _none_to_str(9)
                if prediction.rotation is None
                else _array_to_str(prediction.rotation.flatten())
            )
            translation = (
                _none_to_str(3)
                if prediction.translation is None
                else _array_to_str(prediction.translation)
            )
            record = {
                "image_id": prediction.image_id,
                "dataset": prediction.dataset,
                "scene": cluster_name,
                "image": prediction.filename,
                "rotation_matrix": rotation,
                "translation_vector": translation,
            }
            records.append(record)
    df = pd.DataFrame.from_records(records)
    return df


def load_from_df(
    df: pd.DataFrame,
    data_dirpath: PathLike,
    skip: bool = True,
    strict: bool = True,
    load_pose: bool = False,
) -> DatasetSamples:
    data_dirpath: Path = Path(data_dirpath)
    samples = {}
    for _, row in df.iterrows():
        if row.dataset not in samples:
            samples[row.dataset] = []
        image_filepath = data_dirpath / row.dataset / row.image
        if not image_filepath.exists():
            if skip:
                continue
            else:
                if strict:
                    raise ValueError("File '%s' could not be found", image_filepath)
        rotation, translation = None, None
        if load_pose:
            rotation = np.array(list(map(float, row.rotation_matrix.split(";")))).reshape((3, 3))
            translation = np.array(list(map(float, row.translation_vector.split(";"))))

        samples[row.dataset].append(
            Prediction(
                image_id=row.image_id,
                dataset=row.dataset,
                filename=row.image,
                image_filepath=image_filepath,
                rotation=rotation, 
                translation=translation,
            )
        )
    samples = {
        dataset_name: predictions
        for dataset_name, predictions in samples.items()
        if len(predictions) > 0
    }
    return samples


def load_from_train(
    data_dirpath: PathLike,
    filename: str = "train_labels.csv",
    skip: bool = True,
    strict: bool = True,
    load_pose: bool = False,
) -> tuple[DatasetSamples, pd.DataFrame]:
    data_dirpath = Path(data_dirpath)
    sample_submission_csv = data_dirpath / filename
    df = pd.read_csv(sample_submission_csv)
    df["image_id"] = df.dataset + "_" + df.image
    return load_from_df(
        df,
        data_dirpath / "train",
        skip=skip,
        strict=strict,
        load_pose=load_pose,
    ), df


def load_from_submission(
    data_dirpath: PathLike,
    filename: str = "sample_submission.csv",
    skip: bool = True,
    strict: bool = True,
    load_pose: bool = False,
) -> tuple[DatasetSamples, pd.DataFrame]:
    data_dirpath = Path(data_dirpath)
    sample_submission_csv = data_dirpath / filename
    df = pd.read_csv(sample_submission_csv)
    return load_from_df(
        df,
        data_dirpath / "test",
        skip=skip,
        strict=strict,
        load_pose=load_pose,
    ), df


def load_from_csv(
    data_dirpath: PathLike,
    filename: str,
    skip: bool = True,
    strict: bool = True,
    load_pose: bool = False,
) -> tuple[DatasetSamples, pd.DataFrame]:
    data_dirpath = Path(data_dirpath)
    csv_filepath = data_dirpath / filename
    df = pd.read_csv(csv_filepath)
    train_or_test = "test"
    if "image_id" not in df.columns:
        df["image_id"] = df.dataset + "_" + df.image
        train_or_test = "train"
    return load_from_df(
        df,
        data_dirpath / train_or_test,
        skip=skip,
        strict=strict,
        load_pose=load_pose,
    ), df


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_test_images(
    data_dirpath: PathLike,
    test_subdir: str = "test",
) -> DatasetSamples:
    data_dirpath = Path(data_dirpath)
    test_dirpath = data_dirpath / test_subdir
    samples: DatasetSamples = {}
    for dataset_dir in sorted(test_dirpath.iterdir()):
        if not dataset_dir.is_dir():
            continue
        predictions = [
            Prediction(
                image_id=f"{dataset_dir.name}_{f.name}",
                dataset=dataset_dir.name,
                filename=f.name,
                image_filepath=f,
            )
            for f in sorted(dataset_dir.iterdir())
            if f.suffix.lower() in IMAGE_SUFFIXES
        ]
        if predictions:
            samples[dataset_dir.name] = predictions
    return samples


def append_samples_to_csv(
    samples: DatasetSamples,
    df: pd.DataFrame,
    filepath: PathLike,
) -> None:
    samples_df = to_df(samples)
    df = df.copy()
    df["rotation_matrix"] = _none_to_str(9)
    df["translation_vector"] = _none_to_str(3)
    submission_df = (
        samples_df.set_index("image_id")
        .combine_first(df.set_index("image_id"))
        .reset_index()
    )
    LOGGER.info("Saving submission to %s", str(filepath))
    submission_df.to_csv(
        filepath,
        sep=",",
        index=False,
    )


def sample_to_csv(
    samples: DatasetSamples,
    filepath: PathLike,
) -> None:
    to_df(samples).to_csv(
        filepath,
        sep=",",
        index=False,
    )
