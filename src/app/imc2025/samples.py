import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]='1'

from dataclasses import dataclass
from pathlib import Path

from app.imc2025.prediction import Prediction, load_from_csv
from mts.core.types import Rigid3dDict
from mts.helpers.imc.metric import read_csv
from mts.utils.iterate import group_by


@dataclass
class GTSample:
    sample: Prediction
    gt_pose: Rigid3dDict
    scene_name: str


def load_gt_samples(
    data_filepath: str | Path,
    train_labels_filename: str = "train_labels.csv",
) -> dict[str, dict[str, list[GTSample]]]:
    train_labels_filepath = Path(data_filepath) / train_labels_filename
    samples, _ = load_from_csv(data_filepath, train_labels_filename)
    train_samples = read_csv(train_labels_filepath)
    gt_samples_map = {}
    for dataset_name, dataset in samples.items():
        samples_map = {sample.filename: sample for sample in dataset}

        gt_samples = []

        for scene_name, scene_poses in train_samples[dataset_name].items():
            for image_filename, pose_dict in scene_poses.items():
                gt_samples.append(
                    GTSample(
                        samples_map[image_filename],
                        pose_dict,
                        scene_name,
                    )
                )

        per_scene_gt_samples = group_by(gt_samples, key=lambda x: x.scene_name)
        gt_samples_map[dataset_name] = per_scene_gt_samples
    return gt_samples_map
