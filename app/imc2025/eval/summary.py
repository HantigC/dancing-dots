from typing import TypedDict

import numpy as np

from mts.core.types import Rigid3dDict
from mts.helpers.imc.metric import mAA_on_cameras, register_by_Horn


class CameraParams(TypedDict):
    R: np.ndarray
    t: np.ndarray
    c: np.ndarray


def to_camera_centers(
    cams_dicts: dict[str, CameraParams],
    images_names: list[str],
) -> np.ndarray:

    return np.array(
        [cams_dicts[image_name]["c"] for image_name in images_names],
    ).T


SampleMap = dict[str, dict[str, dict[str, CameraParams]]]
DatasetMap = dict[str, dict[str, CameraParams]]


class EvalMicroSummary(TypedDict):
    model_table: list[list[np.ndarray]]
    err_table: list[list[float]]
    mAA_table: np.ndarray
    cluster_table: np.ndarray
    gt_scene_sum_table: np.ndarray
    user_scene_sum_table: np.ndarray


class BestEvalAlignment(TypedDict):
    gt_scenes: list[int]
    user_scenes: list[int]
    best_models: list[np.ndarray]


class EvalSummary(TypedDict):
    eval_micro_summary: EvalMicroSummary
    best_alignment: BestEvalAlignment


def compute_best_alignment(
    submission_dataset: DatasetMap,
    train_dataset: DatasetMap,
    eval_micro_summary: EvalMicroSummary,
) -> BestEvalAlignment:
    best_gt_scenes = []
    best_user_scenes = []
    best_models = []
    user_scene_list = list(submission_dataset)
    mAA_table = eval_micro_summary["mAA_table"]
    cluster_table = eval_micro_summary["cluster_table"]
    model_table = eval_micro_summary["model_table"]

    for i, gt_scene in enumerate(train_dataset.keys()):
        best_ind = np.lexsort((-mAA_table[i], -cluster_table[i]))[0]
        best_gt_scenes.append(gt_scene)
        best_user_scenes.append(user_scene_list[best_ind])
        best_models.append(model_table[i][best_ind])

    return {
        "best_models": best_models,
        "gt_scenes": best_gt_scenes,
        "user_scenes": best_user_scenes,
    }


def compute_eval_summary(
    train_samples: SampleMap,
    submission_samples: SampleMap,
    thresholds_map: dict[str, dict[str, np.ndarray]],
    dataset_names: str | list[str] | None = None,
    skip_top_thresholds: int = 2,
    to_dec: int = 3,
):
    if dataset_names is None:
        dataset_names = list(train_samples.key())

    eval_map: dict[str, EvalSummary] = {}
    for dataset_name in dataset_names:
        train_dataset = train_samples[dataset_name]
        submission_dataset = submission_samples[dataset_name]
        lg = len(train_dataset)
        lu = len(submission_dataset)

        # full table
        model_table = []
        err_table = []
        mAA_table = np.full((lg, lu), -1).astype(float)
        cluster_table = np.full((lg, lu), -1).astype(int)
        gt_scene_sum_table = np.full((lg, lu), -1).astype(np.float64)
        user_scene_sum_table = np.full((lg, lu), -1).astype(np.float64)

        for i, (train_scene_name, train_scene_cams) in enumerate(train_dataset.items()):
            err_row = []
            model_row = []

            for j, (sub_scene_name, sub_scene_cams) in enumerate(
                submission_dataset.items()
            ):
                if train_scene_name == "outliers":
                    err_row.append(100)
                    mAA_table[i, j] = 0
                    cluster_table[i, j] = 0
                    gt_scene_sum_table[i, j] = len(train_scene_cams)
                    user_scene_sum_table[i, j] = len(sub_scene_cams)
                    model_row.append(None)
                    continue
                thresholds = thresholds_map[dataset_name][train_scene_name]
                good_cams = []

                for image_path in train_scene_cams.keys():
                    if image_path in sub_scene_cams.keys():
                        good_cams.append(image_path)

                model = register_by_Horn(
                    to_camera_centers(sub_scene_cams, good_cams),
                    to_camera_centers(train_scene_cams, good_cams),
                    thresholds,
                    0,
                    -1,
                )

                mAA = mAA_on_cameras(
                    model["err"],
                    thresholds,
                    len(train_scene_cams),
                    skip_top_thresholds,
                    to_dec,
                )

                err_row.append(model["err"])
                mAA_table[i, j] = mAA
                cluster_table[i, j] = len(good_cams)
                gt_scene_sum_table[i, j] = len(train_scene_cams)
                user_scene_sum_table[i, j] = len(sub_scene_cams)
                model_row.append(model)

            model_table.append(model_row)
            err_table.append(err_row)

        eval_micro_summary = {
            "cluster_table": cluster_table,
            "err_table": err_table,
            "gt_scene_sum_table": gt_scene_sum_table,
            "mAA_table": mAA_table,
            "model_table": model_table,
            "user_scene_sum_table": user_scene_sum_table,
        }

        best_alignment = compute_best_alignment(
            submission_dataset,
            train_dataset,
            eval_micro_summary,
        )

        eval_map[dataset_name] = {
            "eval_micro_summary": eval_micro_summary,
            "best_alignment": best_alignment,
        }
    return eval_map


def align_poses(
    dataset_name: str,
    scene_idx: int,
    submission_samples: SampleMap,
    best_alignment: BestEvalAlignment,
    nth_model: int = 0,
) -> list[Rigid3dDict]:
    transf_matrix = best_alignment["best_models"][scene_idx]["transf_matrix"][nth_model]

    rotated_poses = []
    best_user_scene_name = best_alignment["user_scenes"][scene_idx]
    for pose_dict in submission_samples[dataset_name][best_user_scene_name].values():
        new_R = (transf_matrix[:3, :3] @ pose_dict["R"].T).T
        new_t = -new_R @ (transf_matrix[:3, :3] @ pose_dict["c"] + transf_matrix[:3, 3])
        new_pose = {
            "R": new_R,
            "t": new_t,
        }
        rotated_poses.append(new_pose)
    return rotated_poses
