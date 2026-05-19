
import pandas as pd

from mts.helpers.imc.metric import read_csv, tth_from_csv
from mts.pipeline.repository import h5 as h5_repo
from mts.pipeline.step.pair.sift import extract_sift_features

image_repository = h5_repo.H5ImageRepository("iterations/0037/h5_repositories/ETs.h5")
submission_filepath = "iterations/0037/submission.csv"
train_labels_filepath = "data/image-matching-challenge-2025/train_labels.csv"

submission_samples = read_csv(submission_filepath)
train_samples = read_csv(train_labels_filepath)

from dataclasses import dataclass

from app.imc2025.prediction import Prediction
from mts.core.types import Rigid3dDict


@dataclass
class GTSample:
    sample: Prediction
    gt_pose: Rigid3dDict
    scene_name: str

from app.imc2025.prediction import load_from_csv

samples = load_from_csv("data/image-matching-challenge-2025", "train_labels.csv")
samples_map = {sample.filename: sample for sample in samples["ETs"]}

gt_samples = []
for scene_name, scene_poses in train_samples["ETs"].items():
    for image_filename, pose_dict in scene_poses.items():
        gt_samples.append(
            GTSample(
                samples_map[image_filename],
                pose_dict,
                scene_name,
            )
        )

from mts.utils.iterate import group_by

per_scene_gt_samples = group_by(gt_samples, key=lambda x: x.scene_name)
image_repository.get_repository_metadata("dataset_name")

et_gt_samples = per_scene_gt_samples["ET"]

st_image_id = 0
nd_image_id = 7

st_gt_sample = et_gt_samples[st_image_id]
nd_gt_sample = et_gt_samples[nd_image_id]

from typing import Literal, Sequence

import numpy as np
from mts.helpers.colmap.h5_to_db import get_focal
from mts.pipeline.repository.base import BaseImageRepository


def ket_K(
    image_repository: BaseImageRepository,
    gt_sample: GTSample,
    focal_length: float | None = None,
) -> np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float32]]:
    image_filepath = str(gt_sample.sample.image_filepath)
    image = image_repository.load_image(image_repository.get_image_id(image_filepath))
    if focal_length is None:
        focal_length = get_focal(image_filepath)
    h, w = image.shape[:2]
    K = np.array(
        [
            [focal_length, 0, w // 2],
            [0, focal_length, h // 2],
            [0, 0, 1],
        ]
    )
    return K

(kpts, descriptors), *_ = extract_sift_features(
    [
        image_repository.load_image(st_image_id),
    ],
    num_features=1024,
)

h, w = image_repository.load_image(
    image_repository.get_image_id(str(st_gt_sample.sample.image_filepath)),
).shape[:2]

st_K = ket_K(image_repository, st_gt_sample)
nd_K = ket_K(image_repository, nd_gt_sample)

st_K_inv = np.linalg.inv(st_K)
nd_K_inv = np.linalg.inv(nd_K)

st_T = np.eye(4)
st_T[:3, :3] = st_gt_sample.gt_pose["R"]
st_T[:3, 3] = st_gt_sample.gt_pose["t"]

nd_T = np.eye(4)
nd_T[:3, :3] = nd_gt_sample.gt_pose["R"]
nd_T[:3, 3] = nd_gt_sample.gt_pose["t"]


in_camera_kpts = (
    st_K_inv @ np.concatenate([kpts, np.ones((len(kpts), 1))], axis=1).T
).T


in_camera_kpts_at_depths = (
    in_camera_kpts[:, np.newaxis] * np.arange(1, 10, 0.1)[np.newaxis, ..., np.newaxis]
)

in_camera_kpts_at_depths_h = np.concatenate(
    [
        in_camera_kpts_at_depths,
        np.ones((*in_camera_kpts_at_depths.shape[:2], 1)),
    ],
    axis=2,
)

st_T_inv = np.linalg.inv(st_T)

in_world_kpts_at_depths_h = (
    st_T_inv @ in_camera_kpts_at_depths_h.transpose(0, 2, 1)
).transpose(0, 2, 1)
in_world_kpts_at_depths = in_world_kpts_at_depths_h[..., :-1]

in_nd_camera_kpts_at_depths_h = (
    nd_T @ in_world_kpts_at_depths_h.transpose(0, 2, 1)
).transpose(0, 2, 1)
in_nd_camera_kpts_at_depths = in_nd_camera_kpts_at_depths_h[..., :-1]


projectd_nd_camera_kpts_at_depths = (
    nd_K @ in_nd_camera_kpts_at_depths.transpose(0, 2, 1)
).transpose(0, 2, 1)

pixel_points_nd_image = (
    projectd_nd_camera_kpts_at_depths / projectd_nd_camera_kpts_at_depths[..., -1:]
)
pixel_points_nd_image = pixel_points_nd_image.astype(np.int32)
pixel_points_nd_image = pixel_points_nd_image[..., :-1]

import cv2

flatten_kpts_nd_image = pixel_points_nd_image.reshape((-1, 2))

nd_image = image_repository.load_image(nd_image_id)
height, width = nd_image.shape[:2]

mask = (flatten_kpts_nd_image >= (0, 0)).all(axis=1) & (
    flatten_kpts_nd_image <= (height, width)
).all(axis=1)

(indices,) = np.where(mask)

masked_kpts = flatten_kpts_nd_image[mask]

sift = cv2.SIFT_create()

keypoints = [
    cv2.KeyPoint(
        x=x,
        y=y,
        size=31,
    )
    for x, y in masked_kpts.astype(np.float32)
]

gray_nd = cv2.cvtColor(nd_image, cv2.COLOR_RGB2GRAY) if nd_image.ndim == 3 else nd_image
sift.compute(gray_nd, keypoints)