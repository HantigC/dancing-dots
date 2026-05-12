from typing import Any, Iterable, Literal

import numpy as np
from tqdm.auto import tqdm

from mts.core.geometry.covisibility.plane_sweep import covisibility
from mts.core.geometry.rigid3d import Rigid3D
from mts.core.types import ImageId, PairType
from mts.helpers.colmap.h5_to_db import get_focal
from mts.pipeline.repository.base import BaseImageRepository


def compute_k(
    image_repository: BaseImageRepository,
    image_id: ImageId,
    focal_length: float | None = None,
) -> np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float32]]:
    image = image_repository.load_image(image_id)
    image_filepath = image_repository.get_filepath(image_id)
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


def get_image_size(
    repository: BaseImageRepository,
    image_id: ImageId,
) -> PairType[int]:
    image = repository.load_image(image_id)
    h, w = image.shape[:2]
    return w, h


def exhaustive_image_pairs(
    repository: BaseImageRepository,
) -> list[PairType[int]]:
    ids = [
        img_id
        for img_id in repository.image_ids()
        if repository.get_pose(img_id) is not None
    ]
    pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    return pairs


def repository_covisibility(
    repository: BaseImageRepository,
    depth_range: tuple[float, float],
    no_of_depths: int,
    pairs: Iterable[PairType[ImageId]] | None = None,
    tqdm_kwargs: dict[str, Any] = None,
) -> dict[PairType[ImageId], float]:
    tqdm_kwargs = {} if tqdm_kwargs is None else tqdm_kwargs
    if pairs is None:
        pairs = exhaustive_image_pairs(repository)

    image_sizes: dict[ImageId, tuple[int, int]] = {
        image_id: get_image_size(
            repository,
            image_id,
        )
        for image_id in repository.image_ids()
    }

    k_matrices: dict[ImageId, np.ndarray] = {
        image_id: compute_k(
            repository,
            image_id,
        )
        for image_id in repository.image_ids()
    }
    result: dict[PairType[ImageId], float] = {}
    for st_id, nd_id in tqdm(pairs, **tqdm_kwargs):
        st_pose: Rigid3D = repository.get_pose(st_id)
        nd_pose: Rigid3D = repository.get_pose(nd_id)
        if st_pose is None or nd_pose is None:
            continue

        score = covisibility(
            st_pose.as_rigid3d_dict(),
            k_matrices[st_id],
            nd_pose.as_rigid3d_dict(),
            k_matrices[nd_id],
            image_sizes[st_id],
            image_sizes[nd_id],
            depth_range,
            no_of_depths,
        )
        result[(st_id, nd_id)] = score

    return result
