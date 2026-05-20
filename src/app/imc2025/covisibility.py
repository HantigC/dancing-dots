from typing import Any, Protocol

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from app.imc2025.samples import GTSample
from mts.core.geometry.covisibility.frustum import frustum_intersection_covisibility
from mts.core.types import Rigid3dDict
from mts.helpers.colmap.h5_to_db import get_focal


class CovisibilityFn(Protocol):
    def __call__(
        self,
        st_rigid3d_dict: Rigid3dDict,
        st_k: np.ndarray,
        st_image_size: tuple[int, int],
        nd_rigid3d_dict: Rigid3dDict,
        nd_k: np.ndarray,
        nd_image_size: tuple[int, int],
        **kwargs,
    ) -> float: ...


def compute_k(sample: GTSample) -> np.ndarray:
    image_filepath = sample.sample.image_filepath
    focal = get_focal(image_filepath)
    w, h = Image.open(image_filepath).size
    return np.array(
        [
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1],
        ]
    )


def get_image_size(sample: GTSample) -> tuple[int, int]:
    return Image.open(sample.sample.image_filepath).size  # (w, h)


def make_pairs(samples: list) -> list[tuple]:
    n = len(samples)
    pairs = [
        (
            samples[i],
            samples[j],
        )
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return pairs


def gt_samples_covisibility(
    samples: list[GTSample],
    pairs: list[tuple[int, int]] | None = None,
    tqdm_kwargs: dict[str, Any] | None = None,
    covisibility_fn: CovisibilityFn = frustum_intersection_covisibility,
    covisibility_fn_kwargs: dict[str, Any] | None = None,
) -> dict[tuple[str, str], float]:
    tqdm_kwargs = {} if tqdm_kwargs is None else tqdm_kwargs
    covisibility_fn_kwargs = {} if covisibility_fn_kwargs is None else covisibility_fn_kwargs

    if pairs is None:
        n = len(samples)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    k_matrices = [compute_k(s) for s in samples]
    image_sizes = [get_image_size(s) for s in samples]

    result: dict[tuple[str, str], float] = {}
    for i, j in tqdm(pairs, **tqdm_kwargs):
        score = covisibility_fn(
            samples[i].gt_pose,
            k_matrices[i],
            image_sizes[i],
            samples[j].gt_pose,
            k_matrices[j],
            image_sizes[j],
            **covisibility_fn_kwargs,
        )
        result[i, j] = score

    return result
