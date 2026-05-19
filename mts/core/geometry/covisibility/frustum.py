from typing import Callable, NamedTuple

import numpy as np
import trimesh

from mts.core.geometry.covisibility.plane_sweep import img_size_to_rectangle
from mts.core.geometry.rigid3d import as_4x4_Rt
from mts.core.types import Rigid3dDict, PairType


def frustum_corners(
    rigid3d_dict: Rigid3dDict,
    k: np.ndarray,
    image_size: tuple[int, int],
    depth_range: tuple[float, float],
) -> np.ndarray:
    near, far = depth_range
    rectangle = img_size_to_rectangle(image_size)  # 4x3, homogeneous image coords

    k_inv = np.linalg.inv(k)
    cam_dirs = (k_inv @ rectangle.T).T  # 4x3, directions in camera space

    near_pts = cam_dirs * near  # 4x3
    far_pts = cam_dirs * far  # 4x3
    cam_corners = np.vstack([near_pts, far_pts])  # 8x3

    Rt = as_4x4_Rt(rigid3d_dict["R"], rigid3d_dict["t"])
    Rt_inv = np.linalg.inv(Rt)
    cam_corners_h = np.hstack([cam_corners, np.ones((8, 1))])
    world_corners_h = (Rt_inv @ cam_corners_h.T).T
    return world_corners_h[:, :3]


def frustum_mesh(
    rigid3d_dict: Rigid3dDict,
    k: np.ndarray,
    image_size: tuple[int, int],
    depth_range: tuple[float, float],
) -> trimesh.Trimesh:
    corners = frustum_corners(rigid3d_dict, k, image_size, depth_range)
    return trimesh.PointCloud(corners).convex_hull


def frustum_intersection_covisibility(
    st_rigid3d_dict: Rigid3dDict,
    st_k: np.ndarray,
    st_image_size: tuple[int, int],
    nd_rigid3d_dict: Rigid3dDict,
    nd_k: np.ndarray,
    nd_image_size: tuple[int, int],
    depth_range: tuple[float, float],
    engine: str = "manifold",
) -> float:
    mesh_a = frustum_mesh(st_rigid3d_dict, st_k, st_image_size, depth_range)
    mesh_b = frustum_mesh(nd_rigid3d_dict, nd_k, nd_image_size, depth_range)

    try:
        intersection = trimesh.boolean.intersection([mesh_a, mesh_b], engine=engine)
    except Exception:
        return 0.0

    if intersection is None or not hasattr(intersection, "volume"):
        return 0.0

    vol = intersection.volume
    if vol <= 0:
        return 0.0
    denominator = min(mesh_a.volume, mesh_b.volume)
    if denominator == 0:
        return 0
    score = vol / denominator
    return float(np.clip(score, 0.0, 1.0))


def compute_intersection(
    st_rigid3d_dict: Rigid3dDict,
    st_k: np.ndarray,
    st_image_size: tuple[int, int],
    nd_rigid3d_dict: Rigid3dDict,
    nd_k: np.ndarray,
    nd_image_size: tuple[int, int],
    depth_range: tuple[float, float],
    engine: str = "manifold",
) -> trimesh.Trimesh | None:
    mesh_a = frustum_mesh(st_rigid3d_dict, st_k, st_image_size, depth_range)
    mesh_b = frustum_mesh(nd_rigid3d_dict, nd_k, nd_image_size, depth_range)
    try:
        intersection = trimesh.boolean.intersection([mesh_a, mesh_b], engine=engine)
    except Exception:
        return None
    return intersection


def camera_center(rigid3d_dict: Rigid3dDict) -> np.ndarray:
    return -rigid3d_dict["R"].T @ rigid3d_dict["t"]


def _angle_from_intersection(
    intersection: trimesh.Trimesh,
    st_rigid3d_dict: Rigid3dDict,
    nd_rigid3d_dict: Rigid3dDict,
) -> float:
    centroid = intersection.centroid
    v_st = camera_center(st_rigid3d_dict) - centroid
    v_nd = camera_center(nd_rigid3d_dict) - centroid
    cos_angle = np.dot(v_st, v_nd) / (np.linalg.norm(v_st) * np.linalg.norm(v_nd))
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _covisibility_from_intersection(
    intersection: trimesh.Trimesh,
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
) -> float:
    vol = intersection.volume
    if vol <= 0:
        return 0.0
    denominator = min(mesh_a.volume, mesh_b.volume)
    if denominator == 0:
        return 0.0
    return float(np.clip(vol / denominator, 0.0, 1.0))


def frustum_intersection_angle(
    st_rigid3d_dict: Rigid3dDict,
    st_k: np.ndarray,
    st_image_size: tuple[int, int],
    nd_rigid3d_dict: Rigid3dDict,
    nd_k: np.ndarray,
    nd_image_size: tuple[int, int],
    depth_range: tuple[float, float],
    engine: str = "manifold",
) -> float | None:
    intersection = compute_intersection(
        st_rigid3d_dict, st_k, st_image_size,
        nd_rigid3d_dict, nd_k, nd_image_size,
        depth_range, engine,
    )
    if intersection is None or not hasattr(intersection, "centroid"):
        return None
    return _angle_from_intersection(intersection, st_rigid3d_dict, nd_rigid3d_dict)


class FrustumPairMetrics(NamedTuple):
    covisibility: float
    angle: float  # radians


def make_all_pairs(n: int) -> list[PairType[int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]
