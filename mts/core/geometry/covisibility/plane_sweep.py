from typing import NamedTuple, Sequence

import numpy as np
from shapely import GEOSException, Polygon

from mts.core.geometry.rigid3d import as_4x4_Rt
from mts.core.types import Rigid3dDict, PairType


class _PlaneSweep:
    def camera(
        self,
        ref_k: np.ndarray,
        ref_rectangle: np.ndarray,
        depth_range: tuple[float, float] | None = None,
        no_of_depths: int | None = None,
        depths: Sequence[float] | None = None,
    ) -> np.ndarray:
        depths = self._init_depths(depth_range, no_of_depths, depths)
        ref_k_inv = np.linalg.inv(ref_k)

        in_ref_camera_rectangle = (ref_k_inv @ ref_rectangle.T).T

        in_ref_camera_rectangles = (
            in_ref_camera_rectangle[np.newaxis] * depths[..., np.newaxis, np.newaxis]
        )
        return in_ref_camera_rectangles

    def projected(
        self,
        ref_rigid3d_dict: Rigid3dDict,
        ref_k: np.ndarray,
        ref_rectangle: np.ndarray,
        dest_rigid3d_dict: Rigid3dDict,
        dest_k: np.ndarray,
        depth_range: tuple[float, float] | None = None,
        no_of_depths: int | None = None,
        depths: Sequence[float] | None = None,
    ) -> np.ndarray:
        depths = self._init_depths(depth_range, no_of_depths, depths)

        dest_camera_rectangle = self(
            ref_rigid3d_dict,
            ref_k,
            ref_rectangle,
            dest_rigid3d_dict,
            depth_range,
            no_of_depths,
            depths=depths,
        )
        dest_camera_projected_rectangle = (
            dest_k @ dest_camera_rectangle.transpose(0, 2, 1)
        ).transpose(0, 2, 1)
        dest_camera_pixels = (
            dest_camera_projected_rectangle[..., :-1]
            / dest_camera_projected_rectangle[..., -1:]
        )
        return dest_camera_pixels

    def _init_depths(
        self,
        depth_range: tuple[float, float] | None = None,
        no_of_depths: int | None = None,
        depths: Sequence[float] | None = None,
    ) -> np.ndarray:

        if depths is None:
            if no_of_depths is None or depth_range is None:
                raise ValueError(
                    "either (depth_range, no_of_depth) of depths should be None, no both"
                )
            lo_depth, hi_depth = depth_range

            depths = np.linspace(lo_depth, hi_depth, no_of_depths)

        return depths

    def world(
        self,
        ref_rigid3d_dict: Rigid3dDict,
        ref_k: np.ndarray,
        ref_rectangle: np.ndarray,
        depth_range: tuple[float, float] | None = None,
        no_of_depths: int | None = None,
        depths: Sequence[float] | None = None,
    ) -> np.ndarray:
        depths = self._init_depths(depth_range, no_of_depths, depths)

        ref_k_inv = np.linalg.inv(ref_k)

        in_ref_camera_rectangle = (ref_k_inv @ ref_rectangle.T).T
        in_ref_camera_rectangles = (
            in_ref_camera_rectangle[np.newaxis] * depths[..., np.newaxis, np.newaxis]
        )

        in_ref_camera_rectangles_h = np.concatenate(
            [
                in_ref_camera_rectangles,
                np.ones((*in_ref_camera_rectangles.shape[:2], 1)),
            ],
            axis=2,
        )
        ref_Rt = as_4x4_Rt(ref_rigid3d_dict["R"], ref_rigid3d_dict["t"])
        ref_Rt_inv = np.linalg.inv(ref_Rt)

        world_rectangles_h = (
            ref_Rt_inv @ in_ref_camera_rectangles_h.transpose(0, 2, 1)
        ).transpose(0, 2, 1)
        world_rectangles = world_rectangles_h[..., :-1]
        return world_rectangles

    def __call__(
        self,
        ref_rigid3d_dict: Rigid3dDict,
        ref_k: np.ndarray,
        ref_rectangle: np.ndarray,
        dest_rigid3d_dict: Rigid3dDict,
        depth_range: tuple[float, float],
        no_of_depths: int,
        depths: Sequence[float] | None = None,
    ) -> np.ndarray:
        depths = self._init_depths(depth_range, no_of_depths, depths)
        ref_k_inv = np.linalg.inv(ref_k)

        in_ref_camera_rectangle = (ref_k_inv @ ref_rectangle.T).T
        in_ref_camera_rectangles = (
            in_ref_camera_rectangle[np.newaxis] * depths[..., np.newaxis, np.newaxis]
        )

        in_ref_camera_rectangles_h = np.concatenate(
            [
                in_ref_camera_rectangles,
                np.ones((*in_ref_camera_rectangles.shape[:2], 1)),
            ],
            axis=2,
        )
        ref_Rt = as_4x4_Rt(ref_rigid3d_dict["R"], ref_rigid3d_dict["t"])
        ref_Rt_inv = np.linalg.inv(ref_Rt)
        dest_Rt = as_4x4_Rt(dest_rigid3d_dict["R"], dest_rigid3d_dict["t"])

        dest_camera_rectangle_h = (
            dest_Rt @ ref_Rt_inv @ in_ref_camera_rectangles_h.transpose(0, 2, 1)
        ).transpose(0, 2, 1)
        dest_camera_rectangle = dest_camera_rectangle_h[..., :-1]
        return dest_camera_rectangle


sweep_plane = _PlaneSweep()


class PlaneInterval(NamedTuple):
    middle_points: np.ndarray
    half_intervals: np.ndarray


class _DepthPlane:
    def from_half_intervals(
        self,
        plane: float,
        half_interval: float,
        no_intervals: int,
    ) -> PlaneInterval:
        sampled_limits = np.linspace(
            plane - half_interval,
            plane + half_interval,
            no_intervals + 1,
        )

        middle_points = (sampled_limits[1:] + sampled_limits[:-1]) / 2
        half_intervals = (sampled_limits[1:] - sampled_limits[:-1]) / 2
        return PlaneInterval(middle_points, half_intervals)

    def from_limits(
        self,
        lo_depth: float,
        hi_depth: float,
        no_of_depths: int,
    ) -> PlaneInterval:
        depths = np.linspace(lo_depth, hi_depth, no_of_depths + 1)
        half_intervals = (depths[1:] - depths[:-1]) / 2
        middle_points = (depths[1:] + depths[:-1]) / 2
        return PlaneInterval(middle_points, half_intervals)

    def compute_new_intervals(
        self,
        no_of_intervals: np.ndarray,
        planes: np.ndarray,
        half_intervals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_intervals = []
        all_planos = []
        for no_interval, plane, half_interval in zip(
            no_of_intervals, planes, half_intervals
        ):
            dephs = np.linspace(
                plane - half_interval,
                plane + half_interval,
                no_interval + 1,
            )
            za_planos = (dephs[1:] + dephs[:-1]) / 2
            halp_intervals = (dephs[1:] - dephs[:-1]) / 2
            all_planos.extend(za_planos)
            all_intervals.extend(halp_intervals)
        return np.array(all_intervals), np.array(all_planos)


depth_plane = _DepthPlane()


def img_size_to_rectangle(img_size: tuple[int, int]) -> np.ndarray:
    w, h = img_size
    rectangle = np.array(
        [
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1],
        ]
    )
    return rectangle


def img_size_to_rectangle_h(img_size: tuple[int, int]) -> np.ndarray:
    w, h = img_size
    rectangle = np.array(
        [
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1],
        ]
    )
    return rectangle


def make_rect_polygons(rects: np.ndarray) -> list[Polygon]:
    rect_polygons = [
        Polygon(
            camera_pixel_rect,
        )
        for camera_pixel_rect in rects
    ]
    return rect_polygons


class _IouVisibility:
    def from_polygons(
        self,
        img_polygon: Polygon,
        project_img_polygons: list[Polygon],
    ) -> list[float]:
        scores = []
        for project_img_polygon in project_img_polygons:
            try:
                intersection = project_img_polygon.intersection(
                    img_polygon,
                )
            except GEOSException:
                scores.append(0)
                continue

            try:
                union = project_img_polygon.union(
                    img_polygon,
                )
            except GEOSException:
                scores.append(0)
                continue
            scores.append(intersection.area / union.area)

        return scores

    def from_np_rectangles(
        self,
        img_rectangle: np.ndarray,
        projected_img_rectangles: np.ndarray,
    ) -> list[float]:
        img_rect_polygon = Polygon(img_rectangle)
        projected_img_rect_polygons = make_rect_polygons(projected_img_rectangles)
        score = self.from_polygons(img_rect_polygon, projected_img_rect_polygons)
        return score


iou_overlapping = _IouVisibility()


def recursive_drill(
    ref_rigid3d_dict: Rigid3dDict,
    ref_k: np.ndarray,
    ref_image_size: tuple[int, int],
    dest_rigid3d_dict: Rigid3dDict,
    dest_k: np.ndarray,
    dest_image_size: tuple[int, int],
    depth_range: tuple[float, float],
    no_of_depths: int,
    no_iter: int,
) -> np.ndarray:
    lo_depth, hi_depth = depth_range
    dest_img_rectangle = img_size_to_rectangle(dest_image_size)
    ref_img_rectangle_h = img_size_to_rectangle_h(ref_image_size)
    depth_planes, plane_intervals = depth_plane.from_limits(
        lo_depth,
        hi_depth,
        no_of_depths,
    )
    depths, scores = [], []

    for _ in range(no_iter):
        dest_projected_pixel_rectangles = sweep_plane.projected(
            ref_rigid3d_dict,
            ref_k,
            ref_img_rectangle_h,
            dest_rigid3d_dict,
            dest_k,
            depths=depth_planes,
        )
        overlapping_scores = iou_overlapping.from_np_rectangles(
            dest_img_rectangle,
            dest_projected_pixel_rectangles,
        )
        norm_sum = np.sum(overlapping_scores)
        if norm_sum == 0:
            # TODO: do a epsilon check maybe
            overlapping_scores_norm = overlapping_scores
        else:
            overlapping_scores_norm = overlapping_scores / np.sum(overlapping_scores)
        depths.extend(depth_planes)
        scores.extend(overlapping_scores)

        new_no_of_intervals = np.astype(
            np.round(overlapping_scores_norm * no_of_depths),
            np.uint32,
        )
        new_plane_intervals, new_depth_planes = depth_plane.compute_new_intervals(
            new_no_of_intervals,
            depth_planes,
            plane_intervals,
        )

        depth_planes, plane_intervals = new_depth_planes, new_plane_intervals
    depths, scores = np.array(depths), np.array(scores)
    sorted_indices_by_depths = np.argsort(depths)
    sorted_depths, sorted_scores = (
        depths[sorted_indices_by_depths],
        scores[sorted_indices_by_depths],
    )

    return sorted_depths, sorted_scores


def particle_sweep_covisibility(
    st_rigid3d_dict: Rigid3dDict,
    st_k: np.ndarray,
    st_image_size: tuple[int, int],
    nd_rigid3d_dict: Rigid3dDict,
    nd_k: np.ndarray,
    nd_image_size: tuple[int, int],
    depth_range: tuple[float, float],
    no_of_depths: int,
    no_iter: int = 2,
):
    _, ref_to_dest_scores = recursive_drill(
        st_rigid3d_dict,
        st_k,
        st_image_size,
        nd_rigid3d_dict,
        nd_k,
        nd_image_size,
        depth_range,
        no_of_depths,
        no_iter=no_iter,
    )

    _, dest_to_ref_scores = recursive_drill(
        nd_rigid3d_dict,
        nd_k,
        nd_image_size,
        st_rigid3d_dict,
        st_k,
        st_image_size,
        depth_range,
        no_of_depths,
        no_iter=no_iter,
    )
    max_ref_to_dest, max_dest_to_ref = (
        np.max(ref_to_dest_scores),
        np.max(dest_to_ref_scores),
    )
    denominator = max_dest_to_ref + max_ref_to_dest
    enumerator = 2 * max_ref_to_dest * max_dest_to_ref
    if denominator == 0:
        # TODO: do a epsilon check maybe
        return 0
    harmonic_mean = enumerator / denominator
    return harmonic_mean


def create_pairs_exhaustively(n: int) -> list[PairType[int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def sequence_particle_covisibility(
    rigid3d_dicts: list[Rigid3dDict],
    ks: list[np.ndarray],
    image_sizes: list[tuple[int, int]],
    depth_range: tuple[float, float],
    no_of_depths: int,
    pairs: list[tuple[int, int]] | None = None,
    no_iter: int = 2,
) -> dict[tuple[int, int], float]:
    # TODO: Check if the camera look into the same angle
    if pairs is None:
        pairs = create_pairs_exhaustively(len(rigid3d_dicts))
    result: dict[tuple[int, int], float] = {}
    for i, j in pairs:
        score = particle_sweep_covisibility(
            rigid3d_dicts[i],
            ks[i],
            image_sizes[i],
            rigid3d_dicts[j],
            ks[j],
            image_sizes[j],
            depth_range,
            no_of_depths,
            no_iter=no_iter,
        )
        result[(i, j)] = score
    return result
