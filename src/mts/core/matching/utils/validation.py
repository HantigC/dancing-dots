import numpy as np
import pycolmap


def validate_matches(
    st_kpts: np.ndarray,
    nd_kpts: np.ndarray,
    matches: np.ndarray,
    st_hw: tuple[int, int],
    nd_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    st_h, st_w = st_hw
    nd_h, nd_w = nd_hw

    camera1 = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=st_w,
        height=st_h,
        params=[0.9 * max(st_w, st_h), st_w / 2, st_h / 2],
    )

    camera2 = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=nd_w,
        height=nd_h,
        params=[0.9 * max(nd_w, nd_h), nd_w / 2, nd_h / 2],
    )

    options = pycolmap.TwoViewGeometryOptions()
    options.compute_relative_pose = True

    result = pycolmap.estimate_two_view_geometry(
        camera1,
        st_kpts,
        camera2,
        nd_kpts,
        matches=matches,
        options=options,
    )

    return result.inlier_matches


def validate_kps_matches(
    st_kpts: np.ndarray,
    nd_kpts: np.ndarray,
    st_hw: tuple[int, int],
    nd_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    arranged_matches = np.tile(np.arange(0, len(st_kpts))[:, np.newaxis], (1, 2))
    inlier_matches = validate_matches(
        st_kpts,
        nd_kpts,
        arranged_matches,
        st_hw,
        nd_hw,
    )
    return inlier_matches
