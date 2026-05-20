from collections import defaultdict

import numpy as np


def merge_matches(
    out_match: dict[str, dict[str, np.ndarray]],
    round_digits: int = 1,
) -> tuple[
    dict[str, np.ndarray],
    dict[tuple[str, str], np.ndarray],
]:
    keypoints_per_image = defaultdict(list)

    for img1, subdict in out_match.items():
        for img2, match in subdict.items():
            pts1 = np.round(match[:, :2], decimals=round_digits)
            pts2 = np.round(match[:, 2:], decimals=round_digits)
            keypoints_per_image[img1].append(pts1)
            keypoints_per_image[img2].append(pts2)

    global_keypoints = {}
    coord_to_id = {}

    for img, kpt_list in keypoints_per_image.items():
        all_pts = np.concatenate(kpt_list, axis=0)
        all_pts = np.round(all_pts, decimals=round_digits)
        unique_pts, inverse = np.unique(all_pts, axis=0, return_inverse=True)
        global_keypoints[img] = unique_pts

        coord_to_id[img] = {tuple(pt): idx for idx, pt in enumerate(unique_pts)}

    global_matches = {}

    for img1, subdict in out_match.items():
        for img2, match in subdict.items():
            pts1 = np.round(match[:, :2], decimals=round_digits)
            pts2 = np.round(match[:, 2:], decimals=round_digits)

            ids1 = np.array([coord_to_id[img1][tuple(pt)] for pt in pts1])
            ids2 = np.array([coord_to_id[img2][tuple(pt)] for pt in pts2])

            global_matches[(img1, img2)] = np.stack([ids1, ids2], axis=1)

    return global_keypoints, global_matches
