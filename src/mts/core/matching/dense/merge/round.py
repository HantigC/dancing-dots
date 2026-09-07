from collections import defaultdict

import numpy as np


def merge_matches(
    out_match: dict[str, dict[str, np.ndarray]],
    bucket_size: int = 1,
    round_digits: int = 1,
) -> tuple[
    dict[str, np.ndarray],
    dict[tuple[str, str], np.ndarray],
]:
    keypoints_per_image = defaultdict(list)

    bucketed_match = {}
    for img1, subdict in out_match.items():
        for img2, match in subdict.items():
            pts1 = np.round(match[:, :2] // bucket_size, decimals=round_digits) * bucket_size
            pts2 = np.round(match[:, 2:] // bucket_size, decimals=round_digits) * bucket_size
            bucketed_match.setdefault(img1, {})[img2] = pts1, pts2
            keypoints_per_image[img1].append(pts1)
            keypoints_per_image[img2].append(pts2)

    global_keypoints = {}
    coord_to_id = {}

    for img, kpt_list in keypoints_per_image.items():
        all_pts = np.concatenate(kpt_list, axis=0)
        unique_pts, inverse = np.unique(all_pts, axis=0, return_inverse=True)
        global_keypoints[img] = unique_pts

        coord_to_id[img] = {tuple(pt): idx for idx, pt in enumerate(unique_pts)}

    global_matches = {}

    for img1, subdict in bucketed_match.items():
        for img2, (pts1, pts2) in subdict.items():
            ids1 = np.array([coord_to_id[img1][tuple(pt)] for pt in pts1])
            ids2 = np.array([coord_to_id[img2][tuple(pt)] for pt in pts2])

            global_matches[(img1, img2)] = np.unique(np.stack([ids1, ids2], axis=1), axis=0)

    return global_keypoints, global_matches
