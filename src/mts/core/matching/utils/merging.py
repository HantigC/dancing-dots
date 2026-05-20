import numpy as np
from numpy import ma as np_ma
from scipy.spatial import KDTree


def match_kpts(
    kpts_intermediary_from: np.ndarray,
    kpts_intermediary_to: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nd_prev_kpts_tree = KDTree(kpts_intermediary_from)
    distance, indices = nd_prev_kpts_tree.query(kpts_intermediary_to, p=2)

    ma_indices = np_ma.masked_where(distance > 1, indices)

    from_idx, to_idx = np.unique(
        ma_indices,
        return_index=True,
    )
    indices_from = from_idx.compressed()
    indices_to = to_idx[~from_idx.mask]
    return indices_from, indices_to
