"""All-torch reciprocal nearest-neighbour matching for MASt3R dense features.

This is a device-resident (CUDA / MPS / CPU) re-implementation of
``mast3r.fast_nn`` -- no numpy/faiss round-trip -- ported from
``ipynbs/experiments/faster-mast3r.ipynb``. It deliberately mirrors (and
shadows) the numpy ``extract_dense_kpts`` in
:mod:`mts.core.matching.dense.mast3r`; callers pick a backend by importing
from the module they want. Everything here operates on ``torch.Tensor`` and
returns tensors -- the caller is responsible for moving results to CPU.
"""

import math

import numpy as np
import torch

from mts.core.matching.dense.mast3r import DecodedImagePairDict, NdFeatures, StFeatures


@torch.no_grad()
def bruteforce_reciprocal_nns(A, B, device="cuda", block_size=None, dist="l2"):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    A = A.to(device)
    B = B.to(device)

    if dist == "l2":
        dist_func = torch.cdist
        argmin = torch.min
    elif dist == "dot":

        def dist_func(A, B):
            return A @ B.T

        def argmin(X, dim):
            sim, nn = torch.max(X, dim=dim)
            return sim.neg_(), nn

    else:
        raise ValueError(f"Unknown {dist=}")

    if block_size is None or len(A) * len(B) <= block_size**2:
        dists = dist_func(A, B)
        _, nn_A = argmin(dists, dim=1)
        _, nn_B = argmin(dists, dim=0)
    else:
        dis_A = torch.full((A.shape[0],), float("inf"), device=device, dtype=A.dtype)
        dis_B = torch.full((B.shape[0],), float("inf"), device=device, dtype=B.dtype)
        nn_A = torch.full((A.shape[0],), -1, device=device, dtype=torch.int64)
        nn_B = torch.full((B.shape[0],), -1, device=device, dtype=torch.int64)
        number_of_iteration_A = math.ceil(A.shape[0] / block_size)
        number_of_iteration_B = math.ceil(B.shape[0] / block_size)

        for i in range(number_of_iteration_A):
            A_i = A[i * block_size : (i + 1) * block_size]
            for j in range(number_of_iteration_B):
                B_j = B[j * block_size : (j + 1) * block_size]
                dists_blk = dist_func(A_i, B_j)
                min_A_i, argmin_A_i = argmin(dists_blk, dim=1)
                min_B_j, argmin_B_j = argmin(dists_blk, dim=0)

                col_mask = min_A_i < dis_A[i * block_size : (i + 1) * block_size]
                line_mask = min_B_j < dis_B[j * block_size : (j + 1) * block_size]

                dis_A[i * block_size : (i + 1) * block_size][col_mask] = min_A_i[col_mask]
                dis_B[j * block_size : (j + 1) * block_size][line_mask] = min_B_j[line_mask]

                nn_A[i * block_size : (i + 1) * block_size][col_mask] = (
                    argmin_A_i[col_mask] + (j * block_size)
                )
                nn_B[j * block_size : (j + 1) * block_size][line_mask] = (
                    argmin_B_j[line_mask] + (i * block_size)
                )
    return nn_A, nn_B


class cdistMatcher:
    def __init__(self, db_pts, device="cuda"):
        self.db_pts = db_pts.to(device)
        self.device = device

    def query(self, queries, k=1, **kw):
        assert k == 1
        if queries.numel() == 0:
            return None, []
        nnA, nnB = bruteforce_reciprocal_nns(
            queries, self.db_pts, device=self.device, **kw
        )
        dis = None
        return dis, nnA


def unravel_index_torch(idx: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Equivalent of ``np.unravel_index`` for 2D shapes, batched over ``idx``.

    Returns a tensor of shape ``(N, 2)`` with ``(row, col)`` i.e. ``(y, x)`` per index.
    """
    H, W = shape
    y = idx // W
    x = idx % W
    return torch.stack([y, x], dim=1)


def merge_corres_torch(
    idx1, idx2, shape1=None, shape2=None, ret_xy=True, ret_index=False
):
    assert idx1.dtype == idx2.dtype

    # torch.unique(dim=0) sorts lexicographically (primarily by idx1, then
    # idx2), matching the original int64-packed sort.
    pairs = torch.stack([idx1, idx2], dim=1)
    unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)

    if ret_index:
        # first-occurrence index of each unique pair in the original arrays
        n = pairs.shape[0]
        perm = torch.arange(n, device=pairs.device)
        inverse_rev = inverse.flip(0)
        perm_rev = perm.flip(0)
        indices = torch.empty(
            unique_pairs.shape[0], dtype=torch.long, device=pairs.device
        )
        indices.scatter_(0, inverse_rev, perm_rev)

    xy1 = unique_pairs[:, 0]
    xy2 = unique_pairs[:, 1]

    if ret_xy:
        assert shape1 and shape2
        xy1 = unravel_index_torch(xy1, shape1)
        xy2 = unravel_index_torch(xy2, shape2)
        if ret_xy != "y_x":
            xy1 = xy1.flip(-1)
            xy2 = xy2.flip(-1)

    if ret_index:
        return xy1, xy2, indices
    return xy1, xy2


def fast_reciprocal_NNs(
    pts1,
    pts2,
    subsample_or_initxy1=8,
    ret_xy=True,
    pixel_tol=0,
    ret_basin=False,
    device="cuda",
    max_iter: int = 1,
    **matcher_kw,
):
    H1, W1, DIM1 = pts1.shape
    H2, W2, DIM2 = pts2.shape
    assert DIM1 == DIM2
    pts1 = pts1.reshape(-1, DIM1)
    pts2 = pts2.reshape(-1, DIM2)

    if isinstance(subsample_or_initxy1, int) and pixel_tol == 0:
        S = subsample_or_initxy1
        y1, x1 = torch.meshgrid(
            torch.arange(S // 2, H1, S, device=device),
            torch.arange(S // 2, W1, S, device=device),
            indexing="ij",
        )
        x1, y1 = x1.reshape(-1), y1.reshape(-1)
    else:
        x1, y1 = subsample_or_initxy1
        x1 = torch.as_tensor(x1, device=device)
        y1 = torch.as_tensor(y1, device=device)

    # make sure there's no doublons
    xy1 = torch.unique((x1 + W1 * y1).to(torch.long))
    xy2 = torch.full_like(xy1, -1)
    old_xy1 = xy1.clone()
    old_xy2 = xy2.clone()

    tree1 = cdistMatcher(pts1, device=device)
    tree2 = cdistMatcher(pts2, device=device)

    notyet = torch.ones(xy1.shape[0], dtype=torch.bool, device=device)
    basin = (
        torch.full((H1 * W1 + 1,), -1, dtype=torch.long, device=device)
        if ret_basin
        else None
    )

    niter = 0
    while notyet.any():
        _, xy2_upd = tree2.query(pts1[xy1[notyet]], **matcher_kw)
        xy2[notyet] = xy2_upd
        if not ret_basin:
            notyet &= old_xy2 != xy2  # remove points that have converged

        _, xy1_upd = tree1.query(pts2[xy2[notyet]], **matcher_kw)
        xy1[notyet] = xy1_upd
        notyet &= old_xy1 != xy1  # remove points that have converged

        niter += 1
        if niter >= max_iter:
            break
        old_xy2 = xy2.clone()
        old_xy1 = xy1.clone()

    if pixel_tol > 0:
        # in case we only want to match some specific points and still have
        # some way of checking reciprocity
        old_yx1 = torch.stack([old_xy1 // W1, old_xy1 % W1], dim=1).to(torch.float32)
        new_yx1 = torch.stack([xy1 // W1, xy1 % W1], dim=1).to(torch.float32)
        dis = torch.linalg.norm(old_yx1 - new_yx1, dim=-1)
        converged = dis < pixel_tol
        if not isinstance(subsample_or_initxy1, int):
            xy1 = old_xy1  # replace new points by old ones
    else:
        converged = ~notyet  # converged correspondences

    # keep only unique correspondences, and sort on xy1
    xy1, xy2 = merge_corres_torch(
        xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy
    )[:2]

    if ret_basin:
        return xy1, xy2, basin
    return xy1, xy2


def extract_correspondences_nonsym(
    A,
    B,
    confA,
    confB,
    subsample=8,
    device=None,
    ptmap_key="pred_desc",
    pixel_tol=0,
    max_iter=1,
):
    if "3d" in ptmap_key:
        opt = dict(device="cpu", workers=32)
    else:
        opt = dict(device=device, dist="dot", block_size=2**13)

    HA, WA = A.shape[:2]
    HB, WB = B.shape[:2]

    if pixel_tol == 0:
        nn1to2 = fast_reciprocal_NNs(
            A, B, subsample_or_initxy1=subsample, ret_xy=False, max_iter=max_iter, **opt
        )
        nn2to1 = fast_reciprocal_NNs(
            B, A, subsample_or_initxy1=subsample, ret_xy=False, max_iter=max_iter, **opt
        )
    else:
        S = subsample
        yA, xA = torch.meshgrid(
            torch.arange(S // 2, HA, S, device=device),
            torch.arange(S // 2, WA, S, device=device),
            indexing="ij",
        )
        yB, xB = torch.meshgrid(
            torch.arange(S // 2, HB, S, device=device),
            torch.arange(S // 2, WB, S, device=device),
            indexing="ij",
        )
        xA, yA = xA.reshape(-1), yA.reshape(-1)
        xB, yB = xB.reshape(-1), yB.reshape(-1)
        nn1to2 = fast_reciprocal_NNs(
            A,
            B,
            subsample_or_initxy1=(xA, yA),
            ret_xy=False,
            pixel_tol=pixel_tol,
            max_iter=max_iter,
            **opt,
        )
        nn2to1 = fast_reciprocal_NNs(
            B,
            A,
            subsample_or_initxy1=(xB, yB),
            ret_xy=False,
            pixel_tol=pixel_tol,
            max_iter=max_iter,
            **opt,
        )

    idx1 = torch.cat([nn1to2[0], nn2to1[1]])
    idx2 = torch.cat([nn1to2[1], nn2to1[0]])

    c1 = confA.reshape(-1)[idx1]
    c2 = confB.reshape(-1)[idx2]

    xy1, xy2, idx = merge_corres_torch(
        idx1, idx2, (HA, WA), (HB, WB), ret_xy=True, ret_index=True
    )
    conf = torch.minimum(c1[idx], c2[idx])

    corres = (xy1.clone(), xy2.clone(), conf)
    return corres


def transform_keypoints_to_original(
    kpts_crop: torch.Tensor,
    original_size: tuple[int, int],
    size_param: int = 512,
    square_ok: bool = False,
) -> torch.Tensor:
    """Torch copy of :func:`mts.core.model.mast3r.transform.transform_keypoints_to_original`.

    Maps ``(x, y)`` keypoints from a DUST3R-processed (long side resized to
    ``size_param``, centre-cropped to a multiple of 16) image back to the
    original image's coordinate system. ``original_size`` is ``(height, width)``.
    """
    original_height, original_width = original_size
    original_height = float(original_height)
    original_width = float(original_width)

    # dimensions after resizing but before cropping (W_res, H_res)
    if size_param == 224:
        target_long_side = round(
            size_param
            * max(original_width / original_height, original_height / original_width)
        )
        if original_width >= original_height:
            W_res = target_long_side
            H_res = round(original_height * (target_long_side / original_width))
        else:
            H_res = target_long_side
            W_res = round(original_width * (target_long_side / original_height))
    else:
        if original_width >= original_height:
            W_res = size_param
            H_res = round(original_height * (size_param / original_width))
        else:
            H_res = size_param
            W_res = round(original_width * (size_param / original_height))

    # cropping offsets used during processing
    cx, cy = W_res // 2, H_res // 2
    if size_param == 224:
        half = min(cx, cy)
        crop_left = cx - half
        crop_top = cy - half
    else:
        halfw = ((2 * cx) // 16) * 8
        halfh = ((2 * cy) // 16) * 8
        if not square_ok and W_res == H_res:
            halfh = round(3 * halfw / 4)
        crop_left = cx - halfw
        crop_top = cy - halfh

    # reverse the resizing
    if original_width >= original_height:
        scale_factor = size_param / original_width
    else:
        scale_factor = size_param / original_height

    # reverse the cropping
    kpts_resized = kpts_crop.to(torch.float32).clone()
    offset = torch.tensor(
        [crop_left, crop_top], dtype=kpts_resized.dtype, device=kpts_resized.device
    )
    kpts_resized = kpts_resized + offset

    kpts_original = kpts_resized / scale_factor
    return kpts_original


def extract_dense_kpts(
    pred1: StFeatures,
    pred2: NdFeatures,
    st_original_hw_shape: tuple[int, int],
    nd_original_hw_shape: tuple[int, int],
    match_conf_th: float,
    min_pairs: int,
    device,
    top_k: int | None = None,
    pixel_tol: int = 0,
    subsample: int = 8,
    max_iter: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    desc1, desc2 = (
        pred1["desc"].squeeze(0).detach(),
        pred2["desc"].squeeze(0).detach(),
    )
    conf1, conf2 = (
        pred1["conf"].squeeze(0).detach(),
        pred2["conf"].squeeze(0).detach(),
    )
    corres = extract_correspondences_nonsym(
        desc1,
        desc2,
        conf1,
        conf2,
        device=device,
        subsample=subsample,
        pixel_tol=pixel_tol,
        max_iter=max_iter,
    )
    score = corres[2]
    mask = score >= match_conf_th
    matches_im0 = corres[0][mask]
    matches_im1 = corres[1][mask]

    if matches_im0.shape[0] < min_pairs:
        empty = torch.empty(0, device=device)
        return empty, empty

    if top_k is not None:
        masked_score = score[mask]
        indices = torch.argsort(masked_score)
        top_indices = indices[:top_k]
        matches_im0 = matches_im0[top_indices]
        matches_im1 = matches_im1[top_indices]

    H0, W0 = pred1["true_shape"].to(device).flatten()[:2]
    H1, W1 = pred2["true_shape"].to(device).flatten()[:2]

    valid0 = (
        (matches_im0[:, 0] >= 3)
        & (matches_im0[:, 0] < W0 - 3)
        & (matches_im0[:, 1] >= 3)
        & (matches_im0[:, 1] < H0 - 3)
    )
    valid1 = (
        (matches_im1[:, 0] >= 3)
        & (matches_im1[:, 0] < W1 - 3)
        & (matches_im1[:, 1] >= 3)
        & (matches_im1[:, 1] < H1 - 3)
    )
    valid = valid0 & valid1
    matches_im0 = matches_im0[valid]
    matches_im1 = matches_im1[valid]

    if matches_im0.shape[0] < min_pairs:
        empty = torch.empty(0, device=device)
        return empty, empty

    matches_im0_org = transform_keypoints_to_original(matches_im0, st_original_hw_shape)
    matches_im1_org = transform_keypoints_to_original(matches_im1, nd_original_hw_shape)
    return matches_im0_org, matches_im1_org


def dense_extract(
    decoded_feature_pairs: DecodedImagePairDict,
    st_original_size: tuple[int, int],
    nd_original_size: tuple[int, int],
    device: str | torch.device,
    match_conf_th: float = 1.01,
    min_pairs: int = 200,
    subsample: int = 8,
    pixel_tol: int = 0,
    max_iter: int = 1,
    top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    st_kpts, nd_kpts = extract_dense_kpts(
        decoded_feature_pairs["st_features"],
        decoded_feature_pairs["nd_features"],
        st_original_size,
        nd_original_size,
        match_conf_th,
        min_pairs,
        device,
        subsample=subsample,
        top_k=top_k,
        pixel_tol=pixel_tol,
        max_iter=max_iter,
    )
    return st_kpts, nd_kpts
