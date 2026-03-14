from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import torch
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.fast_nn import extract_correspondences_nonsym
from mast3r.model import AsymmetricMASt3R
from tqdm.auto import tqdm

from mts.core.matching.dense.merge.round import merge_matches
from mts.core.model.mast3r.transform import transform_keypoints_to_original
from mts.core.types import PairType


def extract_dense_keypoints(
    mast3r_model: AsymmetricMASt3R,
    index_pairs: list[PairType[int]],
    image_list: list[str],
    min_pairs: int = 15,
    match_conf_th: float = 1.001,
    device: str | torch.device = None,
    tqdm_kwargs: dict[str, Any] = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[tuple[str, str], np.ndarray],
]:
    unique_keypoints = defaultdict(list)
    out_match = defaultdict(dict)
    tqdm_kwargs = tqdm_kwargs or {}

    for idx1, idx2 in tqdm(
        index_pairs,
        desc="Computing the matches using Mast3r",
        **tqdm_kwargs,
    ):
        name1, name2 = image_list[idx1], image_list[idx2]
        key1, key2 = name1, name2

        # Only re-run inference for key1 if not in cache
        images = load_images([name1, name2], size=512, verbose=False)
        output = inference(
            [tuple(images)], mast3r_model, device, batch_size=1, verbose=False
        )

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]

        desc1, desc2 = (
            pred1["desc"].squeeze(0).detach(),
            pred2["desc"].squeeze(0).detach(),
        )

        # matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, device=device)
        # print(f"get pair for {key1}_{key2}, {len(matches_im0)}")

        conf1, conf2 = (
            pred1["desc_conf"].squeeze(0).detach(),
            pred2["desc_conf"].squeeze(0).detach(),
        )
        corres = extract_correspondences_nonsym(
            desc1, desc2, conf1, conf2, device=device, subsample=8, pixel_tol=5
        )
        score = corres[2]
        mask = score >= match_conf_th
        matches_im0 = corres[0][mask].cpu().numpy()
        matches_im1 = corres[1][mask].cpu().numpy()

        if len(matches_im0) < min_pairs:
            continue

        H0, W0 = view1["true_shape"][0].tolist()
        H1, W1 = view2["true_shape"][0].tolist()

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
        if len(matches_im0) < min_pairs:
            continue
        # print("transform_keypoints_to_original begin")
        # print(f"{key1}_{key2}: {len(matches_im0)} matches")
        img0 = cv2.imread(name1)
        img1 = cv2.imread(name2)
        H0, W0 = img0.shape[:2]
        H1, W1 = img1.shape[:2]
        matches_im0_org = transform_keypoints_to_original(matches_im0, (H0, W0))
        matches_im1_org = transform_keypoints_to_original(matches_im1, (H1, W1))

        # matches_im0_org = transform_keypoints_to_original(matches_im0, view1['true_shape'][0].tolist())
        # matches_im1_org = transform_keypoints_to_original(matches_im1, view2['true_shape'][0].tolist())
        # print("transform_keypoints_to_original end")

        unique_keypoints[key1].append(matches_im0_org)
        unique_keypoints[key2].append(matches_im1_org)
        out_match[key1][key2] = np.concatenate(
            [matches_im0_org, matches_im1_org], axis=1
        )
    return out_match



def match_pairs(
    mast3r_model: AsymmetricMASt3R,
    index_pairs: list[PairType[int]],
    image_list: list[str],
    min_pairs: int = 15,
    match_conf_th: float = 1.001,
    device: str | torch.device = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[tuple[str, str], np.ndarray],
]:
    out_match = extract_dense_keypoints(
        mast3r_model,
        index_pairs,
        image_list,
        min_pairs=min_pairs,
        match_conf_th=match_conf_th,
        device=device,
    )

    global_keypoints, global_matches = merge_matches(out_match)
    # print("points and matches unified")
    return global_keypoints, global_matches
