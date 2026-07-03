import logging
import time
from collections import defaultdict
from typing import Any, TypedDict

import cv2
import numpy as np
import torch
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.fast_nn import extract_correspondences_nonsym
from mast3r.model import AsymmetricMASt3R
from torch import nn
from tqdm.auto import tqdm

from mts.core.matching.dense.merge.round import merge_matches
from mts.core.model.mast3r.transform import transform_keypoints_to_original
from mts.core.types import PairType

LOGGER = logging.getLogger(__name__)


class StFeatures(TypedDict):
    pts3d: np.ndarray | torch.Tensor
    conf: np.ndarray | torch.Tensor
    desc: np.ndarray | torch.Tensor
    desc_conf: np.ndarray | torch.Tensor
    true_shape: np.ndarray | torch.Tensor


class NdFeatures(TypedDict):
    pts3d_in_other_view: np.ndarray | torch.Tensor
    conf: np.ndarray | torch.Tensor
    desc: np.ndarray | torch.Tensor
    desc_conf: np.ndarray | torch.Tensor
    true_shape: np.ndarray | torch.Tensor


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
) -> tuple[np.ndarray, np.ndarray]:

    # at this stage, you have the raw dust3r predictions

    desc1, desc2 = (
        pred1["desc"].squeeze(0).detach(),
        pred2["desc"].squeeze(0).detach(),
    )

    conf1, conf2 = (
        pred1["desc_conf"].squeeze(0).detach(),
        pred2["desc_conf"].squeeze(0).detach(),
    )
    corres = extract_correspondences_nonsym(
        desc1,
        desc2,
        conf1,
        conf2,
        device=device,
        subsample=subsample,
        pixel_tol=pixel_tol,
    )
    score = corres[2]
    mask = score >= match_conf_th

    matches_im0 = corres[0][mask]
    matches_im1 = corres[1][mask]

    if len(matches_im0) < min_pairs:
        return (np.array([]), np.array([]))

    if top_k is not None:
        maksed_score = score[mask]
        indices = torch.argsort(maksed_score)
        top_indices = indices[:top_k]
        matches_im0 = matches_im0[top_indices]
        matches_im1 = matches_im1[top_indices]
    matches_im0 = matches_im0.cpu().numpy()
    matches_im1 = matches_im1.cpu().numpy()

    H0, W0 = pred1["true_shape"].numpy()
    H1, W1 = pred2["true_shape"].numpy()

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
        return (np.array([]), np.array([]))

    matches_im0_org = transform_keypoints_to_original(matches_im0, st_original_hw_shape)
    matches_im1_org = transform_keypoints_to_original(matches_im1, nd_original_hw_shape)

    return matches_im0_org, matches_im1_org


def extract_dense_keypoints(
    mast3r_model: AsymmetricMASt3R,
    index_pairs: list[PairType[int]],
    image_list: list[str],
    min_pairs: int = 15,
    match_conf_th: float = 1.001,
    device: str | torch.device = None,
    tqdm_kwargs: dict[str, Any] = None,
    pixel_tol: int = 0,
    batch_size: int = 1,
) -> tuple[
    dict[str, np.ndarray],
    dict[tuple[str, str], np.ndarray],
]:
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
            [tuple(images)], mast3r_model, device, batch_size=batch_size, verbose=False
        )

        # at this stage, you have the raw dust3r predictions
        start_time = time.time()
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
        try:
            corres = extract_correspondences_nonsym(
                desc1, desc2, conf1, conf2, device=device, subsample=8, pixel_tol=pixel_tol,
            )
        except Exception:
            LOGGER.exception(
                "Something happened when extracting the matches for (%s, %s) pair",
                name1,
                name2,
            )
            continue
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
        LOGGER.info("Took %f seconds to extract", time.time() - start_time)

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
    try:
        out_match = extract_dense_keypoints(
            mast3r_model,
            index_pairs,
            image_list,
            min_pairs=min_pairs,
            match_conf_th=match_conf_th,
            device=device,
        )
    except Exception:
        LOGGER.exception("An error occurred while matching")

    global_keypoints, global_matches = merge_matches(out_match)
    # print("points and matches unified")
    return global_keypoints, global_matches


class ImageDict(TypedDict):
    image: torch.Tensor
    true_shape: torch.Tensor


class EncodedImageDict(TypedDict):
    encoded_features: torch.Tensor
    decoded_features: torch.Tensor
    positions: torch.Tensor


class EncodedImageFeaturesDict(TypedDict):
    encoded_features: torch.Tensor
    decoded_features: torch.Tensor
    positions: torch.Tensor
    true_shape: torch.Tensor

    @staticmethod
    def from_add_shape(
        encoded_image_dict: EncodedImageDict,
        true_shape: torch.Tensor,
    ) -> "EncodedImageFeaturesDict":
        encoded_image_features_dict: EncodedImageFeaturesDict = {
            "true_shape": true_shape
        }
        encoded_image_features_dict.update(encoded_image_dict)
        return encoded_image_features_dict

    @staticmethod
    def from_combined(
        encoded_image_dict: EncodedImageDict,
        image_dict: ImageDict,
    ) -> "EncodedImageFeaturesDict":
        encoded_image_features_dict = EncodedImageFeaturesDict.from_add_shape(
            encoded_image_dict,
            image_dict["true_shape"],
        )
        return encoded_image_features_dict


class DecodedImagePairDict(TypedDict):
    st_features: StFeatures
    nd_features: NdFeatures


class Mast3rTwoStep(nn.Module):
    def __init__(self, mast3r_model: AsymmetricMASt3R) -> None:
        super().__init__()
        self.mast3r_model = mast3r_model

    def encode_image(self, image_dict: ImageDict) -> EncodedImageDict:
        image = image_dict["image"]
        true_shape = image_dict["true_shape"]

        encoded_features, positions, _ = self.mast3r_model._encode_image(
            image, true_shape
        )
        decoded_features = self.mast3r_model.decoder_embed(encoded_features)
        encoded_image_dict = EncodedImageDict(
            encoded_features=encoded_features,
            decoded_features=decoded_features,
            positions=positions,
        )
        return encoded_image_dict

    def decode_feature_pairs(
        self,
        st_encoded_image_features: EncodedImageFeaturesDict,
        nd_encoded_image_features: EncodedImageFeaturesDict,
    ) -> DecodedImagePairDict:
        shape1 = st_encoded_image_features["true_shape"]
        shape2 = nd_encoded_image_features["true_shape"]
        dec1, dec2 = self._apply_decoder_blocks(
            st_encoded_image_features,
            nd_encoded_image_features,
        )

        st_features = self.mast3r_model._downstream_head(
            1, [tok.float() for tok in dec1], shape1
        )
        nd_features = self.mast3r_model._downstream_head(
            2, [tok.float() for tok in dec2], shape2
        )
        st_features["true_shape"] = shape1
        nd_features["true_shape"] = shape2

        nd_features["pts3d_in_other_view"] = nd_features.pop(
            "pts3d"
        )  # predict view2's pts3d in view1's frame
        decoded_image_pair_dict = DecodedImagePairDict(
            st_features=st_features,
            nd_features=nd_features,
        )
        return decoded_image_pair_dict

    def _apply_decoder_blocks(
        self,
        st_encoded_image_features: EncodedImageFeaturesDict,
        nd_encoded_image_features: EncodedImageFeaturesDict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_output = [
            (
                st_encoded_image_features["encoded_features"],
                nd_encoded_image_features["encoded_features"],
            ),
            (
                st_encoded_image_features["decoded_features"],
                nd_encoded_image_features["decoded_features"],
            ),
        ]

        pos1 = st_encoded_image_features["positions"]
        pos2 = nd_encoded_image_features["positions"]

        for blk1, blk2 in zip(
            self.mast3r_model.dec_blocks, self.mast3r_model.dec_blocks2
        ):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.mast3r_model.dec_norm, final_output[-1]))

        dec1, dec2 = zip(*final_output)
        return dec1, dec2
