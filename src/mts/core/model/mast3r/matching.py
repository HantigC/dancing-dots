# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Standalone coarse-to-fine 2D-2D matching.
#
# Give it two images, get back matched pixel coordinates in each image plus a
# per-match confidence. This is the coarse-to-fine pipeline from visloc.py with
# the visual-localization plumbing (3D points, camera intrinsics, dataset "view"
# dicts) stripped out.
#
# Role convention: ``img0`` is the "query" image, ``img1`` is the "reference"
# ("map") image. The distinction only affects the order of the returned arrays;
# the matcher itself is symmetric.
# --------------------------------------------------------
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image

from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.utils.coarse_to_fine import crop_slice, select_pairs_of_crops
from mast3r.utils.collate import cat_collate, cat_collate_fn_map

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference, loss_of_one_batch
from dust3r.utils.geometry import geotrf
from .visloc import get_HW_resolution, get_resize_function
from .image import ImgNorm


@dataclass
class CoarseToFineMatches:
    """Result of :func:`coarse_to_fine_match`.

    ``matches_im0`` / ``matches_im1`` are ``(N, 2)`` float arrays of pixel
    coordinates in the *original* resolution of ``img0`` / ``img1`` respectively
    (OpenCV convention: integer coordinates fall on pixel centers, origin at the
    top-left corner). ``confidence`` is an ``(N,)`` array, or ``None`` for the
    coarse-only fallback which does not produce per-match confidences.
    """
    matches_im0: np.ndarray
    matches_im1: np.ndarray
    confidence: Optional[np.ndarray]
    coarse_used: bool

    def __len__(self):
        return len(self.matches_im0)


def _load_image(x):
    if isinstance(x, Image.Image):
        return x.convert('RGB')
    if isinstance(x, np.ndarray):
        return Image.fromarray(x).convert('RGB')
    return Image.open(x).convert('RGB')


def _model_resolution(model):
    return max(model.patch_embed.img_size), model.patch_embed.patch_size


def _prepare_coarse_view(pil, model):
    """Rescale a PIL image to the model's coarse resolution.

    Mirrors what the visloc datasets do per view (see
    ``dust3r_visloc/datasets/sevenscenes.py``).
    """
    maxdim, patch_size = _model_resolution(model)
    W, H = pil.size
    resize_func, _to_resize, to_orig = get_resize_function(maxdim, patch_size, H, W)
    rgb_rescaled = resize_func(ImgNorm(pil))
    return dict(rgb=pil, rgb_rescaled=rgb_rescaled, to_orig=to_orig)


@torch.no_grad()
def _coarse_match(view0, view1, model, device, fast_nn_params):
    """Full-image low-resolution matching (visloc ``coarse_matching``, pixel_tol=0).

    Returns ``(matches_im0, matches_im1)`` in the *original* image coordinates.
    """
    imgs = []
    for idx, img in enumerate([view0['rgb_rescaled'], view1['rgb_rescaled']]):
        imgs.append(dict(img=img.unsqueeze(0), true_shape=np.int32([img.shape[1:]]),
                         idx=idx, instance=str(idx)))
    output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    desc0 = pred1['desc'].squeeze(0).detach()
    desc1 = pred2['desc'].squeeze(0).detach()
    if len(desc0) == 0 or len(desc1) == 0:
        return np.zeros((0, 2), np.float64), np.zeros((0, 2), np.float64)

    # reciprocal NNs over a regular grid of all points
    matches_im1, matches_im0 = fast_reciprocal_NNs(desc1, desc0, subsample_or_initxy1=8, **fast_nn_params)

    H1, W1 = view1['rgb_rescaled'].shape[1:]
    H0, W0 = view0['rgb_rescaled'].shape[1:]
    # ignore a small border around the edge
    valid1 = ((matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1 - 3) &
              (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1 - 3))
    valid0 = ((matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) &
              (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3))
    valid = valid0 & valid1
    matches_im0 = matches_im0[valid]
    matches_im1 = matches_im1[valid]

    # rescale from the coarse resolution back to the original one (cv2 -> colmap -> cv2)
    matches_im0 = matches_im0.astype(np.float64)
    matches_im1 = matches_im1.astype(np.float64)
    matches_im0[:, :2] += 0.5
    matches_im1[:, :2] += 0.5
    matches_im0 = geotrf(view0['to_orig'], matches_im0, norm=True)
    matches_im1 = geotrf(view1['to_orig'], matches_im1, norm=True)
    matches_im0[:, :2] -= 0.5
    matches_im1[:, :2] -= 0.5
    return matches_im0, matches_im1


def _resize_to_max(pil, max_image_size):
    """Resize a PIL image so its largest side is at most ``max_image_size``.

    Adapted from visloc ``resize_image_to_max`` (intrinsics handling removed).
    Returns ``(rgb_tensor_HWC, to_orig, to_resize, (H, W))`` where the transforms
    map between the resized frame and the original frame.
    """
    W, H = pil.size
    if max_image_size and max(W, H) > max_image_size:
        if W >= H:
            WMax = max_image_size
            HMax = int(H * (WMax / W))
        else:
            HMax = max_image_size
            WMax = int(W * (HMax / H))
        resize_op = tvf.Compose([ImgNorm, tvf.Resize(size=[HMax, WMax])])
        rgb_tensor = resize_op(pil).permute(1, 2, 0)
        to_orig = np.array([[W / WMax, 0, 0], [0, H / HMax, 0], [0, 0, 1]])
        to_resize = np.array([[WMax / W, 0, 0], [0, HMax / H, 0], [0, 0, 1]])
    else:
        rgb_tensor = ImgNorm(pil).permute(1, 2, 0)
        to_orig = np.eye(3)
        to_resize = np.eye(3)
        HMax, WMax = H, W
    return rgb_tensor, to_orig, to_resize, (HMax, WMax)


def _crop(img, crop):
    """Plain rectangular crop of an HxWxC tensor.

    ``crop`` is ``[x0, y0, x1, y1]``. Returns the cropped tensor and the 3x3
    transform that maps cropped coordinates back to ``img`` coordinates.
    """
    out = img[crop_slice(crop)]
    to_orig = torch.eye(3, device=img.device)
    to_orig[:2, -1] = torch.tensor(crop[:2])
    return out, to_orig


@torch.no_grad()
def _crops_inference(pairs, model, device, batch_size=48):
    """Forward a (possibly large) batch of crop pairs in blocks of ``batch_size``.

    Verbatim from visloc ``crops_inference``.
    """
    assert len(pairs) == 2, "Error, data should be a tuple of dicts containing the batch of image pairs"
    B = pairs[0]['img'].shape[0]
    if B < batch_size:
        return loss_of_one_batch(pairs, model, None, device=device, symmetrize_batch=False)
    preds = []
    for ii in range(0, B, batch_size):
        sel = slice(ii, ii + min(B - ii, batch_size))
        temp_data = [{}, {}]
        for di in [0, 1]:
            temp_data[di] = {kk: pairs[di][kk][sel]
                             for kk in pairs[di].keys() if pairs[di][kk] is not None}
        preds.append(loss_of_one_batch(temp_data, model, None, device=device, symmetrize_batch=False))
    return cat_collate(preds, collate_fn_map=cat_collate_fn_map)


@torch.no_grad()
def _fine_match(query_crops, ref_crops, model, device, max_batch_size, pixel_tol, subsample, fast_nn_params):
    """Match a batch of crop pairs and uncrop the results.

    Adapted from visloc ``fine_matching``: instead of matching only the valid
    (3D-backed) reference pixels, it seeds matching from a regular pixel grid
    over each reference crop.
    """
    assert pixel_tol > 0
    output = _crops_inference([query_crops, ref_crops], model, device, batch_size=max_batch_size)
    pred1, pred2 = output['pred1'], output['pred2']
    descs0 = pred1['desc'].clone()
    descs1 = pred2['desc'].clone()
    confs0 = pred1['desc_conf'].clone()
    confs1 = pred2['desc_conf'].clone()

    matches_im0, matches_im1, matches_confs = [], [], []
    for ppi, (pp0, pp1, cc0, cc1) in enumerate(zip(descs0, descs1, confs0, confs1)):
        conf0_np = cc0.cpu().numpy()
        conf1_np = cc1.cpu().numpy()

        Hc, Wc = pp1.shape[:2]
        s = max(1, int(subsample))
        y_grid, x_grid = np.mgrid[s // 2:Hc:s, s // 2:Wc:s].reshape(2, -1)
        matches_im1_ppi, matches_im0_ppi = fast_reciprocal_NNs(pp1, pp0, (x_grid, y_grid),
                                                               pixel_tol=pixel_tol, **fast_nn_params)

        matches_confs_ppi = np.minimum(
            conf1_np[matches_im1_ppi[:, 1], matches_im1_ppi[:, 0]],
            conf0_np[matches_im0_ppi[:, 1], matches_im0_ppi[:, 0]],
        )
        # uncrop pixel coordinates back to the resized (pre-crop) frame
        matches_im1_ppi = geotrf(ref_crops['to_orig'][ppi].cpu().numpy(), matches_im1_ppi.copy(), norm=True)
        matches_im0_ppi = geotrf(query_crops['to_orig'][ppi].cpu().numpy(), matches_im0_ppi.copy(), norm=True)

        matches_im0.append(matches_im0_ppi)
        matches_im1.append(matches_im1_ppi)
        matches_confs.append(matches_confs_ppi)

    if len(matches_im0) == 0:
        return (np.zeros((0, 2), np.float64), np.zeros((0, 2), np.float64), np.zeros((0,), np.float64))

    return (np.concatenate(matches_im0, axis=0),
            np.concatenate(matches_im1, axis=0),
            np.concatenate(matches_confs, axis=0))


@torch.no_grad()
def _fine_from_coarse_matches(img0, img1, coarse_im0, coarse_im1, model, device, *,
                              max_image_size, pixel_tol, subsample, max_batch_size,
                              overlap, conf_thr, fast_nn_params):
    """Crop-selection + fine pass, seeded by pre-computed correspondences.

    ``img0`` / ``img1`` are already-loaded PIL images. ``coarse_im0`` /
    ``coarse_im1`` are ``(N, 2)`` pixel coordinates in the *original* frame of
    ``img0`` / ``img1`` respectively; they are only used to pick which
    overlapping crop pairs to run the fine MASt3R pass on.

    This is the second half of :func:`coarse_to_fine_match` (everything after the
    coarse pass), shared with :func:`fine_match`.
    """
    maxdim, patch_size = _model_resolution(model)

    rgb0, to_orig0, to_resize0, (Hm0, Wm0) = _resize_to_max(img0, max_image_size)
    rgb1, to_orig1, to_resize1, (Hm1, Wm1) = _resize_to_max(img1, max_image_size)

    if len(coarse_im0) == 0:
        return CoarseToFineMatches(matches_im0=np.zeros((0, 2), np.float64),
                                   matches_im1=np.zeros((0, 2), np.float64),
                                   confidence=np.zeros((0,), np.float64), coarse_used=True)

    # move coarse matches into the resized (fine) frame
    c0 = geotrf(to_resize0, coarse_im0, norm=True)
    c1 = geotrf(to_resize1, coarse_im1, norm=True)

    res0 = get_HW_resolution(Hm0, Wm0, maxdim=maxdim, patchsize=patch_size)
    res1 = get_HW_resolution(Hm1, Wm1, maxdim=maxdim, patchsize=patch_size)

    query_crops, ref_crops = [], []
    query_to_orig, ref_to_orig = [], []
    for crop_ref, crop_qry, _tag in select_pairs_of_crops(rgb1, rgb0, c1, c0,
                                                          maxdim=maxdim, overlap=overlap,
                                                          forced_resolution=[res1, res0]):
        c_ref, tr_ref = _crop(rgb1, crop_ref)
        c_qry, tr_qry = _crop(rgb0, crop_qry)
        ref_crops.append(c_ref)
        query_crops.append(c_qry)
        ref_to_orig.append(tr_ref)
        query_to_orig.append(tr_qry)

    if len(query_crops) == 0:
        return CoarseToFineMatches(matches_im0=np.zeros((0, 2), np.float64),
                                   matches_im1=np.zeros((0, 2), np.float64),
                                   confidence=np.zeros((0,), np.float64), coarse_used=True)

    query_crops = torch.stack(query_crops)
    ref_crops = torch.stack(ref_crops)
    if query_crops.ndim == 3:
        query_crops, ref_crops = query_crops[None], ref_crops[None]
    query_to_orig = torch.stack(query_to_orig)
    ref_to_orig = torch.stack(ref_to_orig)

    query_crop_view = dict(img=query_crops.permute(0, 3, 1, 2),
                           instance=['0' for _ in range(query_crops.shape[0])],
                           to_orig=query_to_orig)
    ref_crop_view = dict(img=ref_crops.permute(0, 3, 1, 2),
                         instance=['1' for _ in range(ref_crops.shape[0])],
                         to_orig=ref_to_orig)

    matches_im0, matches_im1, confidence = _fine_match(query_crop_view, ref_crop_view, model, device,
                                                      max_batch_size, pixel_tol, subsample, fast_nn_params)

    # back from the resized frame to the original image frame
    matches_im0 = geotrf(to_orig0, matches_im0, norm=True)
    matches_im1 = geotrf(to_orig1, matches_im1, norm=True)

    if conf_thr is not None and len(confidence) > 0:
        keep = confidence >= conf_thr
        matches_im0 = matches_im0[keep]
        matches_im1 = matches_im1[keep]
        confidence = confidence[keep]

    return CoarseToFineMatches(matches_im0=matches_im0, matches_im1=matches_im1,
                               confidence=confidence, coarse_used=True)


@torch.no_grad()
def fine_match(img0, img1, matches_im0, matches_im1, model, device='cuda', *,
               max_image_size=None,
               pixel_tol=5,
               subsample=8,
               max_batch_size=48,
               overlap=0.5,
               conf_thr=None,
               fast_nn_params=None):
    """MASt3R fine matching seeded by pre-computed correspondences.

    Runs only the crop-selection + fine-refinement stage of
    :func:`coarse_to_fine_match`: instead of a coarse MASt3R pass,
    ``matches_im0`` / ``matches_im1`` (each ``(N, 2)`` float pixel coordinates in
    the *original* resolution of ``img0`` / ``img1``, OpenCV convention -- the
    same convention as :class:`CoarseToFineMatches`) are used to choose which
    overlapping crop pairs to refine.

    Args:
        img0, img1: query and reference images. Each may be a ``PIL.Image``, a
            path, or an ``HxWx3`` uint8 array.
        matches_im0, matches_im1: existing correspondences seeding crop
            selection, in the original ``img0`` / ``img1`` pixel frame.
        model: an ``AsymmetricMASt3R`` instance, already moved to ``device``.
        device: torch device string.
        max_image_size: cap on the fine-pass resolution (largest image side).
            ``None`` keeps the native resolution.
        pixel_tol: reciprocity tolerance (pixels) for the fine NN matching.
        subsample: pixel-grid stride used to seed fine matching within each crop.
        max_batch_size: crop-inference batch size.
        overlap: crop-grid overlap forwarded to ``select_pairs_of_crops``.
        conf_thr: if set, matches with ``confidence < conf_thr`` are dropped.
        fast_nn_params: kwargs for ``fast_reciprocal_NNs``. Defaults to
            ``dict(device=device, dist='dot', block_size=2**13)``.

    Returns:
        CoarseToFineMatches -- freshly computed fine matches (the input seeds are
        not included in the output). ``coarse_used`` is ``True`` to signal the
        fine pass ran.
    """
    assert pixel_tol > 0, "pixel_tol must be > 0"
    if fast_nn_params is None:
        fast_nn_params = dict(device=device, dist='dot', block_size=2**13)

    img0 = _load_image(img0)
    img1 = _load_image(img1)

    matches_im0 = np.asarray(matches_im0, dtype=np.float64).reshape(-1, 2)
    matches_im1 = np.asarray(matches_im1, dtype=np.float64).reshape(-1, 2)
    if len(matches_im0) == 0 or len(matches_im1) == 0:
        return CoarseToFineMatches(matches_im0=np.zeros((0, 2), np.float64),
                                   matches_im1=np.zeros((0, 2), np.float64),
                                   confidence=np.zeros((0,), np.float64), coarse_used=True)

    return _fine_from_coarse_matches(img0, img1, matches_im0, matches_im1, model, device,
                                     max_image_size=max_image_size, pixel_tol=pixel_tol,
                                     subsample=subsample, max_batch_size=max_batch_size,
                                     overlap=overlap, conf_thr=conf_thr,
                                     fast_nn_params=fast_nn_params)


@torch.no_grad()
def coarse_to_fine_match(img0, img1, model, device='cuda', *,
                         max_image_size=None,
                         pixel_tol=5,
                         coarse_to_fine=True,
                         subsample=8,
                         max_batch_size=48,
                         overlap=0.5,
                         conf_thr=None,
                         fast_nn_params=None):
    """Match two images with MASt3R, optionally using coarse-to-fine refinement.

    Args:
        img0, img1: query and reference images. Each may be a ``PIL.Image``, a
            path, or an ``HxWx3`` uint8 array.
        model: an ``AsymmetricMASt3R`` instance, already moved to ``device``.
        device: torch device string.
        max_image_size: cap on the fine-pass resolution (largest image side).
            ``None`` keeps the native resolution.
        pixel_tol: reciprocity tolerance (pixels) for the fine NN matching.
        coarse_to_fine: if ``False``, or if the image already fits the model's
            coarse resolution, only the single coarse pass runs.
        subsample: pixel-grid stride used to seed fine matching within each crop.
        max_batch_size: crop-inference batch size.
        overlap: crop-grid overlap forwarded to ``select_pairs_of_crops``.
        conf_thr: if set, matches with ``confidence < conf_thr`` are dropped
            (ignored on the coarse-only path, which has no confidences).
        fast_nn_params: kwargs for ``fast_reciprocal_NNs``. Defaults to
            ``dict(device=device, dist='dot', block_size=2**13)``.

    Returns:
        CoarseToFineMatches
    """
    assert pixel_tol > 0, "pixel_tol must be > 0"
    if fast_nn_params is None:
        fast_nn_params = dict(device=device, dist='dot', block_size=2**13)

    img0 = _load_image(img0)
    img1 = _load_image(img1)
    maxdim, _patch_size = _model_resolution(model)

    view0 = _prepare_coarse_view(img0, model)
    view1 = _prepare_coarse_view(img1, model)

    W0, H0 = img0.size
    W1, H1 = img1.size
    run_fine = coarse_to_fine and (maxdim < max(W0, H0) or maxdim < max(W1, H1))

    if not run_fine:
        matches_im0, matches_im1 = _coarse_match(view0, view1, model, device, fast_nn_params)
        return CoarseToFineMatches(matches_im0=matches_im0, matches_im1=matches_im1,
                                   confidence=None, coarse_used=False)

    # --- coarse pass: seeds the crop selection ---
    coarse_im0, coarse_im1 = _coarse_match(view0, view1, model, device, fast_nn_params)

    return _fine_from_coarse_matches(img0, img1, coarse_im0, coarse_im1, model, device,
                                     max_image_size=max_image_size, pixel_tol=pixel_tol,
                                     subsample=subsample, max_batch_size=max_batch_size,
                                     overlap=overlap, conf_thr=conf_thr,
                                     fast_nn_params=fast_nn_params)
