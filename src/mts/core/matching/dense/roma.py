import math
import os
from warnings import warn

import torch
import torch.nn as nn
from PIL import Image
from romatch.models.matcher import RegressionMatcher, _check_input
from romatch.utils import get_tuple_transform_ops
import torch.nn.functional as F


class RoMaTwoSteps(nn.Module):
    def __init__(
        self,
        regression_matcher: RegressionMatcher,
        h=448,
        w=448,
        sample_mode="threshold_balanced",
        upsample_preds=False,
        symmetric=False,
        sample_thresh=0.05,
        name=None,
        attenuate_cert=None,
        upsample_res=None,
    ) -> None:
        super().__init__()
        self.regression_matcher = regression_matcher
        self.attenuate_cert = attenuate_cert
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res or (14 * 16 * 6, 14 * 16 * 6)
        self.symmetric = symmetric
        self.sample_thresh = sample_thresh
        self.scale_factor = math.sqrt(
            self.h_resized * self.w_resized / (560**2)
        )  # divide by training resolution

    def _pre_half(
        self,
        im_A_input,
        *args,
        im_A_high_res=None,
        batched=True,
        device=None,
        upsample_preds: bool = True,
    ):
        self.train(False)
        if not batched:
            raise ValueError(
                "batched must be True, non-batched inference is no longer supported."
            )
        if device is None and not isinstance(im_A_input, torch.Tensor):
            device = self._get_device()
        elif device is None and isinstance(im_A_input, torch.Tensor):
            device = im_A_input.device

        # Check if inputs are file paths or already loaded images
        im_A = _check_input(im_A_input)
        ws = self.w_resized
        hs = self.h_resized

        if isinstance(im_A, Image.Image):
            b = 1
            w, h = im_A.size
            # Get images in good format

            test_transform = get_tuple_transform_ops(
                resize=(hs, ws), normalize=True, clahe=False
            )
            im_A, *_ = test_transform((im_A,))
            batch = {
                "im_A": im_A[None].to(device),
                "ws": ws,
                "hs": hs,
                "scale_factor": self.scale_factor,
            }
        elif isinstance(im_A, torch.Tensor):
            b, c, h, w = im_A.shape
            batch = {"im_A": im_A.to(device)}
            if h != self.h_resized or self.w_resized != w:
                warn(
                    "Model resolution and batch resolution differ, may produce unexpected results"
                )
            hs, ws = h, w
        else:
            raise ValueError(f"Unsupported input type: {type(im_A)=}")

        if upsample_preds:
            upscale_hs, upscale_ws = self.upsample_res

            test_transform = get_tuple_transform_ops(
                resize=(upscale_hs, upscale_ws), normalize=True
            )
            if isinstance(im_A_input, (str, os.PathLike)):
                im_A, *_ = test_transform((Image.open(im_A_input).convert("RGB"),))
            batch["ws_up"] = upscale_ws
            batch["hs_up"] = upscale_hs

            batch["upscale_im_A"] = im_A[None].to(device)
            batch["upscale_factor"] = math.sqrt(upscale_hs * upscale_ws / (560**2))

        return batch

    def encode(self, filepath, device=None, upsample_preds: bool = True):
        batch, scale_factor = self._pre_half(
            filepath,
            device=device,
        )
        x_q = batch["im_A"]
        s = self.regression_matcher.encoder(x_q, upsample=False)
        result = {"s": s}
        if upsample_preds:
            x_q_up = batch["upscale_im_A"]
            s_up = self.regression_matcher.encoder(x_q_up, upsample=True)
            result["s_up"] = s_up

        return result

    def post(
        self,
        corresps,
        ws,
        hs,
        device,
        finest_scale=1,
    ):
        b = 1
        im_A_to_im_B = corresps[finest_scale]["flow"]
        certainty = corresps[finest_scale]["certainty"]
        if finest_scale != 1:
            im_A_to_im_B = F.interpolate(
                im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
            )
            certainty = F.interpolate(
                certainty, size=(hs, ws), align_corners=False, mode="bilinear"
            )
        im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
        # Create im_A meshgrid
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            ),
            indexing="ij",
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
        certainty = certainty.sigmoid()  # logits -> probs
        im_A_coords = im_A_coords.permute(0, 2, 3, 1)
        if (im_A_to_im_B.abs() > 1).any() and True:
            wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
            certainty[wrong[:, None]] = 0
        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
        return (
            warp[0],
            certainty[0, 0],
        )

    def compute_kpts(self, f_q_pyramid, f_s_pyramid, H_A, W_A, H_B, W_B):
        corresps = self.regression_matcher.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=False,
            scale_factor=self.scale_factor,
        )
        warp, certainty = self.post(corresps, self.w_resized, self.h_resized)

        matches, certainty = self.regression_matcher.sample(warp, certainty, 1000)

        kptsA, kptsB = self.regression_matcher.to_pixel_coordinates(
            matches,
            H_A,
            W_A,
            H_B,
            W_B,
        )
        return kptsA, kptsB
