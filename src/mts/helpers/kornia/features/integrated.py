from typing import ClassVar, Dict, List, Optional, Tuple

import torch
from kornia.constants import pi
from kornia.core import Tensor, deg2rad


from kornia.feature.laf import (
    get_laf_center,
    get_laf_orientation,
    get_laf_scale,
)
from kornia.feature.matching import GeometryAwareDescriptorMatcher, _no_match
from mts.helpers.kornia.features.lightglue import LightGlue


class LightGlueMatcher(GeometryAwareDescriptorMatcher):
    """LightGlue-based matcher in kornia API.

    This is based on the original code from paper "LightGlue: Local Feature Matching at Light Speed".
    See :cite:`LightGlue2023` for more details.

    Args:
        feature_name: type of feature for matching, can be `disk` or `superpoint`.
        params: LightGlue params.

    """

    known_modes: ClassVar[List[str]] = [
        "aliked",
        "dedodeb",
        "dedodeg",
        "disk",
        "dog_affnet_hardnet",
        "doghardnet",
        "keynet_affnet_hardnet",
        "sift",
        "superpoint",
    ]

    def __init__(self, feature_name: str = "disk", params: Optional[Dict] = None) -> None:  # type: ignore
        feature_name_: str = feature_name.lower()
        super().__init__(feature_name_)
        self.feature_name = feature_name_
        if params is None:
            params = {}
        self.params = params
        self.matcher = LightGlue(self.feature_name, **params)

    def forward(
        self,
        desc1: Tensor,
        desc2: Tensor,
        lafs1: Tensor,
        lafs2: Tensor,
        hw1: Optional[Tuple[int, int]] = None,
        hw2: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run forward.

        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.
            lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
            lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.
            hw1: Height/width of image.
            hw2: Height/width of image.

        Return:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2,
                shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.

        """
        if (desc1.shape[0] < 2) or (desc2.shape[0] < 2):
            return _no_match(desc1)
        keypoints1 = get_laf_center(lafs1)
        keypoints2 = get_laf_center(lafs2)
        if len(desc1.shape) == 2:
            desc1 = desc1.unsqueeze(0)
        if len(desc2.shape) == 2:
            desc2 = desc2.unsqueeze(0)
        dev = lafs1.device
        if hw1 is None:
            hw1_ = keypoints1.max(dim=1)[0].squeeze().flip(0)
        else:
            hw1_ = torch.tensor(hw1, device=dev)
        if hw2 is None:
            hw2_ = keypoints2.max(dim=1)[0].squeeze().flip(0)
        else:
            hw2_ = torch.tensor(hw2, device=dev)
        ori0 = deg2rad(get_laf_orientation(lafs1).reshape(1, -1))
        ori0[ori0 < 0] += 2.0 * pi
        ori1 = deg2rad(get_laf_orientation(lafs2).reshape(1, -1))
        ori1[ori1 < 0] += 2.0 * pi
        input_dict = {
            "image0": {
                "keypoints": keypoints1,
                "scales": get_laf_scale(lafs1).reshape(1, -1),
                "oris": ori0,
                "lafs": lafs1,
                "descriptors": desc1,
                "image_size": hw1_.flip(0).reshape(-1, 2).to(dev),
            },
            "image1": {
                "keypoints": keypoints2,
                "lafs": lafs2,
                "scales": get_laf_scale(lafs2).reshape(1, -1),
                "oris": ori1,
                "descriptors": desc2,
                "image_size": hw2_.flip(0).reshape(-1, 2).to(dev),
            },
        }
        pred = self.matcher(input_dict)
        matches0, mscores0 = pred["matches0"], pred["matching_scores0"]
        valid = matches0 > -1
        matches = torch.stack([torch.where(valid)[1], matches0[valid]], -1)
        return mscores0[valid].reshape(-1, 1), matches
