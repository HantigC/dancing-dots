import torch
from lightglue import ALIKED

from mts.core.extractor.base import BaseExtractor


class AlikedExtractor(BaseExtractor):
    def __init__(self, num_features, resize_to, weights: str | None = None,) -> None:
        super().__init__()
        self.extractor = ALIKED(
            max_num_keypoints=num_features,
            detection_threshold=0.01,
            resize=resize_to,
            weights=weights,
        )

    def extract(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.to(torch.float32)
        image /= 255.0
        feats = self.extractor.extract(image)
        kpts = feats["keypoints"].reshape(-1, 2)
        descs = feats["descriptors"].reshape(len(kpts), -1)
        return kpts, descs
