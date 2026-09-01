from __future__ import annotations

import logging
from typing import Any

import lightglue
import torch
from tqdm.auto import tqdm
from lightglue.utils import rbd

from mts.pipeline.repository.base import BaseImageRepository, SceneScopedImageRepository
from mts.pipeline.step.base import PerSceneStep

LOGGER = logging.getLogger(__name__)


class LightGlueMatchStep(PerSceneStep):
    def __init__(
        self,
        features: str | None = None,
        weights: str | None = None,
        min_matches: int = 15,
        keypoints_name: str = "keypoints",
        descriptors_name: str = "descriptors",
        matches_name: str = "lightglue",
        reuse: bool = True,
    ) -> None:
        super().__init__()
        self.matcher = lightglue.LightGlue(features, **{"weights": weights} if weights else {})
        self.min_matches = min_matches
        self.keypoints_name = keypoints_name
        self.descriptors_name = descriptors_name
        self.matches_name = matches_name
        self.reuse = reuse

    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> Any:
        device = self.device
        pairs = list(image_repository.get_pairs())
        LOGGER.info("Matching %d pairs with LightGlue...", len(pairs))

        self.matcher.eval()
        with torch.inference_mode():
            for idx1, idx2 in tqdm(pairs, desc="LightGlue matching"):
                if (
                    self.reuse
                    and image_repository.get_matches(idx1, idx2, name=self.matches_name) is not None
                ):
                    continue

                feats0 = self._load_features(image_repository, idx1, device)
                feats1 = self._load_features(image_repository, idx2, device)

                result = self.matcher({"image0": feats0, "image1": feats1})
                result = rbd(result)
                matches = result["matches"].cpu().numpy()

                if len(matches) >= self.min_matches:
                    image_repository.add_matches(idx1, idx2, matches, name=self.matches_name)

    def _load_features(self, image_repository: BaseImageRepository, img_id, device) -> dict:
        kpts = torch.from_numpy(
            image_repository.get_keypoints(img_id, name=self.keypoints_name)
        ).unsqueeze(0).to(device)
        descs = torch.from_numpy(
            image_repository.get_descriptors(img_id, name=self.descriptors_name)
        ).unsqueeze(0).to(device)
        image = image_repository.load_image(img_id)
        h, w = image.shape[:2]
        image_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)
        return {"keypoints": kpts, "descriptors": descs, "image_size": image_size}
