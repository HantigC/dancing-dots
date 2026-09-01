from typing import Any

import torch
from tqdm.auto import tqdm

from mts.core.extractor.base import BaseExtractor
from mts.pipeline.repository.base import SceneScopedImageRepository
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import PerSceneStep
from mts.utils.torchx import to_torch_format


class TorchExtractStep(PerSceneStep):
    def __init__(
        self,
        extractor: BaseExtractor,
        keypoints_name: str = "keypoints",
        descriptors_name: str = "descriptors",
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.keypoints_name = keypoints_name
        self.descriptors_name = descriptors_name

    def _extract(
        self,
        image_repository: ImageRepository,
    ) -> None:
        self.extractor.eval()
        device = self.device
        with torch.inference_mode():
            for image_index in tqdm(
                image_repository.image_ids(),
                total=image_repository.images_num(),
                desc="Extract keypoints and descriptors",
            ):
                image = image_repository.load_image(image_index)
                image = to_torch_format(image)
                image = image.to(device)
                keypoints, descriptors = self.extractor.extract(image)

                keypoints = keypoints.detach().cpu().numpy()
                descriptors = descriptors.detach().cpu().numpy()

                image_repository.add_keypoints(image_index, keypoints, name=self.keypoints_name)
                image_repository.add_descriptors(image_index, descriptors, name=self.descriptors_name)

    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> None:
        self._extract(image_repository)
