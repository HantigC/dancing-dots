from typing import Sequence

import torch
from tqdm.auto import tqdm

from mts.core.extractor.base import BaseExtractor
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import BasePipelineStep
from mts.utils.torchx import to_torch_format


class TorchExtractStep(BasePipelineStep):
    def __init__(
        self,
        image_repository: ImageRepository,
        extractor: BaseExtractor,
    ) -> None:
        super().__init__()
        self.repository = image_repository
        self.extractor = extractor

    def _extract(self, image_indices: Sequence[int]) -> None:
        self.extractor.eval()
        with torch.inference_mode():
            for image_index in tqdm(
                image_indices,
                total=self.repository.images_num(),
                desc="Extract keypoints and descriptors",
            ):
                image = self.repository.load_image(image_index)
                image = to_torch_format(image)
                image = image.to(self.device)
                keypoints, descriptors = self.extractor.extract(image)

                keypoints = keypoints.detach().cpu().numpy()
                descriptors = descriptors.detach().cpu().numpy()

                self.repository.add_keypoints(image_index, keypoints)
                self.repository.add_descriptors(image_index, descriptors)

    def run(
        self,
    ) -> None:
        self._extract(self.repository.image_ids())
