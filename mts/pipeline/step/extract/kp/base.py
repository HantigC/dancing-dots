import torch
from tqdm.auto import tqdm

from mts.core.extractor.base import BaseExtractor
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import BasePipelineStep, use_image_repository
from mts.utils.torchx import to_torch_format


class TorchExtractStep(BasePipelineStep):
    def __init__(
        self,
        extractor: BaseExtractor,
    ) -> None:
        super().__init__()
        self.extractor = extractor

    def _extract(
        self,
        image_repository: ImageRepository,
    ) -> None:
        self.extractor.eval()
        with torch.inference_mode():
            for image_index in tqdm(
                image_repository.image_ids(),
                total=image_repository.images_num(),
                desc="Extract keypoints and descriptors",
            ):
                image = self.repository.load_image(image_index)
                image = to_torch_format(image)
                image = image.to(self._run_on_device)
                keypoints, descriptors = self.extractor.extract(image)

                keypoints = keypoints.detach().cpu().numpy()
                descriptors = descriptors.detach().cpu().numpy()

                self.image_repository.add_keypoints(image_index, keypoints)
                self.image_repository.add_descriptors(image_index, descriptors)

    @use_image_repository
    def run(self, image_repository: ImageRepository) -> None:
        self._extract(image_repository)
