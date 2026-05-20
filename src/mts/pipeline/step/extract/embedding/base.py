import torch
import tqdm

from mts.core.embedder.base import BaseEmbedder
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import BasePipelineStep, use_image_repository
from mts.utils.torchx import to_torch_format


class GlobalDescriptorStep(BasePipelineStep):
    def __init__(
        self,
        global_extractor: BaseEmbedder[torch.Tensor, torch.Tensor],
    ) -> None:
        super().__init__()
        self.global_extractor = global_extractor

    @torch.no_grad
    def _extract(
        self,
        image_repository: ImageRepository,
    ) -> None:
        self.global_extractor.eval()

        with torch.inference_mode():
            for image_index in tqdm.tqdm(
                image_repository.image_ids(),
                total=image_repository.images_num(),
                desc="Extract global the descriptors",
            ):
                image = image_repository.load_image(image_index)
                image = to_torch_format(image)
                image = image.to(self.device)

                global_descriptor = self.global_extractor.embed_image(image)
                global_descriptor = global_descriptor.detach().cpu().numpy()

                image_repository.add_global_descriptor(image_index, global_descriptor)

    @use_image_repository
    def run(self, image_repository: ImageRepository) -> None:
        self._extract(image_repository)
