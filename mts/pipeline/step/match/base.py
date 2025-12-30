import logging

import torch
from tqdm.auto import tqdm

from mts.core.matcher.base import BaseMatcher
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import BasePipelineStep, use_image_repository

LOGGER = logging.getLogger(__name__)


class MatchingStep(BasePipelineStep):
    def __init__(
        self,
        matcher: BaseMatcher,
        min_matches: int = 50,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.min_matches = min_matches

    @use_image_repository
    @torch.no_grad
    def run(self, image_repository: ImageRepository) -> None:
        device = self._run_on_device
        with torch.inference_mode():
            for pair_idx in tqdm(image_repository.get_pairs()):
                idx1, idx2 = pair_idx

                kps1 = torch.from_numpy(
                    image_repository.get_keypoints(idx1),
                ).to(device)
                kps2 = torch.from_numpy(
                    image_repository.get_keypoints(idx2),
                ).to(device)
                desc1 = torch.from_numpy(
                    image_repository.get_descriptors(idx1),
                ).to(device)
                desc2 = torch.from_numpy(
                    image_repository.get_descriptors(idx2),
                ).to(device)

                dists, idxs = self.matcher.match(
                    kps1,
                    kps2,
                    desc1,
                    desc2,
                )
                if len(idxs) == 0:
                    continue

                n_matches = len(idxs)
                if n_matches >= self.min_matches:
                    LOGGER.info(
                        "Matched (%d, %d) having %d of matches", idx1, idx2, len(idxs)
                    )
                    image_repository.add_matches(
                        idx1,
                        idx2,
                        idxs.detach().cpu().numpy().reshape(-1, 2),
                    )
