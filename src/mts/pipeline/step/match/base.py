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
        keypoints_name: str = "keypoints",
        descriptors_name: str = "descriptors",
        matches_name: str = "matches",
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.min_matches = min_matches
        self.keypoints_name = keypoints_name
        self.descriptors_name = descriptors_name
        self.matches_name = matches_name

    @use_image_repository
    @torch.no_grad
    def run(self, image_repository: ImageRepository) -> None:
        device = self.device
        with torch.inference_mode():
            for pair_idx in tqdm(
                image_repository.get_pairs(),
                desc="Match the keypoints and descriptors",
            ):
                idx1, idx2 = pair_idx

                kps1 = torch.from_numpy(
                    image_repository.get_keypoints(idx1, name=self.keypoints_name),
                ).to(device)
                kps2 = torch.from_numpy(
                    image_repository.get_keypoints(idx2, name=self.keypoints_name),
                ).to(device)
                desc1 = torch.from_numpy(
                    image_repository.get_descriptors(idx1, name=self.descriptors_name),
                ).to(device)
                desc2 = torch.from_numpy(
                    image_repository.get_descriptors(idx2, name=self.descriptors_name),
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
                    image_repository.add_matches(
                        idx1,
                        idx2,
                        idxs.detach().cpu().numpy().reshape(-1, 2),
                        name=self.matches_name,
                    )
