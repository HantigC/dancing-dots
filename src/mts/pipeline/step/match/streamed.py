import gc
import logging
from typing import Any

import torch
from tqdm.auto import tqdm

from mts.core.matching.base import BaseMatcher
from mts.pipeline.repository.base import SceneScopedImageRepository
from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.base import PerSceneStep

LOGGER = logging.getLogger(__name__)


class StreamedMatchingStep(PerSceneStep):
    """MatchingStep that uses CUDA streams to overlap H2D transfers with GPU compute.

    Double-buffering pattern: while compute_stream runs the matcher on pair N,
    transfer_stream pre-fetches pair N+1 from pinned CPU memory to GPU.
    Falls back to sequential execution on non-CUDA devices (MPS, CPU).
    """

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

    def _load_pinned(self, image_repository: ImageRepository, idx1, idx2):
        return (
            torch.from_numpy(image_repository.get_keypoints(idx1, name=self.keypoints_name)).pin_memory(),
            torch.from_numpy(image_repository.get_keypoints(idx2, name=self.keypoints_name)).pin_memory(),
            torch.from_numpy(image_repository.get_descriptors(idx1, name=self.descriptors_name)).pin_memory(),
            torch.from_numpy(image_repository.get_descriptors(idx2, name=self.descriptors_name)).pin_memory(),
        )

    def _h2d(self, cpu_tensors, device):
        return tuple(t.to(device, non_blocking=True) for t in cpu_tensors)

    @torch.no_grad
    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> None:
        device = self.device
        device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]

        pairs = list(image_repository.get_pairs())
        if not pairs:
            return

        if device_type != "cuda":
            self._run_sequential(image_repository, pairs, device)
            return

        transfer_stream = torch.cuda.Stream(device=device)
        compute_stream = torch.cuda.Stream(device=device)

        with torch.inference_mode():
            cpu = self._load_pinned(image_repository, *pairs[0])
            with torch.cuda.stream(transfer_stream):
                gpu_tensors = self._h2d(cpu, device)
            del cpu

            for i, (idx1, idx2) in enumerate(tqdm(pairs, desc="Match keypoints (streamed)")):
                # Ensure current pair is fully on GPU before compute
                compute_stream.wait_stream(transfer_stream)
                kps1, kps2, desc1, desc2 = gpu_tensors

                # Pre-fetch next pair: CPU load + async H2D overlaps with GPU compute below
                if i + 1 < len(pairs):
                    gc.collect()
                    next_cpu = self._load_pinned(image_repository, *pairs[i + 1])
                    with torch.cuda.stream(transfer_stream):
                        next_gpu = self._h2d(next_cpu, device)
                    del next_cpu

                with torch.cuda.stream(compute_stream):
                    _, idxs = self.matcher.match(kps1, kps2, desc1, desc2)
                    idxs_cpu = idxs.detach().cpu()

                compute_stream.synchronize()
                del kps1, kps2, desc1, desc2

                idxs_np = idxs_cpu.numpy()
                if len(idxs_np) >= self.min_matches:
                    image_repository.add_matches(idx1, idx2, idxs_np.reshape(-1, 2), name=self.matches_name)

                gpu_tensors = next_gpu if i + 1 < len(pairs) else None

            torch.cuda.current_stream(device).wait_stream(compute_stream)

    def _run_sequential(self, image_repository: ImageRepository, pairs, device) -> None:
        with torch.inference_mode():
            for idx1, idx2 in tqdm(pairs, desc="Match keypoints"):
                gc.collect()
                kps1 = torch.from_numpy(image_repository.get_keypoints(idx1, name=self.keypoints_name)).to(device)
                kps2 = torch.from_numpy(image_repository.get_keypoints(idx2, name=self.keypoints_name)).to(device)
                desc1 = torch.from_numpy(image_repository.get_descriptors(idx1, name=self.descriptors_name)).to(device)
                desc2 = torch.from_numpy(image_repository.get_descriptors(idx2, name=self.descriptors_name)).to(device)

                _, idxs = self.matcher.match(kps1, kps2, desc1, desc2)
                idxs = idxs.detach().cpu().numpy()
                del kps1, kps2, desc1, desc2

                if len(idxs) >= self.min_matches:
                    image_repository.add_matches(idx1, idx2, idxs.reshape(-1, 2), name=self.matches_name)
