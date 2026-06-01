import numpy as np
import torch
from hloc.extractors import netvlad
from torch import nn

from .base import BaseEmbedder


class NetVladEmbedding(
    nn.Module,
    BaseEmbedder[
        np.ndarray | torch.Tensor,
        torch.Tensor,
    ],
):
    def __init__(self, cfg=None, model_location: str | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = {}
        if model_location is not None:
            torch.hub.set_dir(model_location)
        self.net_vlad = netvlad.NetVLAD(cfg)

    def embed_image(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        img_t = self.img_to_vlad(img)
        data_dict = {"image": img_t}
        descriptor_dict = self.net_vlad(data_dict)

        global_descriptor = descriptor_dict["global_descriptor"].squeeze()
        return global_descriptor

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device=device, **kwargs)

    def img_to_vlad(self, img):
        if not isinstance(img, torch.Tensor):
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            else:
                raise ValueError("np or tensor")

        img = img.to(self.device, dtype=torch.float32)
        img = img / 255.0
        if img.ndim == 3:
            img.unsqueeze_(0)
        # img = img.permute(0, 3, 1, 2)
        return img
