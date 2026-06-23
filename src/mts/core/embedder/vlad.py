import numpy as np
import torch
from torch import nn

from mts.core.model.netvald import NetVLAD
from mts.helpers.torch import nn as nnx

from .base import BaseEmbedder


class NetVladEmbedding(
    BaseEmbedder[
        torch.Tensor,
        torch.Tensor,
    ],
    nnx.DeviceMixin,
    nn.Module,
):
    def __init__(self, cfg=None, model_location: str | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = {}
        if model_location is not None:
            torch.hub.set_dir(model_location)
        self.net_vlad = NetVLAD(cfg)

    def embed_image(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        img_t = self.img_to_vlad(img)
        data_dict = {"image": img_t}
        descriptor_dict = self.net_vlad(data_dict)

        global_descriptor = descriptor_dict["global_descriptor"].squeeze()
        return global_descriptor

    def to(self, device, **kwargs):
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
