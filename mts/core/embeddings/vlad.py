import numpy as np
from torch import nn
from hloc.extractors import netvlad
import torch
import gc

from .base import BaseEmbedder


class NetVladEmbedding(
    nn.Module,
    BaseEmbedder[
        np.ndarray | torch.Tensor,
        torch.Tensor,
    ],
):
    def __init__(self, cfg=None, device=torch.device("cuda:0")) -> None:
        super().__init__()
        if cfg is None:
            cfg = {}
        self.net_vlad = netvlad.NetVLAD(cfg).to(device)
        self.device = device

    def embed_image(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        img_t = self.img_to_vlad(img)
        data_dict = {"image": img_t}
        descriptor_dict = self.net_vlad(data_dict)

        global_descriptor = descriptor_dict["global_descriptor"].squeeze()
        del img_t
        gc.collect()
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
        img = img.permute(0, 3, 1, 2)
        return img
