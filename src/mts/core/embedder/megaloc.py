
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms

from mts.core.embedder.base import BaseEmbedder
from mts.helpers.gmberton_MegaLoc_main.megaloc_model import MegaLoc
from mts.helpers.torch import nn as nnx
from mts.utils.torchx import to_torch_format


class MegaLocDescriptors(
    BaseEmbedder[
        torch.Tensor,
        torch.Tensor,
    ],
    nnx.DeviceMixin,
    nn.Module,
):
    def __init__(self, model) -> None:
        super(MegaLocDescriptors, self).__init__()
        self.model = model

        transformations = [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(transformations)

    @torch.no_grad
    def embed_image(self, image: np.ndarray) -> torch.Tensor:
        inputs = self.transform(to_torch_format(image).to(torch.float32))
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        outputs.squeeze_()
        return outputs

    @classmethod
    def from_pretrained(
        cls, weights_path: str = "/Users/stefan-cristianhantig/.cache/huggingface/hub/models--gberton--MegaLoc/snapshots/7cb9f7970d366fdf059963d04d372e503e8e9df9/model.safetensors"
    ):
    
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)

        model = MegaLoc()

        model.load_state_dict(state_dict)
        return cls(model)
