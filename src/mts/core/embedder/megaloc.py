
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms

from mts.core.embedder.base import BaseEmbedder
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
        cls, model_location: str = "gmberton/MegaLoc"
    ):
        model = torch.hub.load(model_location, "get_trained_model")
        return cls(model)
