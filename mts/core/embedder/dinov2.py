import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoImageProcessor, AutoModel

from mts.core.embedder.base import BaseEmbedder
from mts.helpers.torch import nn as nnx


class DinoV2GlobalDescriptors(
    BaseEmbedder[
        torch.Tensor,
        torch.Tensor,
    ],
    nnx.DeviceMixin,
    nn.Module,
):
    def __init__(self, processor, model) -> None:
        super(DinoV2GlobalDescriptors, self).__init__()
        self.processor = processor
        self.model = model

    @torch.no_grad
    def embed_image(self, image: np.ndarray) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False)
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        outputs = F.normalize(
            outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=1, p=2
        )
        outputs.squeeze_()
        return outputs

    @classmethod
    def from_pretrained(
        cls, model_location: str, processor_location: str | None = None
    ):
        processor_location = processor_location or model_location
        processor = AutoImageProcessor.from_pretrained(processor_location)
        model = AutoModel.from_pretrained(model_location)
        return cls(processor, model)
