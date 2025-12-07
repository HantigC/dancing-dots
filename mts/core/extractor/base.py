from abc import ABC, abstractmethod
import torch
from torch import nn

from mts.helpers.torch.nn import DeviceMixin


class BaseExtractor(ABC, DeviceMixin, nn.Module):

    @abstractmethod
    def extract(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass
