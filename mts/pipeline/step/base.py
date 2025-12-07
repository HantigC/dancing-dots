from torch import nn

from abc import ABC, abstractmethod

from mts.helpers.torch.nn import DeviceMixin


class BasePipelineStep(ABC, DeviceMixin, nn.Module):
    @abstractmethod
    def run(
        self,
    ) -> None:
        pass