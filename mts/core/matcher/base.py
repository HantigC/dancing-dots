from abc import ABC, abstractmethod

import torch


class BaseMatcher(ABC):
    @abstractmethod
    def match(
        self,
        st_kp: torch.Tensor,
        nd_kp: torch.Tensor,
        st_descriptors: torch.Tensor,
        nd_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        pass
