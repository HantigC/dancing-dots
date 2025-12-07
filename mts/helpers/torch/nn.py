from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class ModuleParamInfo(NamedTuple):
    device: torch.device
    dtype: torch.dtype

    def __eq__(self, other: ModuleParamInfo) -> bool:
        return self.device == other.device and self.dtype == other.dtype


def get(module: nn.Module) -> ModuleParamInfo:
    param = next(module.parameters())
    return ModuleParamInfo(
        param.device,
        param.dtype,
    )


class DeviceMixin:
    @property
    def device(self) -> torch.device:
        return get(self).device

