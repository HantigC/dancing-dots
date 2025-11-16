from typing import NamedTuple

import torch
from torch import nn


class ModuleParamInfo(NamedTuple):
    device: torch.device
    dtype: torch.dtype


def get(module: nn.Module) -> ModuleParamInfo:
    param = next(module.parameters())
    return ModuleParamInfo(
        param.device,
        param.dtype,
    )
