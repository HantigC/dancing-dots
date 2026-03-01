from __future__ import annotations
from torch import nn
import torch
from mts.helpers.torch import nn as nnx
from .base import BaseMatcher
import kornia.feature as KF


class LightGlueMatcher(
    BaseMatcher,
    nnx.DeviceMixin,
    nn.Module,
):
    def __init__(self, lg_matcher) -> None:
        super().__init__()
        self.lg_matcher = lg_matcher

    def match(
        self,
        st_kp: torch.Tensor,
        nd_kp: torch.Tensor,
        st_descriptors: torch.Tensor,
        nd_descriptors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dists, idxs = self.lg_matcher(
            st_descriptors,
            nd_descriptors,
            KF.laf_from_center_scale_ori(st_kp[None]),
            KF.laf_from_center_scale_ori(nd_kp[None]),
        )
        return dists, idxs 

    @classmethod
    def from_config(cls, feature_name: str, params: dict | None = None) -> LightGlueMatcher:
        return cls(KF.LightGlueMatcher(feature_name, params))
