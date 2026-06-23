from pathlib import Path
from typing import Any, NamedTuple, TypedDict, TypeVar

import numpy as np

T = TypeVar("T")

PairType = tuple[T, T]
ImageId = int
PathLike = str | Path
StateType = dict[str, Any]
Pairs = list[PairType[T]]


class DistancedTriple(NamedTuple):
    st: int
    nd: int
    distance: float


class Rigid3dDict(TypedDict):
    R: np.ndarray
    t: np.ndarray

