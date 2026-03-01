from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass
class Image:
    height: int
    width: int

    @property
    def hw(self) -> tuple[int, int]:
        return self.height, self.width


class MatchKind(Enum):
    MATCHED = "Matched"
    MERGED = "Merged"


@dataclass
class TwoViewEdge:
    st_filepath: str
    nd_filepath: str
    kpts_for: dict[str, np.ndarray]
    match_kind: MatchKind
    num_matches: int
