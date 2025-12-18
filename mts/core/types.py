from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

PairType = tuple[T, T]
ImageId = int
PathLike = str | Path