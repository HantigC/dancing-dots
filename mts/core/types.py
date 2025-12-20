from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

PairType = tuple[T, T]
ImageId = int
PathLike = str | Path
StateType = dict[str, Any]
