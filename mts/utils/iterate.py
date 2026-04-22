import itertools as it
from typing import Callable, Hashable, Sequence, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


def identity(item: K) -> K:
    return item


def group_by(
    items: Sequence[T],
    key: Callable[[T], K] | None = None,
) -> dict[K, list[T]]:
    if key is None:
        key = identity

    grouped_dict: dict[K, list[T]] = {}
    for item in items:
        grouped_dict.setdefault(key(item), []).append(item)
    return grouped_dict


def cycle_or_no(items: list[T], default: T | None = None) -> it.cycle:
    if items is None:
        items = it.cycle([default])
    elif isinstance(items, list) and len(items) == 1:
        items = it.cycle(items)
    return items
