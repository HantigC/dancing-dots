from typing import Any, Sequence


def extract_kwargs(
    kwargs: dict[str, str | int | list],
    names: list[str] | str,
    merge: bool = False,
) -> dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not isinstance(names, list):
        if isinstance(names, str):
            names = [names]
        else:
            raise ValueError(f"What dict should be of type list[str] | str, not {names.__class__.name}")

    what_dict: dict[str, dict[str, Any]] = {w: {} for w in names}
    left_dict = {}
    for k, v in kwargs.items():
        try:
            group, name = k.split(":", 1)
        except ValueError:
            if merge:
                for wv in what_dict.values():
                    wv[k] = v
            else:
                left_dict[k] = v
        else:
            group_dict = what_dict.get(group)
            if group_dict is not None:
                group_dict[name] = v
            else:
                if merge:
                    for wv in what_dict.values():
                        wv[k] = v
                else:
                    left_dict[k] = v
    if merge:
        return what_dict
    return what_dict, left_dict


def dict_from_dups(dups: Sequence[tuple[Any, Any]]) -> dict[Any, list[Any]]:
    dictt = {}
    for k, v in dups:
        dictt.setdefault(k, []).append(v)
    return dictt
