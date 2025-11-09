from typing import Any, Sequence


def extract_kwargs(
    kwargs: dict[str],
    what_dict: list[str] | str,
    merge: bool = False,
) -> tuple[dict[str, dict], dict]:
    if not isinstance(what_dict, list):
        what_dict = [what_dict]

    what_dict = {w: {} for w in what_dict}
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
