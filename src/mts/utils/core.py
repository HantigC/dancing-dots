from typing import ClassVar, get_origin


def collect_classvars(cls):
    classvars = {}
    for name, annotation in cls.__annotations__.items():
        if get_origin(annotation) is ClassVar:
            classvars[name] = getattr(cls, name)
    cls.__classvars__ = classvars
    return cls
