from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable

from torch import nn

from mts.core.types import StateType
from mts.helpers.torch.nn import DeviceMixin
from mts.pipeline.repository.inmemeory import ImageRepository


class BasePipelineStep(ABC, DeviceMixin, nn.Module):
    @abstractmethod
    def run(
        self,
        *,
        image_repository: ImageRepository,
        input: Any,
        state: StateType,
    ) -> Any:
        pass


def use_image_repository(
    fn: Callable[..., Any] = None,
    *,
    params: list[str] | None = None,
):
    return use_params(fn, params=["image_repository"] + params)


def use_no_params(fn: Callable[..., Any] = None):
    return use_params(fn, params=[])


class ParamsException(BaseException):
    """Exception in case the parameters are not provided correctly"""


def use_params(fn: Callable[..., Any] = None, *, params: list[str] | None = None):
    def outer_wrapper(fn):
        @wraps(fn)
        def wrapper(self, **kwargs):
            try:
                arguments = {param: kwargs.get(param) for param in params}
            except KeyError as e:
                raise ParamsException("Parameters were not provided correctly") from e

            result = fn(self, **arguments)
            return result

        return wrapper

    if fn is not None:
        return outer_wrapper(fn)
    else:
        return outer_wrapper


def run_pipeline(
    steps: list[BasePipelineStep] | None = None,
    *,
    image_repository: ImageRepository,
    input: Any | None = None,
    state: StateType | None = None,
) -> Any:
    state = {}
    for step in steps:
        input = step.run(
            image_repository=image_repository,
            input=input,
            state=state,
        )
    return input
