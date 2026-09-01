import logging
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Hashable

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from mts.core.types import PathLike, StateType
from mts.helpers.torch.nn import DeviceMixin
from mts.pipeline.repository.base import SceneScopedImageRepository
from mts.pipeline.repository.inmemeory import ImageRepository

LOGGER = logging.getLogger(__name__)


class BasePipelineStep(ABC, DeviceMixin, nn.Module):

    def __init__(self, name=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._name = name or self.__class__.__name__

    def name(self) -> str:
        return self._name

    @abstractmethod
    def run(
        self,
        *,
        image_repository: ImageRepository,
        input: Any,
        state: StateType,
    ) -> Any:
        pass


class OnDeviceRunner(BasePipelineStep):
    def __init__(
        self,
        step: BasePipelineStep,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._run_on_device = device
        self.step = step

    def name(self) -> str:
        if self._name is None:
            return self.step.name()
        return self._name

    def run(
        self,
        *,
        image_repository: ImageRepository,
        input: Any,
        state: StateType,
    ) -> Any:

        LOGGER.info(
            "Running step '%s' on device %s",
            self.step.__class__.__name__,
            self._run_on_device,
        )
        prev_device = self.step.device
        self.step.to(self._run_on_device)
        result = self.step.run(
            image_repository=image_repository,
            input=input,
            state=state,
        )
        self.step.to(prev_device)
        return result


class PerSceneStep(BasePipelineStep):
    """A pipeline step that runs once per scene in the repository.

    ``run`` iterates ``image_repository.scenes()`` (deterministic order) and
    calls the subclass' ``run_scene`` once per scene. Per-scene exceptions
    are logged and skipped so one bad scene does not abort the rest of the
    run (this mirrors the old per-dataset isolation in
    ``IMC2025Pipeline.run``). ``run`` returns ``{scene: run_scene_result}``.

    ``run_scene`` receives a :class:`SceneScopedImageRepository` as its
    ``image_repository`` -- image enumeration, pairs and ``store``/``load``
    are already scoped to ``scene`` -- so step bodies can stay
    scene-agnostic. The unscoped repository is reachable via
    ``image_repository.unscoped`` if ever needed.

    Per-scene scratch state lives at ``state["scenes"][scene]`` and is
    passed to ``run_scene`` as ``scene_state``; the shared ``state`` (with
    ``colmap_dirpath`` / ``images_dir``) is passed as ``state`` as well.

    When wrapped in ``OnDeviceRunner`` the inner step is moved to the target
    device once, around the whole scene loop -- model load is the expensive
    part, so this is intentional.
    """

    def run(
        self,
        *,
        image_repository: ImageRepository,
        input: Any = None,
        state: StateType | None = None,
    ) -> dict[str, Any]:
        state = state if state is not None else {}
        scenes_state = state.setdefault("scenes", {})
        results: dict[str, Any] = {}
        for scene in image_repository.scenes():
            scene_state = scenes_state.setdefault(scene, {})
            LOGGER.info("Step '%s' -- scene '%s'", self.name(), scene)
            try:
                results[scene] = self.run_scene(
                    image_repository=SceneScopedImageRepository(
                        image_repository, scene
                    ),
                    scene=scene,
                    input=input,
                    state=state,
                    scene_state=scene_state,
                )
            except Exception:
                LOGGER.exception(
                    "Step '%s' failed for scene '%s', skipping",
                    self.name(),
                    scene,
                )
        return results

    @abstractmethod
    def run_scene(
        self,
        *,
        image_repository: SceneScopedImageRepository,
        scene: str,
        input: Any = None,
        state: dict[str, Any] | None = None,
        scene_state: dict[str, Any] | None = None,
    ) -> Any:
        pass


def use_image_repository(
    fn: Callable[..., Any] = None,
    *,
    params: list[str] | None = None,
):
    params = params or []
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


class StateKeyError(Exception):
    pass


class State:
    def __init__(self) -> None:
        self._state_map: dict[Hashable, Any] = {}

    def add(self, key: Hashable, value: Any) -> None:
        self._state_map[key] = value

    def at(self, key: Hashable) -> Any:
        try:
            value = self._state_map[key]
        except KeyError as e:
            raise StateKeyError(f"key `{key}` cannot be found in state") from e
        else:
            return value


def run_pipeline(
    steps: list[BasePipelineStep] | None = None,
    *,
    image_repository: ImageRepository,
    input: Any | None = None,
    state: StateType | None = None,
) -> Any:
    state = state or {}
    step_timings = {}
    for step in steps:
        step_name = step.name()
        LOGGER.info("Running step '%s'", step_name)
        t0 = time.perf_counter()
        input = step.run(
            image_repository=image_repository,
            input=input,
            state=state,
        )
        elapsed = time.perf_counter() - t0
        step_timings[step_name] = elapsed
        LOGGER.info("Step '%s' finished in %.2fs", step_name, elapsed)
    image_repository.upsert_repository_metadata(step_timings=step_timings)
    return input


def from_hydra_config_file(config_filepath: PathLike) -> list[BasePipelineStep]:
    cfg = OmegaConf.load(config_filepath)
    return from_hydra_config(cfg)


def from_hydra_config(cfg) -> list[BasePipelineStep]:
    pipeline_steps = []
    for name, step in cfg.pipeline_steps.items():
        step_instance: BasePipelineStep = instantiate(step)
        step_instance._name = name
        pipeline_steps.append(step_instance)
    return pipeline_steps
