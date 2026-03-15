from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from mts.core.types import PathLike


class NoIterationException(BaseException):
    """Raised in case there is no iteration"""


class BaseProject(ABC):
    def __init__(
        self,
        project_dir: PathLike,
        iteration_name: str,
        create: bool = True,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.iteration_name = iteration_name
        if create:
            self._create()

    @property
    def iteration_dirpath(self) -> Path:
        return self.project_dir / self.iteration_name

    def _create(self):
        self.iteration_dirpath.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> BaseProject:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    @classmethod
    def _make_first_iter(cls, resolution: int = 4) -> str:
        return cls._make_iteration_name(1, resolution=resolution)

    @staticmethod
    def _make_iteration_name(index: int, resolution: int = 4) -> str:
        return f"{index:0{resolution}d}"

    @staticmethod
    def _get_last_iter(iterations_dirpaths: list[Path]) -> str:
        iterations = []
        for iteration_dirpath in iterations_dirpaths:
            try:
                iteration_num = int(iteration_dirpath.name)
            except ValueError:
                pass
            else:
                iterations.append((iteration_num, iteration_dirpath))

        if len(iterations) == 0:
            raise NoIterationException("There is no automatically created iteration")
        last_iteration_dirpath: Path = max(iterations, key=lambda x: x[0])[1]
        return last_iteration_dirpath.name

    @staticmethod
    @abstractmethod
    def _list_iterations(project_dirpath: Path) -> list[Path]:
        """Return paths of existing iteration directories."""

    @classmethod
    def from_next_iteration(
        cls,
        project_dirpath: PathLike,
        resolution: int = 4,
        **kwargs,
    ) -> BaseProject:
        project_dirpath = Path(project_dirpath)
        iterations = cls._list_iterations(project_dirpath)
        if len(iterations) == 0:
            next_iteration_name = cls._make_first_iter(resolution)
        else:
            try:
                last_iteration_name = cls._get_last_iter(iterations)
                next_index = int(last_iteration_name) + 1
                next_iteration_name = cls._make_iteration_name(next_index, resolution)
            except NoIterationException:
                next_iteration_name = cls._make_first_iter(resolution)
        return cls(project_dirpath, next_iteration_name, **kwargs)

    @classmethod
    def from_last_iteration(
        cls,
        project_dirpath: PathLike,
        **kwargs,
    ) -> BaseProject:
        project_dirpath = Path(project_dirpath)
        iterations = cls._list_iterations(project_dirpath)
        if len(iterations) == 0:
            raise NoIterationException("There is no iteration")
        last_iteration_name = cls._get_last_iter(iterations)
        return cls(project_dirpath, last_iteration_name, **kwargs)
