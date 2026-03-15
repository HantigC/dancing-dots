from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from mts.core.types import PathLike
from mts.helpers.project.base_project import BaseProject, NoIterationException

LOGGER = logging.getLogger(__name__)

__all__ = ["GitProject", "BranchOutOfSyncException"]


class BranchOutOfSyncException(Exception):
    """Raised when the local branch is not up-to-date with the remote."""


class GitProject(BaseProject):
    def __init__(
        self,
        project_dir: PathLike,
        iteration_name: str,
        remote: str = "origin",
        branch: str = "main",
        create: bool = True,
    ) -> None:
        self.remote = remote
        self.branch = branch
        self._check_in_sync()
        super().__init__(project_dir, iteration_name, create=create)

    @staticmethod
    def _list_iterations(project_dirpath: Path) -> list[Path]:
        return list(project_dirpath.glob("*"))

    def _check_in_sync(self) -> None:
        LOGGER.info("Check if everything is up to date")
        subprocess.run(["git", "fetch", self.remote], capture_output=True)

        local = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        remote_ref = subprocess.run(
            ["git", "rev-parse", f"{self.remote}/{self.branch}"],
            capture_output=True,
            text=True,
        ).stdout.strip()

        if local != remote_ref:
            raise BranchOutOfSyncException(
                f"Local branch is not in sync with '{self.remote}/{self.branch}'. "
                "Pull or push before creating a new iteration."
            )
        LOGGER.info("everything is up to date")

    def __enter__(self) -> GitProject:
        if not self.iteration_dirpath.exists():
            self._create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            return
        subprocess.run(
            ["git", "add", str(self.iteration_dirpath)],
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"add {self.iteration_name}"],
            check=True,
        )
        subprocess.run(
            ["git", "push", self.remote, self.branch],
            check=True,
        )

    @classmethod
    def from_next_iteration(
        cls,
        project_dirpath: PathLike,
        remote: str = "origin",
        branch: str = "main",
        resolution: int = 4,
        **kwargs,
    ) -> GitProject:
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
        return cls(
            project_dirpath, next_iteration_name, remote=remote, branch=branch, **kwargs
        )

    @classmethod
    def from_last_iteration(
        cls,
        project_dirpath: PathLike,
        remote: str = "origin",
        branch: str = "main",
        **kwargs,
    ) -> GitProject:
        project_dirpath = Path(project_dirpath)
        iterations = cls._list_iterations(project_dirpath)
        if len(iterations) == 0:
            raise NoIterationException("There is no iteration on the remote")
        last_iteration_name = cls._get_last_iter(iterations)
        return cls(
            project_dirpath,
            last_iteration_name,
            remote=remote,
            branch=branch,
            **kwargs,
        )
