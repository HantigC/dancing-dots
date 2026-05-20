from __future__ import annotations

from pathlib import Path

from mts.helpers.project.base_project import BaseProject, NoIterationException

__all__ = ["Project", "NoIterationException"]


class Project(BaseProject):
    @staticmethod
    def _list_iterations(project_dirpath: Path) -> list[Path]:
        return list(project_dirpath.glob("*"))
