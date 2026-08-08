from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from mts.core.types import PathLike
from mts.pipeline.repository.h5 import H5ImageRepository


def _read_iteration_labels(iteration_dir: Path) -> tuple[str | None, str | None]:
    config_path = iteration_dir / "config.yaml"
    if not config_path.exists():
        return None, None
    cfg = OmegaConf.load(config_path)
    tag = cfg.get("tag") or None
    mark = cfg.get("mark") or None
    return tag, mark


def collect_step_timings(iteration_dirs: list[PathLike]) -> pd.DataFrame:
    """Read step_timings from every H5 repository across the given iteration directories.

    Returns a DataFrame with columns: iteration, dataset, tag, mark, step, duration_s.
    """
    rows = []
    for iteration_dir in iteration_dirs:
        iteration_dir = Path(iteration_dir)
        tag, mark = _read_iteration_labels(iteration_dir)
        h5_dir = iteration_dir / "h5_repositories"
        if not h5_dir.exists():
            continue
        for h5_path in sorted(h5_dir.glob("*.h5")):
            repo = H5ImageRepository(h5_path)
            timings = repo.get_repository_metadata("step_timings")
            if timings is None:
                continue
            for step, duration in timings.items():
                rows.append(
                    {
                        "iteration": iteration_dir.name,
                        "dataset": h5_path.stem,
                        "tag": tag,
                        "mark": mark,
                        "step": step,
                        "duration_s": float(duration),
                    }
                )
    return pd.DataFrame(
        rows, columns=["iteration", "dataset", "tag", "mark", "step", "duration_s"]
    )


def aggregate_step_timings(iteration_dirs: list[PathLike]) -> pd.DataFrame:
    """Aggregate step durations grouped by tag, with each mark as a column.

    Returns a DataFrame with a (tag, step) MultiIndex and one column per mark
    containing the mean duration in seconds. Rows are sorted by total time
    descending within each tag.
    """
    df = collect_step_timings(iteration_dirs)
    if df.empty:
        return pd.DataFrame()

    tagged = df[df["tag"].notna() & df["mark"].notna()]
    if tagged.empty:
        return pd.DataFrame()

    mean_df = (
        tagged.groupby(["tag", "step", "mark"])["duration_s"]
        .mean()
        .reset_index()
    )

    pivoted = mean_df.pivot_table(
        index=["tag", "step"],
        columns="mark",
        values="duration_s",
    )
    pivoted.columns.name = None

    total = pivoted.sum(axis=1)
    pivoted = pivoted.assign(_total=total)
    pivoted = (
        pivoted.sort_values(["tag", "_total"], ascending=[True, False])
        .drop(columns="_total")
    )
    return pivoted


def aggregate_step_timings_from_project(
    project_dir: PathLike,
    iterations: list[str | int] | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: resolve iteration directories from a project root.

    Pass `iterations` as names (e.g. ["0313", "0318"]) or ints (e.g. [313, 318]).
    If None, all numeric subdirectories are used.
    """
    project_dir = Path(project_dir)
    if iterations is None:
        iteration_dirs = sorted(
            p for p in project_dir.iterdir() if p.is_dir() and p.name.isdigit()
        )
    else:
        iteration_dirs = [
            project_dir / (f"{int(i):04d}" if isinstance(i, int) else str(i))
            for i in iterations
        ]
    return aggregate_step_timings(iteration_dirs)
