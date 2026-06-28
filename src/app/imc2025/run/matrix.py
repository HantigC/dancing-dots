import itertools
import os

import click
from omegaconf import OmegaConf

from app.constants import DEBUG
from app.imc2025.run.from_config import run_from_cfg
from app.setup import setup_from_env
from mts.helpers.project.git_project import GitProject


def _parse_matrix(param: str) -> tuple[str, list[str]]:
    key, _, values_str = param.partition("=")
    return key.strip(), [v.strip() for v in values_str.split(",")]


@click.command()
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("merge", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("--override", "-o", multiple=True, metavar="KEY=VALUE")
@click.option(
    "--matrix",
    "-m",
    multiple=True,
    metavar="KEY=V1,V2,...",
    help="Sweep values (comma-separated). Multiple -m flags produce Cartesian product.",
)
@click.option("--environment", "-e", default=DEBUG, show_default=True)
def main(config_file, merge, override, matrix, environment):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    setup_from_env(environment)

    base_cfg = OmegaConf.load(config_file)
    for path in merge:
        base_cfg = OmegaConf.merge(base_cfg, OmegaConf.load(path))
    if override:
        base_cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(list(override)))

    if not matrix:
        run_from_cfg(base_cfg)
        return

    keys, value_lists = zip(*[_parse_matrix(m) for m in matrix])
    combinations = list(itertools.product(*value_lists))

    # Allocate one top-level iteration directory for the entire matrix run.
    # Sub-runs create their own numbered subdirs inside it.
    matrix_project = GitProject.from_next_iteration(
        base_cfg.project_path,
        create=True,
        save=False,
    )
    matrix_iter_path = matrix_project.iteration_dirpath
    click.echo(f"Matrix iteration: {matrix_iter_path}")

    for i, combo in enumerate(combinations, 1):
        param_overrides = [f"{k}={v}" for k, v in zip(keys, combo)]
        click.echo(f"[{i}/{len(combinations)}] {', '.join(param_overrides)}")
        cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(param_overrides))
        cfg.project_path = str(matrix_iter_path)
        run_from_cfg(cfg)
