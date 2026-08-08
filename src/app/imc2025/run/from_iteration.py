import shutil
from pathlib import Path

import click
from omegaconf import OmegaConf

from app.constants import DEBUG
from app.imc2025.run.from_config import run_from_cfg
from app.setup import setup_from_env
from mts.helpers.project.git_project import GitProject


@click.command()
@click.argument("source_iteration", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--override",
    "-o",
    multiple=True,
    metavar="KEY=VALUE",
    help="Dot-notation overrides applied on top of the source config, e.g. -o datasets_names=[ETs].",
)
@click.option("--environment", "-e", default=DEBUG, show_default=True)
def main(source_iteration: str, override: tuple, environment: str):
    setup_from_env(environment)

    source_dirpath = Path(source_iteration)
    cfg = OmegaConf.load(source_dirpath / "config.yaml")

    if override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(override)))

    cfg.origin = str(source_dirpath)

    project = GitProject.from_next_iteration(
        cfg.project_path,
        create=False,
        save=cfg.save_project_to_git,
    )
    project.__enter__()

    source_h5_dir = source_dirpath / "h5_repositories"
    if source_h5_dir.exists():
        dest_h5_dir = project.iteration_dirpath / "h5_repositories"
        dest_h5_dir.mkdir(exist_ok=True)
        for h5_file in source_h5_dir.glob("*.h5"):
            shutil.copy2(h5_file, dest_h5_dir / h5_file.name)

    run_from_cfg(cfg, project=project)


if __name__ == "__main__":
    main()
