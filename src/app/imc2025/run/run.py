import os
import click
from omegaconf import OmegaConf

from app.constants import DEBUG
from app.imc2025.run.from_config import run_from_cfg
from app.setup import setup_from_env


@click.command()
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("merge", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--override", "-o",
    multiple=True,
    metavar="KEY=VALUE",
    help="Dot-notation overrides applied after merges, e.g. -o datasets_names=[ETs].",
)
@click.option("--environment", "-e", default=DEBUG, show_default=True)
def main(config_file: str, merge: tuple, override: tuple, environment: str):
    os.environ["HYDRA_FULL_ERROR"] = "1"

    setup_from_env(environment)

    cfg = OmegaConf.load(config_file)

    for path in merge:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))

    if override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(override)))

    run_from_cfg(cfg)


if __name__ == "__main__":
    main()
