import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path

import click
from omegaconf import OmegaConf

from app.constants import DEBUG
from app.imc2025.run.from_config import run_from_cfg
from app.setup import setup_from_env


@click.command()
@click.option("--pipeline", "-p", default="0005", show_default=True, help="Pipeline version directory (e.g. 0005)")
@click.option("--config", "-c", default="base", show_default=True, help="Config name inside the pipeline directory (without .yaml)")
@click.option(
    "--override", "-o",
    multiple=True,
    metavar="KEY=VALUE",
    help="Dot-notation overrides applied last, e.g. -o datasets_names=[ETs].",
)
@click.option("--environment", "-e", default=DEBUG, show_default=True, help="Environment (run or debug)")
def main(pipeline: str, config: str, override: tuple, environment: str):
    setup_from_env(environment)

    config_dir = Path(f"config/pipeline/imc2025/{pipeline}").resolve()
    cfg = OmegaConf.load(config_dir / "base.yaml")

    if config != "base":
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_dir / f"{config}.yaml"))

    mps_path = config_dir / "mps.yaml"
    if mps_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(mps_path))

    if override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(override)))

    run_from_cfg(cfg)


if __name__ == "__main__":
    main()
