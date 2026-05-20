import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import click

from app.constants import DEBUG
from app.imc2025.run.from_config import run_from_cfg


@click.command()
@click.option("--pipeline", "-p", default="0005", show_default=True, help="Pipeline version directory (e.g. 0005)")
@click.option("--config", "-c", default="base", show_default=True, help="Config name inside the pipeline directory (without .yaml)")
@click.option("--environment", "-e", default=DEBUG, show_default=True, help="Environment (run or debug)")
def main(pipeline: str, config: str, environment: str):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from app.setup import setup_from_env

    setup_from_env(environment)

    os.environ["HYDRA_FULL_ERROR"] = "1"

    config_dir = Path(f"config/pipeline/imc2025/{pipeline}").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config)

    mps_path = config_dir / "mps.yaml"
    if mps_path.exists():
        mps = OmegaConf.load(mps_path)
        cfg = OmegaConf.merge(cfg, mps)

    cfg.datasets_names = [
        "pt_piazzasanmarco_grandplace",
        "ETs",
        "imc2023_haiper",
        # "imc2023_theather_imc2024_church",
        # "amy_gardens",
        # "fbk_vineyard",
        # "pt_brandenburg_british_buckingham",
        # "imc2023_heritage",
        # "pt_sacrecoeur_trevi_tajmahal",
        # "pt_stpeters_stpauls",
        # "imc2024_dioscuri_baalshamin",
        # "imc2024_lizard_pond",
        # "stairs",
    ]

    run_from_cfg(cfg)


if __name__ == "__main__":
    main()
