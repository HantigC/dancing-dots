import logging
import logging.config
import yaml
from pathlib import Path

from app.constants import RUN, DEBUG, Environment
from mts.core.types import PathLike


def setup(environment: Environment) -> None:
    logging_config_map = {
        RUN: "config/logging/run.yaml",
        DEBUG: "config/logging/debug.yaml",
    }

    with open(logging_config_map[environment], "rt") as f:
        config = yaml.safe_load(f.read())

    logging.config.dictConfig(config)


def setup_file(environment: Environment, project_path: PathLike) -> None:
    setup(environment)
    log_filepath = Path(project_path) / "run.log"
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
    for name in ("mts", "app"):
        logger = logging.getLogger(name)
        logger.addHandler(file_handler)