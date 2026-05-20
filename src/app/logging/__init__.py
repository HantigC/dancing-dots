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


def setup_file_logging(project_path: PathLike) -> None:
    log_filepath = Path(project_path) / "run.log"
    # log_filepath.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_filepath)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    for logger in (logging.root, logging.getLogger("mts"), logging.getLogger("app")):
        logger.addHandler(file_handler)