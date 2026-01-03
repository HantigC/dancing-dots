import logging
import yaml

from app.constants import RUN, DEBUG, Environment


def setup(environment: Environment) -> None:
    logging_config_map = {
        RUN: "config/logging/run.yaml",
        DEBUG: "config/logging/debug.yaml",
    }

    with open(logging_config_map[environment], "rt") as f:
        config = yaml.safe_load(f.read())

    logging.config.dictConfig(config)