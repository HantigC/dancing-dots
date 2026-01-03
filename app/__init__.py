import os

from app.constants import APP_ENVIRONMENT, DEBUG
from config.logging import setup


def setup_from_env(environment: str = None):
    environment = environment or os.environ.get(APP_ENVIRONMENT, DEBUG)
    setup(environment)
