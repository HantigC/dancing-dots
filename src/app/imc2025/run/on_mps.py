import logging
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import click
from .run import main as _run_main

LOGGER = logging.getLogger(__name__)


class _MpsCommand(click.Command):
    def invoke(self, ctx):
        LOGGER.info("Running pipeline on MPS (Apple Silicon)")
        return super().invoke(ctx)


main = _MpsCommand(
    name=_run_main.name or "main",
    callback=_run_main.callback,
    params=_run_main.params,
    help=_run_main.help,
)

if __name__ == "__main__":
    main()
