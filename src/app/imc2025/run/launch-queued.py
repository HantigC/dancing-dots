"""
Convenience launcher: auto-detects GPU count, then delegates to the
queue-based distributed runner.

    uv run launch-distributed \\
        config/pipeline/imc2025/0009/base.yaml \\
        [merge1.yaml ...] \\
        [--workers-per-gpu 2] \\
        [-o datasets_names=[ETs,stairs]]

This is equivalent to ``run-distributed`` with ``--num-gpus`` set
automatically via ``torch.cuda.device_count()``.
"""

import os

import click
import torch
from omegaconf import OmegaConf

from app.constants import DEBUG
from app.imc2025.run.distributed import _build_rank_devices, _run
from app.setup import setup_from_env


@click.command()
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("merge", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--override",
    "-o",
    multiple=True,
    metavar="KEY=VALUE",
    help="Dot-notation overrides forwarded to the runner.",
)
@click.option("--environment", "-e", default=DEBUG, show_default=True)
@click.option(
    "--workers-per-gpu",
    default=2,
    show_default=True,
    help="Worker processes per GPU.",
)
@click.option(
    "--num-gpus",
    default=None,
    type=int,
    help="GPUs to use (default: torch.cuda.device_count()).",
)
def main(
    config_file: str,
    merge: tuple,
    override: tuple,
    environment: str,
    workers_per_gpu: int,
    num_gpus: int | None,
) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    setup_from_env(environment)

    cfg = OmegaConf.load(config_file)
    for path in merge:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(override)))

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise click.UsageError(
            "No CUDA GPUs detected. Pass --num-gpus explicitly or check your environment."
        )

    rank_devices = _build_rank_devices(cfg, num_gpus, workers_per_gpu)
    click.echo(
        f"Launching {len(rank_devices)} worker(s) "
        f"({workers_per_gpu} per GPU) across {num_gpus} GPU(s).\n"
        f"  devices: {rank_devices}"
    )
    _run(cfg, rank_devices)


if __name__ == "__main__":
    main()
