"""
Smart launcher for the distributed pipeline runner.

Discovers the dataset list from the config, then builds ``rank_devices`` and
``nproc_per_node`` automatically so that exactly ``--workers-per-gpu``
processes share each GPU.  Execs ``torchrun`` — no manual nproc counting
needed.

    uv run launch-distributed \\
        config/pipeline/imc2025/0009/base.yaml \\
        [merge1.yaml ...] \\
        [-o datasets_names=[ETs,stairs]] \\
        [--workers-per-gpu 2] \\
        [--num-gpus 4]          # defaults to torch.cuda.device_count()

Dataset→rank assignment is round-robin inside distributed.py, so each rank
receives an even slice.  With ``--workers-per-gpu 2`` and 4 GPUs you get 8
ranks: rank 0 and 1 share cuda:0, rank 2 and 3 share cuda:1, etc.
"""

import os
import subprocess
import sys
from pathlib import Path

import click
import torch
from omegaconf import OmegaConf

from app.constants import DEBUG
from app.imc2025.prediction import (
    load_from_csv,
    load_from_submission,
    load_test_images,
)
from app.imc2025.run.types import RunType
from app.setup import setup_from_env


def _discover_datasets(cfg) -> list[str]:
    """Return the ordered list of dataset names that the run will process."""
    data_dirpath = Path(cfg.get("data_dirpath", "data"))
    run_type = RunType(cfg.get("run_type", RunType.TRAIN))
    samples_filename = cfg.get("sample_filepath")

    if samples_filename:
        all_samples, _ = load_from_csv(data_dirpath, samples_filename)
    elif run_type == RunType.TRAIN:
        all_samples, _ = load_from_csv(data_dirpath, "train_labels.csv")
    elif run_type == RunType.SUBMISSION:
        all_samples, _ = load_from_submission(data_dirpath)
    else:
        all_samples, _ = load_test_images(data_dirpath)

    datasets_cfg = cfg.get("datasets_names")
    return list(datasets_cfg) if datasets_cfg is not None else list(all_samples.keys())


def _build_rank_devices(num_gpus: int, workers_per_gpu: int) -> list[int]:
    """Return [0,0,1,1,...] — GPU index for each rank."""
    return [gpu for gpu in range(num_gpus) for _ in range(workers_per_gpu)]


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("merge", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--override",
    "-o",
    multiple=True,
    metavar="KEY=VALUE",
    help="Dot-notation overrides forwarded to distributed.py.",
)
@click.option("--environment", "-e", default=DEBUG, show_default=True)
@click.option(
    "--backend",
    default="nccl",
    show_default=True,
    help="torch.distributed backend passed to distributed.py.",
)
@click.option(
    "--workers-per-gpu",
    default=2,
    show_default=True,
    help="Number of processes (ranks) that share each GPU.",
)
@click.option(
    "--num-gpus",
    default=None,
    type=int,
    help="GPUs to use (default: torch.cuda.device_count()).",
)
@click.option(
    "--torchrun",
    "torchrun_bin",
    default="torchrun",
    show_default=True,
    help="Path to the torchrun executable.",
)
def main(
    config_file: str,
    merge: tuple,
    override: tuple,
    environment: str,
    backend: str,
    workers_per_gpu: int,
    num_gpus: int | None,
    torchrun_bin: str,
) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    setup_from_env(environment)

    # Build config only to discover datasets — not used for the actual run.
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

    datasets = _discover_datasets(cfg)
    nproc = num_gpus * workers_per_gpu
    rank_devices = _build_rank_devices(num_gpus, workers_per_gpu)

    # Serialize rank_devices as an OmegaConf list override: [0,0,1,1,...]
    rank_devices_override = f"rank_devices=[{','.join(map(str, rank_devices))}]"

    click.echo(
        f"Launching {nproc} ranks ({workers_per_gpu} per GPU) across {num_gpus} GPU(s) "
        f"for {len(datasets)} dataset(s)."
    )
    click.echo(f"  datasets    : {datasets}")
    click.echo(f"  rank_devices: {rank_devices}")

    cmd = [
        torchrun_bin,
        f"--nproc_per_node={nproc}",
        "-m",
        "app.imc2025.run.distributed",
        config_file,
        *merge,
        "-o",
        rank_devices_override,
        "--backend",
        backend,
        "--environment",
        environment,
        *[arg for o in override for arg in ("-o", o)],
    ]

    click.echo(f"  cmd: {' '.join(cmd)}")
    os.execvp(torchrun_bin, cmd)


if __name__ == "__main__":
    main()
