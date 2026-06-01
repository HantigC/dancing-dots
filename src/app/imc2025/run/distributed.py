"""
Distributed pipeline runner — designed to be launched via torchrun:

    torchrun --nproc_per_node=2 -m app.imc2025.run.distributed \\
        config/pipeline/imc2025/0005/base.yaml \\
        [merge1.yaml ...] \\
        [-o datasets_names=[ETs,stairs]]

Each rank handles a round-robin slice of the dataset list and runs on its
own CUDA device (local_rank 0 → cuda:0, local_rank 1 → cuda:1).  After all
ranks finish, rank 0 gathers predictions, renumbers cluster indices, and
writes the submission CSV + summary.
"""

import json
import logging
import os
from functools import partial
from pathlib import Path

import click
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from app.constants import DEBUG
from app.imc2025.pipeline import IMC2025Pipeline
from app.imc2025.prediction import (
    DatasetSamples,
    append_samples_to_csv,
    load_from_csv,
    load_from_submission,
    load_test_images,
)
from app.imc2025.run.types import RunType
from app.logging import setup_file_logging
from app.setup import setup_from_env
from mts.helpers.imc import metric
from mts.helpers.project.git_project import GitProject
from mts.utils.git import NotAGitRepositoryError, get_git_commit

LOGGER = logging.getLogger(__name__)


def _replace_cuda_devices(cfg, device: str):
    """Walk the config tree and replace every ``device: cuda:*`` with *device*."""
    raw = OmegaConf.to_container(cfg, resolve=False)

    def _walk(obj):
        if isinstance(obj, dict):
            for k in list(obj):
                v = obj[k]
                if k == "device" and isinstance(v, str) and v.startswith("cuda:"):
                    obj[k] = device
                else:
                    _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(raw)
    return OmegaConf.create(raw)


def _renumber_clusters(merged_samples: DatasetSamples, dataset_order: list[str]) -> None:
    """Re-index cluster indices so they are globally unique across all datasets."""
    offset = 0
    for name in dataset_order:
        if name not in merged_samples:
            continue
        preds = merged_samples[name]
        local_max = max(
            (p.cluster_index for p in preds if p.cluster_index is not None),
            default=-1,
        )
        if local_max < 0:
            continue
        for pred in preds:
            if pred.cluster_index is not None:
                pred.cluster_index += offset
        offset += local_max + 1


def _worker(cfg) -> None:
    from hydra.utils import get_method

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    cfg = _replace_cuda_devices(cfg, f"cuda:{local_rank}")

    # --- Rank 0 allocates the iteration directory ---
    last_project_iteration = None
    if rank == 0:
        last_project_iteration = GitProject.from_next_iteration(
            cfg.project_path,
            create=False,
            save=cfg.save_project_to_git,
        )
        last_project_iteration.__enter__()
        shared = [str(last_project_iteration.iteration_dirpath)]
    else:
        shared = [None]

    dist.broadcast_object_list(shared, src=0)
    iteration_dirpath = Path(shared[0])

    if rank == 0:
        setup_file_logging(iteration_dirpath)
        LOGGER.info(
            "Distributed run (world_size=%d):\n%s", world_size, OmegaConf.to_yaml(cfg)
        )
        try:
            cfg.git_commit = get_git_commit()
        except NotAGitRepositoryError:
            cfg.git_commit = None
        OmegaConf.save(cfg, iteration_dirpath / "config.yaml")

    # --- Load samples (identical on every rank; each takes its slice) ---
    data_dirpath = Path(cfg.get("data_dirpath", "data"))
    cfg.data_dirpath = data_dirpath
    run_type = RunType(cfg.get("run_type", RunType.TRAIN))
    samples_filename = cfg.get("sample_filepath")

    if samples_filename:
        all_samples, df = load_from_csv(data_dirpath, samples_filename)
    elif run_type == RunType.TRAIN:
        all_samples, df = load_from_csv(data_dirpath, "train_labels.csv")
    elif run_type == RunType.SUBMISSION:
        all_samples, df = load_from_submission(data_dirpath)
    else:
        all_samples, df = load_test_images(data_dirpath)

    datasets_cfg = cfg.get("datasets_names")
    all_datasets = list(datasets_cfg) if datasets_cfg is not None else list(all_samples.keys())

    # Round-robin assignment: rank 0 → [0, 2, 4, …], rank 1 → [1, 3, 5, …]
    my_datasets = all_datasets[rank::world_size]
    my_samples = {k: all_samples[k] for k in my_datasets if k in all_samples}

    LOGGER.info("[rank %d / %d] datasets: %s", rank, world_size, my_datasets)

    create_repo = get_method(cfg.reconstruction_runner.create_repository_method)
    create_pipe = get_method(cfg.reconstruction_runner.create_pipeline_method)
    create_pipe_state = get_method(cfg.reconstruction_runner.create_pipeline_state_method)

    imc2025_pipeline = IMC2025Pipeline(
        iteration_dirpath,
        my_samples,
        create_repo,
        partial(create_pipe, cfg),
        create_pipeline_state=create_pipe_state,
    )
    cfg.origin = imc2025_pipeline.project_dirpath

    dist.barrier()

    if my_datasets:
        imc2025_pipeline.run(my_datasets)

    # --- Gather predictions from all ranks to rank 0 ---
    dist.barrier()
    all_samples_per_rank = [None] * world_size
    dist.all_gather_object(all_samples_per_rank, imc2025_pipeline.samples)

    if rank == 0:
        merged_samples: DatasetSamples = {}
        for rank_samples in all_samples_per_rank:
            merged_samples.update(rank_samples)

        _renumber_clusters(merged_samples, all_datasets)

        submission_dest = Path(cfg.get("submission_dest_dirpath") or iteration_dirpath)
        submission_filepath = submission_dest / "submission.csv"
        append_samples_to_csv(merged_samples, df, submission_filepath)

        if run_type == RunType.TRAIN:
            summary_dict = metric.score(
                gt_csv=data_dirpath / "train_labels.csv",
                user_csv=submission_filepath,
                thresholds_csv=data_dirpath / "train_thresholds.csv",
                mask_csv=None,
                inl_cf=0,
                strict_cf=-1,
                verbose=True,
            )
            with open(iteration_dirpath / "summary.json", "w") as f:
                json.dump(summary_dict, f, indent=4)

        last_project_iteration.__exit__(None, None, None)

    dist.barrier()
    dist.destroy_process_group()


@click.command()
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("merge", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--override",
    "-o",
    multiple=True,
    metavar="KEY=VALUE",
    help="Dot-notation overrides applied after merges, e.g. -o datasets_names=[ETs].",
)
@click.option("--environment", "-e", default=DEBUG, show_default=True)
@click.option(
    "--backend",
    default="nccl",
    show_default=True,
    help="torch.distributed backend (nccl for NVIDIA, gloo for CPU fallback).",
)
def main(
    config_file: str,
    merge: tuple,
    override: tuple,
    environment: str,
    backend: str,
) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    setup_from_env(environment)

    cfg = OmegaConf.load(config_file)
    for path in merge:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(override)))

    dist.init_process_group(backend=backend)
    _worker(cfg)


if __name__ == "__main__":
    main()
