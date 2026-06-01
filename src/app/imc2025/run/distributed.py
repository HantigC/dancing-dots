"""
Queue-based distributed pipeline runner.

All datasets are pushed onto a shared work queue.  Each worker process pulls
one dataset at a time; as soon as it finishes it picks up the next one.  This
avoids idle workers caused by uneven dataset sizes.

No torchrun needed — launch directly:

    uv run run-distributed \\
        config/pipeline/imc2025/0009/base.yaml \\
        [merge1.yaml ...] \\
        [--workers-per-gpu 2] \\
        [--num-gpus 4] \\
        [-o datasets_names=[ETs,stairs]]

Alternatively, use ``launch-distributed`` which auto-detects GPU count.

``rank_devices`` config field (list of ints) overrides the per-worker GPU
assignment, e.g. ``rank_devices: [0,0,1,1]`` for 4 workers sharing 2 GPUs.
"""

import json
import logging
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path

import click
import torch
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

_DONE = None  # sentinel that tells a worker to exit


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


def _worker(
    rank: int,
    device: str,
    cfg,
    iteration_dirpath: Path,
    all_samples: DatasetSamples,
    dataset_queue: "mp.Queue[str | None]",
    result_queue: "mp.Queue[DatasetSamples]",
) -> None:
    """Worker process: pull datasets from the queue until the sentinel is received."""
    from hydra.utils import get_method

    cfg = _replace_cuda_devices(cfg, device)

    create_repo = get_method(cfg.reconstruction_runner.create_repository_method)
    create_pipe = get_method(cfg.reconstruction_runner.create_pipeline_method)
    create_pipe_state = get_method(cfg.reconstruction_runner.create_pipeline_state_method)

    completed: DatasetSamples = {}
    try:
        while True:
            dataset_name = dataset_queue.get(timeout=60)
            if dataset_name is _DONE:
                LOGGER.info("[worker %d / %s] is DONe", rank, device)
                break

            LOGGER.info("[worker %d / %s] starting: %s", rank, device, dataset_name)
            try:
                pipeline = IMC2025Pipeline(
                    iteration_dirpath,
                    {dataset_name: all_samples[dataset_name]},
                    create_repo,
                    partial(create_pipe, cfg),
                    create_pipeline_state=create_pipe_state,
                )
                pipeline.run([dataset_name])
                completed.update(pipeline.samples)
                LOGGER.info("[worker %d / %s] done: %s", rank, device, dataset_name)
            except Exception:
                LOGGER.exception(
                    "[worker %d / %s] dataset %s failed — skipping",
                    rank,
                    device,
                    dataset_name,
                )
    finally:
        result_queue.put(completed)


def _build_rank_devices(cfg, num_gpus: int, workers_per_gpu: int) -> list[str]:
    """Return one device string per worker."""
    rank_devices_cfg = cfg.get("rank_devices")
    if rank_devices_cfg is not None:
        return [f"cuda:{g}" for g in rank_devices_cfg]
    return [f"cuda:{g}" for g in range(num_gpus) for _ in range(workers_per_gpu)]


def _run(cfg, rank_devices: list[str]) -> None:
    num_workers = len(rank_devices)

    last_project_iteration = GitProject.from_next_iteration(
        cfg.project_path,
        create=False,
        save=cfg.save_project_to_git,
    )
    last_project_iteration.__enter__()
    iteration_dirpath = last_project_iteration.iteration_dirpath

    setup_file_logging(iteration_dirpath)
    LOGGER.info(
        "Queue-distributed run — %d worker(s), devices %s:\n%s",
        num_workers,
        rank_devices,
        OmegaConf.to_yaml(cfg),
    )
    try:
        cfg.git_commit = get_git_commit()
    except NotAGitRepositoryError:
        cfg.git_commit = None
    OmegaConf.save(cfg, iteration_dirpath / "config.yaml")

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

    LOGGER.info("Queuing %d dataset(s): %s", len(all_datasets), all_datasets)

    ctx = mp.get_context("spawn")
    dataset_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    for ds in all_datasets:
        dataset_queue.put(ds)
    for _ in range(num_workers):
        dataset_queue.put(_DONE)

    processes = [
        ctx.Process(
            target=_worker,
            args=(rank, device, cfg, iteration_dirpath, all_samples, dataset_queue, result_queue),
            daemon=True,
        )
        for rank, device in enumerate(rank_devices)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    merged_samples: DatasetSamples = {}
    for _ in range(num_workers):
        merged_samples.update(result_queue.get_nowait())

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


@click.command()
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("merge", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--override",
    "-o",
    multiple=True,
    metavar="KEY=VALUE",
    help="Dot-notation overrides, e.g. -o datasets_names=[ETs].",
)
@click.option("--environment", "-e", default=DEBUG, show_default=True)
@click.option(
    "--workers-per-gpu",
    default=1,
    show_default=True,
    help="Worker processes per GPU (ignored when rank_devices is set in config).",
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
        num_gpus = torch.cuda.device_count() or 1

    rank_devices = _build_rank_devices(cfg, num_gpus, workers_per_gpu)
    click.echo(f"Workers: {len(rank_devices)}  devices: {rank_devices}")
    _run(cfg, rank_devices)


if __name__ == "__main__":
    main()
