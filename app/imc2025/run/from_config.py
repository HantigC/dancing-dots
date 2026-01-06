import json
import logging
import os
from functools import partial
from pathlib import Path

from hydra.utils import get_method
from omegaconf import OmegaConf

from app.constants import DEBUG
from app.imc2025.pipeline import ALL, IMC2025Pipeline
from app.imc2025.prediction import load_from_csv, sample_to_csv
from mts.core.types import PathLike
from mts.helpers.imc import metric
from mts.helpers.project.project import Project

LOGGER = logging.getLogger(__name__)


def create_imc2025_from_cfg(cfg):
    last_project_iteration = Project.from_next_iteration(cfg.project_path)
    create_pipeline = get_method(cfg.reconstruction_runner.create_pipeline_method)
    create_repository = get_method(cfg.reconstruction_runner.create_repository_method)
    create_pipeline_state = get_method(
        cfg.reconstruction_runner.create_pipeline_state_method
    )
    data_dirpath = Path(cfg.get("data_dirpath", "data"))
    cfg.data_dirpath = data_dirpath
    samples_filename = cfg.get("sample_filepath", "train_labels.csv")
    samples = load_from_csv(data_dirpath, samples_filename)

    imc2025_pipeline = IMC2025Pipeline(
        last_project_iteration.iteration_dirpath,
        samples,
        create_repository,
        partial(create_pipeline, cfg),
        create_pipeline_state=create_pipeline_state,
    )
    cfg.origin = imc2025_pipeline.project_dirpath
    return imc2025_pipeline


def run_from_cfg(cfg) -> IMC2025Pipeline:
    imc2025_pipeline = create_imc2025_from_cfg(cfg)
    LOGGER.info(
        "Saving the config `%s` to path: `%s`",
        str(cfg.origin),
        str(imc2025_pipeline.project_dirpath),
    )
    OmegaConf.save(cfg, imc2025_pipeline.project_dirpath / "config.yaml")
    datasets_names = cfg.get("datasets_names", ALL)
    imc2025_pipeline.run(list(datasets_names))
    is_train = cfg.get("is_train", True)

    submission_filepath = imc2025_pipeline.project_dirpath / "submission.csv"
    sample_to_csv(imc2025_pipeline.samples, submission_filepath)
    data_dirpath = Path(cfg.data_dirpath)

    if is_train:
        summary_dict = metric.score(
            gt_csv=data_dirpath / "train_labels.csv",
            user_csv=submission_filepath,
            thresholds_csv=data_dirpath / "train_thresholds.csv",
            mask_csv=None if is_train else data_dirpath / "mask.csv",
            inl_cf=0,
            strict_cf=-1,
            verbose=True,
        )
        with open(
            imc2025_pipeline.project_dirpath / "summary.json", "w"
        ) as summary_file:
            json.dump(
                summary_dict,
                summary_file,
                indent=4,
            )

    return imc2025_pipeline


def run_from_config_filepath(hydra_config_filepath: PathLike) -> IMC2025Pipeline:
    cfg = OmegaConf.load(hydra_config_filepath)
    run_from_cfg(cfg)


def main(environment: str = DEBUG):
    from app import setup_from_env

    setup_from_env((environment))
    PIPELINE_CONFIG = os.environ.get("PIPELINE_CONFIG")
    run_from_config_filepath(PIPELINE_CONFIG)
