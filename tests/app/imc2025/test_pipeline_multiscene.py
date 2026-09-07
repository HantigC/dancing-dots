from pathlib import Path

import numpy as np

from app.imc2025.pipeline import (
    IMC2025Pipeline,
    create_inmemory_repository,
    create_pipeline_state,
)
from app.imc2025.prediction import Prediction
from mts.core.geometry.rigid3d import Rigid3D
from mts.pipeline.step.base import PerSceneStep


class _FakeReconStep(PerSceneStep):
    """Registers every image of a scene into cluster 0 with an identity pose."""

    def run_scene(self, *, image_repository, scene, input, state, scene_state):
        for image_id in image_repository.image_ids(scene=scene):
            image_repository.add_pose(
                image_id, Rigid3D(np.eye(3), np.zeros(3))
            )
            image_repository.add_metadata(image_id, cluster=0, match_kind=None)


def _samples(tmp_path, dataset, names):
    out = []
    for name in names:
        fp = tmp_path / dataset / name
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(b"x")
        out.append(
            Prediction(image_id=None, dataset=dataset, filename=name, image_filepath=fp)
        )
    return out


def test_run_adds_every_dataset_as_a_scene_and_collects_per_scene(tmp_path):
    samples = {
        "ds_a": _samples(tmp_path, "ds_a", ["a0.jpg", "a1.jpg"]),
        "ds_b": _samples(tmp_path, "ds_b", ["b0.jpg"]),
    }
    captured = {}

    def create_pipeline(image_repository):
        captured["repo"] = image_repository
        return [_FakeReconStep()]

    pipeline = IMC2025Pipeline(
        project_dirpath=tmp_path / "iter",
        samples=samples,
        create_repository=create_inmemory_repository,
        create_pipeline=create_pipeline,
        create_pipeline_state=create_pipeline_state,
    )
    pipeline.run("all")

    repo = captured["repo"]
    assert repo.scenes() == ["ds_a", "ds_b"]

    # one repository, both scenes populated
    assert set(repo.image_ids(scene="ds_a")) and set(repo.image_ids(scene="ds_b"))

    # every prediction got a pose
    for preds in samples.values():
        for pred in preds:
            assert pred.rotation is not None
            assert pred.translation is not None

    # per-scene cluster 0 gets globally offset so scenes don't collide
    a_clusters = {p.cluster_index for p in samples["ds_a"]}
    b_clusters = {p.cluster_index for p in samples["ds_b"]}
    assert a_clusters == {0}
    assert b_clusters == {1}
