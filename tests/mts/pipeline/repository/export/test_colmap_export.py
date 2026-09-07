import numpy as np
import pytest
from PIL import Image

from mts.helpers.colmap.database import COLMAPDatabase
from mts.pipeline.repository.base import SceneScopedImageRepository
from mts.pipeline.repository.export.colmap import export_to_colmap
from mts.pipeline.repository.inmemeory import ImageRepository


def _write_image(path):
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(path)


@pytest.fixture
def repo(tmp_path):
    repository = ImageRepository()
    layout = {"scene_a": ["a0.jpg", "a1.jpg"], "scene_b": ["b0.jpg", "b1.jpg", "b2.jpg"]}
    for scene, names in layout.items():
        for name in names:
            fp = tmp_path / name
            _write_image(fp)
            img_id = repository.add_image(str(fp), scene=scene)
            repository.add_keypoints(
                img_id, np.random.rand(8, 2).astype(np.float32), name="keypoints"
            )
    ids_a = list(repository.image_ids(scene="scene_a"))
    ids_b = list(repository.image_ids(scene="scene_b"))
    repository.add_pairs([(ids_a[0], ids_a[1])])
    repository.add_pairs([(ids_b[0], ids_b[1]), (ids_b[1], ids_b[2])])
    for i, j in [(ids_a[0], ids_a[1]), (ids_b[0], ids_b[1]), (ids_b[1], ids_b[2])]:
        repository.add_matches(i, j, np.zeros((4, 2), dtype=np.int64), name="matches")
    return repository


def test_export_to_colmap_is_scene_scoped_and_repeatable(repo, tmp_path):
    scoped_a = SceneScopedImageRepository(repo, "scene_a")
    scoped_b = SceneScopedImageRepository(repo, "scene_b")

    export_to_colmap(scoped_a, tmp_path / "scene_a.db")
    # second scene must not raise (upsert, not add, on id_to_db_id metadata)
    export_to_colmap(scoped_b, tmp_path / "scene_b.db")

    db_a = COLMAPDatabase.connect(tmp_path / "scene_a.db")
    db_b = COLMAPDatabase.connect(tmp_path / "scene_b.db")
    n_a = db_a.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    n_b = db_b.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    assert n_a == 2
    assert n_b == 3
