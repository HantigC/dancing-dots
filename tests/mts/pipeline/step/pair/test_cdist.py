import numpy as np
import pytest

from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.pair.cdist import CrossEmbeddingParerStep


@pytest.fixture
def repo():
    repository = ImageRepository()
    for scene, names in {"s1": ["a0", "a1", "a2"], "s2": ["b0", "b1"]}.items():
        for name in names:
            img_id = repository.add_image(f"{name}.jpg", scene=scene)
            repository.add_global_descriptor(
                img_id, np.random.rand(8).astype(np.float32)
            )
    return repository


def test_cdist_pairs_are_within_a_single_scene(repo):
    step = CrossEmbeddingParerStep(min_images=20)  # combinations branch
    step.run(image_repository=repo, input=None, state={})

    ids_s1 = set(repo.image_ids(scene="s1"))
    ids_s2 = set(repo.image_ids(scene="s2"))
    pairs_s1 = {tuple(sorted(p)) for p in repo.get_pairs(scene="s1")}
    pairs_s2 = {tuple(sorted(p)) for p in repo.get_pairs(scene="s2")}

    assert pairs_s1 and pairs_s2
    assert pairs_s1.isdisjoint(pairs_s2)
    assert all(a in ids_s1 and b in ids_s1 for a, b in pairs_s1)
    assert all(a in ids_s2 and b in ids_s2 for a, b in pairs_s2)
    # no pair spans scenes
    assert {tuple(sorted(p)) for p in repo.get_pairs()} == pairs_s1 | pairs_s2
