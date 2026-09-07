import pytest

from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.step.pair.knn import KnnEmbeddingParerStep


@pytest.fixture
def repo():
    repository = ImageRepository()
    repository.add_images(["a0.jpg", "a1.jpg", "a2.jpg"], scene="scene_a")
    repository.add_images(["b0.jpg", "b1.jpg"], scene="scene_b")
    return repository


def test_knn_pairer_is_scene_scoped(repo):
    step = KnnEmbeddingParerStep(min_images=20)  # forces the combinations branch
    state = {}
    result = step.run(image_repository=repo, input=None, state=state)

    assert set(result) == {"scene_a", "scene_b"}

    pairs_a = {tuple(sorted(p)) for p in repo.get_pairs(scene="scene_a")}
    pairs_b = {tuple(sorted(p)) for p in repo.get_pairs(scene="scene_b")}
    ids_a = set(repo.image_ids(scene="scene_a"))
    ids_b = set(repo.image_ids(scene="scene_b"))

    assert pairs_a and pairs_b
    assert pairs_a.isdisjoint(pairs_b)
    assert all(a in ids_a and b in ids_a for a, b in pairs_a)
    assert all(a in ids_b and b in ids_b for a, b in pairs_b)

    assert state["scenes"]["scene_a"]["starting_pairs"]
    assert state["scenes"]["scene_b"]["starting_pairs"]


def test_knn_pairer_never_creates_cross_scene_pairs(repo):
    step = KnnEmbeddingParerStep(min_images=20)
    step.run(image_repository=repo, input=None, state={})
    # add_pairs would have raised on a cross-scene pair; getting here is the assertion
    assert {tuple(sorted(p)) for p in repo.get_pairs()} == {
        tuple(sorted(p)) for p in repo.get_pairs(scene="scene_a")
    } | {tuple(sorted(p)) for p in repo.get_pairs(scene="scene_b")}
