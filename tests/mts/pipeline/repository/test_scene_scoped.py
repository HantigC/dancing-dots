import numpy as np
import pytest

from mts.pipeline.repository.base import SceneScopedImageRepository, scene_key
from mts.pipeline.repository.inmemeory import ImageRepository


@pytest.fixture
def base_repo():
    repo = ImageRepository()
    repo.add_images(["a0.jpg", "a1.jpg"], scene="s1")
    repo.add_images(["b0.jpg", "b1.jpg", "b2.jpg"], scene="s2")
    a = list(repo.image_ids(scene="s1"))
    b = list(repo.image_ids(scene="s2"))
    repo.add_pairs([(a[0], a[1])])
    repo.add_pairs([(b[0], b[1]), (b[1], b[2])])
    return repo


def test_image_enumeration_is_scoped(base_repo):
    scoped = SceneScopedImageRepository(base_repo, "s1")
    assert sorted(scoped.image_ids()) == [0, 1]
    assert scoped.images_num() == 2
    assert scoped.scenes() == ["s1"]
    assert [p.name for p in scoped.image_filepaths()] == ["a0.jpg", "a1.jpg"]
    assert {tuple(sorted(p)) for p in scoped.get_pairs()} == {(0, 1)}
    assert scoped.pair_num() == 1


def test_ignores_explicit_scene_argument(base_repo):
    scoped = SceneScopedImageRepository(base_repo, "s2")
    # asking for another scene still yields this scope
    assert sorted(scoped.image_ids(scene="s1")) == [2, 3, 4]


def test_store_load_is_scene_namespaced(base_repo):
    s1 = SceneScopedImageRepository(base_repo, "s1")
    s2 = SceneScopedImageRepository(base_repo, "s2")
    s1.store("starting_pairs", [(0, 1)])
    s2.store("starting_pairs", [(2, 3), (3, 4)])

    assert s1.load("starting_pairs") == [(0, 1)]
    assert s2.load("starting_pairs") == [(2, 3), (3, 4)]
    # namespaced on the underlying repo
    assert base_repo.load(scene_key("starting_pairs", "s1")) == [(0, 1)]
    assert base_repo.load("starting_pairs") is None


def test_add_image_forced_into_scope(base_repo):
    scoped = SceneScopedImageRepository(base_repo, "s1")
    new_id = scoped.add_image("a2.jpg", scene="s2")  # scene arg ignored
    assert base_repo.get_scene(new_id) == "s1"


def test_delegates_unknown_methods(base_repo):
    scoped = SceneScopedImageRepository(base_repo, "s1")
    scoped.add_keypoints(0, np.zeros((3, 2), dtype=np.float32), name="kp")
    assert base_repo.get_keypoints(0, name="kp").shape == (3, 2)
    assert scoped.get_filepath(0) == base_repo.get_filepath(0)
    assert scoped.unscoped is base_repo


def test_get_stored_pairs_filtered_to_scene(base_repo):
    base_repo.store_pair(0, 1, "dec", {"x": np.zeros(2)})
    base_repo.store_pair(2, 3, "dec", {"x": np.zeros(2)})
    s1 = SceneScopedImageRepository(base_repo, "s1")
    assert [tuple(sorted(p)) for p in s1.get_stored_pairs("dec")] == [(0, 1)]


def test_iterate_over_matches_is_scoped(base_repo):
    base_repo.add_matches(0, 1, np.zeros((2, 2), dtype=np.int64))
    base_repo.add_matches(2, 3, np.zeros((3, 2), dtype=np.int64))

    s1 = SceneScopedImageRepository(base_repo, "s1")
    s2 = SceneScopedImageRepository(base_repo, "s2")

    assert [pair for pair, _ in s1.iterate_over_matches()] == [(0, 1)]
    assert [pair for pair, _ in s2.iterate_over_matches()] == [(2, 3)]
    # explicit scene arg is ignored in favour of the wrapper's scope
    assert [pair for pair, _ in s1.iterate_over_matches(scene="s2")] == [(0, 1)]
