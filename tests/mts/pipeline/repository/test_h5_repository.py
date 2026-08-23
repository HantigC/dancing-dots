from pathlib import Path

import numpy as np
import pytest

from mts.core.geometry.rigid3d import Rigid3D
from mts.pipeline.repository.base import DEFAULT_SCENE, RepositoryException
from mts.pipeline.repository.h5 import H5ImageRepository


@pytest.fixture
def repo(tmp_path):
    repository = H5ImageRepository(tmp_path / "repo.h5")
    with repository:
        yield repository


# --- images -------------------------------------------------------------


def test_add_image_assigns_incrementing_ids(repo):
    id_a = repo.add_image("a.jpg")
    id_b = repo.add_image("b.jpg")
    assert id_a == 0
    assert id_b == 1


def test_get_image_id_and_get_filepath_roundtrip(repo):
    img_id = repo.add_image("a.jpg")
    assert repo.get_image_id("a.jpg") == img_id
    assert repo.get_filepath(img_id) == "a.jpg"


def test_get_image_id_returns_none_for_unknown_filepath(repo):
    assert repo.get_image_id("missing.jpg") is None


def test_image_ids_returns_all_ids_by_default(repo):
    id_a = repo.add_image("a.jpg")
    id_b = repo.add_image("b.jpg")
    assert set(repo.image_ids()) == {id_a, id_b}


def test_image_filepaths(repo):
    repo.add_image("a.jpg")
    repo.add_image("b.jpg")
    assert set(repo.image_filepaths()) == {Path("a.jpg"), Path("b.jpg")}


# --- metadata -------------------------------------------------------------
# Note: unlike the in-memory repository, H5ImageRepository.add_metadata does
# not raise on a duplicate key (it warns and overwrites), and
# update_metadata raises a plain ValueError rather than NotFoundException.
# These tests capture the actual current H5 behavior, not the in-memory one.


def test_add_metadata_then_get_metadata(repo):
    img_id = repo.add_image("a.jpg")
    repo.add_metadata(img_id, width=10, height=20)
    assert repo.get_metadata(img_id) == {"width": 10, "height": 20}


def test_add_metadata_overwrites_on_duplicate_key(repo):
    img_id = repo.add_image("a.jpg")
    repo.add_metadata(img_id, width=10)
    repo.add_metadata(img_id, width=20)
    assert repo.get_metadata(img_id)["width"] == 20


def test_update_metadata_raises_on_missing_key(repo):
    img_id = repo.add_image("a.jpg")
    with pytest.raises(ValueError):
        repo.update_metadata(img_id, width=20)


def test_upsert_metadata_inserts_and_updates(repo):
    img_id = repo.add_image("a.jpg")
    repo.upsert_metadata(img_id, width=10)
    repo.upsert_metadata(img_id, width=20)
    assert repo.get_metadata(img_id)["width"] == 20


def test_get_metadata_values(repo):
    img_id = repo.add_image("a.jpg")
    repo.add_metadata(img_id, width=10, height=20)
    assert repo.get_metadata_values(img_id, "width", "height") == {10, 20}


# --- poses -------------------------------------------------------------


def test_add_pose_then_get_pose(repo):
    img_id = repo.add_image("a.jpg")
    pose = Rigid3D(rotation=np.eye(3), translation=np.zeros(3))
    repo.add_pose(img_id, pose)
    stored = repo.get_pose(img_id)
    assert np.array_equal(stored.rotation, pose.rotation)
    assert np.array_equal(stored.translation, pose.translation)


def test_get_pose_returns_none_when_unset(repo):
    img_id = repo.add_image("a.jpg")
    assert repo.get_pose(img_id) is None


# --- pairs (current, scene-less behavior) --------------------------------


def test_add_pairs_and_get_pairs(repo):
    id_a = repo.add_image("a.jpg")
    id_b = repo.add_image("b.jpg")
    id_c = repo.add_image("c.jpg")
    repo.add_pairs([(id_a, id_b), (id_b, id_c)])
    assert {tuple(p) for p in repo.get_pairs()} == {(id_a, id_b), (id_b, id_c)}
    assert repo.pair_num() == 2


# --- generic per-pair store ------------------------------------------------


def test_store_pair_and_load_pair_non_directional(repo):
    id_a = repo.add_image("a.jpg")
    id_b = repo.add_image("b.jpg")
    repo.store_pair(id_a, id_b, "homography", {"h": 1})
    data, direction = repo.load_pair(id_b, id_a, name="homography")
    assert data == {"h": 1}
    assert direction == (id_a, id_b)


def test_store_pair_directional_keeps_order_distinct(repo):
    id_a = repo.add_image("a.jpg")
    id_b = repo.add_image("b.jpg")
    repo.store_pair(id_a, id_b, "flow", "a->b", directional=True)
    repo.store_pair(id_b, id_a, "flow", "b->a", directional=True)
    assert (
        repo.load_pair(id_a, id_b, name="flow", directional=True, with_direction=False)
        == "a->b"
    )
    assert (
        repo.load_pair(id_b, id_a, name="flow", directional=True, with_direction=False)
        == "b->a"
    )


def test_get_stored_pairs(repo):
    id_a = repo.add_image("a.jpg")
    id_b = repo.add_image("b.jpg")
    repo.store_pair(id_a, id_b, "homography", {"h": 1})
    assert repo.get_stored_pairs("homography") == [(id_a, id_b)]


# --- scenes (new behavior) -------------------------------------------------


def test_add_image_defaults_to_base_scene(repo):
    img_id = repo.add_image("a.jpg")
    assert repo.get_scene(img_id) == DEFAULT_SCENE == "base"


def test_add_image_with_explicit_scene(repo):
    img_id = repo.add_image("a.jpg", scene="scene_a")
    assert repo.get_scene(img_id) == "scene_a"


def test_add_scene_overwrites_previous_value(repo):
    img_id = repo.add_image("a.jpg", scene="scene_a")
    repo.add_scene(img_id, "scene_b")
    assert repo.get_scene(img_id) == "scene_b"


def test_image_ids_reflects_scene_reassignment(repo):
    """add_scene must move the image between the per-scene indexes used by
    image_ids(scene=...), not just update the get_scene cache."""
    img_id = repo.add_image("a.jpg", scene="scene_a")
    other_id = repo.add_image("b.jpg", scene="scene_a")
    assert set(repo.image_ids(scene="scene_a")) == {img_id, other_id}

    repo.add_scene(img_id, "scene_b")

    assert set(repo.image_ids(scene="scene_a")) == {other_id}
    assert set(repo.image_ids(scene="scene_b")) == {img_id}


def test_image_ids_filters_by_scene(repo):
    id_a = repo.add_image("a.jpg", scene="scene_a")
    id_b = repo.add_image("b.jpg", scene="scene_a")
    id_c = repo.add_image("c.jpg", scene="scene_b")
    id_default = repo.add_image("d.jpg")

    assert set(repo.image_ids(scene="scene_a")) == {id_a, id_b}
    assert set(repo.image_ids(scene="scene_b")) == {id_c}
    assert set(repo.image_ids(scene=DEFAULT_SCENE)) == {id_default}
    assert set(repo.image_ids()) == {id_a, id_b, id_c, id_default}


def test_image_ids_scene_with_no_matches_is_empty(repo):
    repo.add_image("a.jpg")
    assert list(repo.image_ids(scene="unknown_scene")) == []


def test_get_pairs_filters_by_scene(repo):
    id_a = repo.add_image("a.jpg", scene="scene_a")
    id_b = repo.add_image("b.jpg", scene="scene_a")
    id_c = repo.add_image("c.jpg", scene="scene_b")
    id_d = repo.add_image("d.jpg", scene="scene_b")

    repo.add_pairs([(id_a, id_b), (id_c, id_d)])

    assert {tuple(p) for p in repo.get_pairs(scene="scene_a")} == {(id_a, id_b)}
    assert {tuple(p) for p in repo.get_pairs(scene="scene_b")} == {(id_c, id_d)}
    assert {tuple(p) for p in repo.get_pairs()} == {(id_a, id_b), (id_c, id_d)}


def test_get_pairs_unknown_scene_is_empty(repo):
    id_a = repo.add_image("a.jpg", scene="scene_a")
    id_b = repo.add_image("b.jpg", scene="scene_a")
    repo.add_pairs([(id_a, id_b)])
    assert repo.get_pairs(scene="scene_b") == []


def test_add_pairs_raises_when_pair_spans_two_scenes(repo):
    id_a = repo.add_image("a.jpg", scene="scene_a")
    id_b = repo.add_image("b.jpg", scene="scene_b")
    with pytest.raises(RepositoryException):
        repo.add_pairs([(id_a, id_b)])
