from pathlib import Path

import pytest

from mts.pipeline.repository.inmemeory import ImageRepository
from mts.pipeline.repository.load.directory import (
    iter_image_filepaths,
    load_from_directory,
    load_scenes_from_directory,
)


@pytest.fixture
def repo():
    return ImageRepository()


def _touch(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


# --- iter_image_filepaths --------------------------------------------------


def test_iter_image_filepaths_finds_known_extensions(tmp_path):
    _touch(
        tmp_path / "a.jpg",
        tmp_path / "b.PNG",
        tmp_path / "c.txt",
        tmp_path / "d.jpeg",
    )
    filepaths = iter_image_filepaths(tmp_path)
    assert [p.name for p in filepaths] == ["a.jpg", "b.PNG", "d.jpeg"]


def test_iter_image_filepaths_is_sorted(tmp_path):
    _touch(tmp_path / "c.jpg", tmp_path / "a.jpg", tmp_path / "b.jpg")
    filepaths = iter_image_filepaths(tmp_path)
    assert [p.name for p in filepaths] == ["a.jpg", "b.jpg", "c.jpg"]


def test_iter_image_filepaths_ignores_subdirectories_by_default(tmp_path):
    _touch(tmp_path / "a.jpg", tmp_path / "nested" / "b.jpg")
    filepaths = iter_image_filepaths(tmp_path)
    assert [p.name for p in filepaths] == ["a.jpg"]


def test_iter_image_filepaths_recursive(tmp_path):
    _touch(tmp_path / "a.jpg", tmp_path / "nested" / "b.jpg")
    filepaths = iter_image_filepaths(tmp_path, recursive=True)
    assert {p.name for p in filepaths} == {"a.jpg", "b.jpg"}


def test_iter_image_filepaths_custom_extensions(tmp_path):
    _touch(tmp_path / "a.jpg", tmp_path / "b.exr")
    filepaths = iter_image_filepaths(tmp_path, extensions=("exr",))
    assert [p.name for p in filepaths] == ["b.exr"]


def test_iter_image_filepaths_raises_on_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        iter_image_filepaths(tmp_path / "missing")


# --- load_from_directory ----------------------------------------------------


def test_load_from_directory_adds_all_images(repo, tmp_path):
    _touch(tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.txt")
    image_ids = load_from_directory(repo, tmp_path)
    assert repo.images_num() == 2
    assert len(image_ids) == 2


def test_load_from_directory_preserves_sorted_order(repo, tmp_path):
    _touch(tmp_path / "b.jpg", tmp_path / "a.jpg")
    image_ids = load_from_directory(repo, tmp_path)
    filepaths = [repo.get_filepath(image_id) for image_id in image_ids]
    assert filepaths == [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]


def test_load_from_directory_assigns_default_scene(repo, tmp_path):
    _touch(tmp_path / "a.jpg")
    (image_id,) = load_from_directory(repo, tmp_path)
    assert repo.get_scene(image_id) == "base"


def test_load_from_directory_assigns_given_scene(repo, tmp_path):
    _touch(tmp_path / "a.jpg")
    (image_id,) = load_from_directory(repo, tmp_path, scene="scene-1")
    assert repo.get_scene(image_id) == "scene-1"


def test_load_from_directory_recursive(repo, tmp_path):
    _touch(tmp_path / "a.jpg", tmp_path / "nested" / "b.jpg")
    image_ids = load_from_directory(repo, tmp_path, recursive=True)
    assert len(image_ids) == 2


def test_load_from_directory_raises_on_missing_directory(repo, tmp_path):
    with pytest.raises(NotADirectoryError):
        load_from_directory(repo, tmp_path / "missing")


# --- load_scenes_from_directory ---------------------------------------------


def test_load_scenes_from_directory_assigns_scene_per_subdirectory(repo, tmp_path):
    _touch(
        tmp_path / "scene1" / "a.jpg",
        tmp_path / "scene1" / "b.jpg",
        tmp_path / "scene2" / "c.jpg",
    )
    result = load_scenes_from_directory(repo, tmp_path)

    assert result is None
    assert repo.images_num() == 3
    scenes = [repo.get_scene(image_id) for image_id in repo.image_ids()]
    assert sorted(scenes) == ["scene1", "scene1", "scene2"]
    assert len(list(repo.image_ids(scene="scene1"))) == 2


def test_load_scenes_from_directory_ignores_loose_files(repo, tmp_path):
    _touch(tmp_path / "loose.jpg", tmp_path / "scene1" / "a.jpg")
    load_scenes_from_directory(repo, tmp_path)
    assert repo.images_num() == 1
    (image_id,) = list(repo.image_ids())
    assert repo.get_scene(image_id) == "scene1"


def test_load_scenes_from_directory_recursive(repo, tmp_path):
    _touch(
        tmp_path / "scene1" / "a.jpg",
        tmp_path / "scene1" / "nested" / "b.jpg",
    )
    load_scenes_from_directory(repo, tmp_path, recursive=True)
    assert repo.images_num() == 2


def test_load_scenes_from_directory_raises_on_missing_directory(repo, tmp_path):
    with pytest.raises(NotADirectoryError):
        load_scenes_from_directory(repo, tmp_path / "missing")
