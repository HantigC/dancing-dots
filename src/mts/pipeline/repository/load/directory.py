import logging
from pathlib import Path
from typing import Iterable

from mts.core.types import ImageId, PathLike
from mts.pipeline.repository.base import BaseImageRepository

LOGGER = logging.getLogger(__name__)

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def iter_image_filepaths(
    dirpath: PathLike,
    *,
    extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    recursive: bool = False,
) -> list[Path]:
    """Lists image files inside a directory, sorted by filepath.

    Args:
        dirpath: Directory to scan.
        extensions: File extensions to include, case-insensitive, with or
            without the leading dot (e.g. "jpg" or ".jpg"). Defaults to a
            set of common image formats.
        recursive: If True, also scans subdirectories.

    Returns:
        Sorted list of matching filepaths.

    Raises:
        NotADirectoryError: If `dirpath` doesn't exist or isn't a directory.
    """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise NotADirectoryError(f"`{dirpath}` is not a directory")

    normalized_extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
    }
    glob = dirpath.rglob if recursive else dirpath.glob
    filepaths = (
        filepath
        for filepath in glob("*")
        if filepath.is_file() and filepath.suffix.lower() in normalized_extensions
    )
    return sorted(filepaths)


def load_from_directory(
    image_repository: BaseImageRepository,
    dirpath: PathLike,
    *,
    scene: str | None = None,
    extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    recursive: bool = False,
) -> list[ImageId]:
    """Loads every image found in a directory into an image repository.

    Args:
        image_repository: Repository to add the images to.
        dirpath: Directory to scan for images.
        scene: Scene the images are assigned to (see `BaseImageRepository.
            add_image`). If None (default), images are assigned to the
            repository's default scene.
        extensions: File extensions to include, case-insensitive, with or
            without the leading dot. Defaults to a set of common image
            formats.
        recursive: If True, also scans subdirectories.

    Returns:
        The IDs assigned to the added images, in the order they were added
        (sorted by filepath).

    Raises:
        NotADirectoryError: If `dirpath` doesn't exist or isn't a directory.
    """
    filepaths = iter_image_filepaths(dirpath, extensions=extensions, recursive=recursive)
    LOGGER.info("Found %d images in `%s`", len(filepaths), dirpath)

    return [
        image_repository.add_image(str(filepath), scene=scene) for filepath in filepaths
    ]


def load_scenes_from_directory(
    image_repository: BaseImageRepository,
    dirpath: PathLike,
    *,
    extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    recursive: bool = False,
) -> None:
    """Loads a directory of per-scene subdirectories into one image repository.

    Expects `dirpath` to contain one subdirectory per scene, e.g.::

        <dirpath>/scene1/*.jpg
        <dirpath>/scene2/*.jpg

    Each immediate subdirectory's name is used as the scene name (passed to
    `add_image`'s `scene` argument, see `BaseImageRepository.add_image`), and
    its images are loaded via `load_from_directory` into `image_repository` —
    a single repository ends up holding every scene, distinguished by each
    image's `scene`. Files directly inside `dirpath` (not in a subdirectory)
    are ignored.

    Args:
        image_repository: Repository to add the images to.
        dirpath: Directory containing one subdirectory per scene.
        extensions: File extensions to include, case-insensitive, with or
            without the leading dot. Defaults to a set of common image
            formats.
        recursive: If True, also scans subdirectories of each scene
            directory.

    Raises:
        NotADirectoryError: If `dirpath` doesn't exist or isn't a directory.
    """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise NotADirectoryError(f"`{dirpath}` is not a directory")

    scene_dirpaths = sorted(p for p in dirpath.iterdir() if p.is_dir())
    for scene_dirpath in scene_dirpaths:
        filepaths = iter_image_filepaths(scene_dirpath, extensions=extensions, recursive=recursive,)
        for filepath in filepaths:
            image_repository.add_image(str(filepath), scene=scene_dirpath.name,)
