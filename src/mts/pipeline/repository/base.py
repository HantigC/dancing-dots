from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, Iterable, Tuple

import numpy as np
from PIL import Image

from mts.core.types import ImageId, PairType

DEFAULT_SCENE = "base"


class BaseImageRepository(ABC):
    """Abstract base class for image repositories.

    This interface defines the contract for storing and retrieving images,
    features, matches, poses, and metadata.

    Implementations may store data in memory, HDF5 files, databases,
    cloud storage, or other backends.
    """

    @abstractmethod
    def add_repository_metadata(self, **kwargs) -> None:
        """Adds repository-level metadata.

        Args:
            **kwargs: Arbitrary key-value pairs representing global
                repository metadata.

        Raises:
            Exception: If a metadata key already exists (implementation-specific).
        """
        pass

    @abstractmethod
    def update_repository_metadata(self, **kwargs) -> None:
        """Updates existing repository-level metadata keys.

        Raises:
            Exception: If a metadata key does not exist (implementation-specific).
        """
        pass

    @abstractmethod
    def upsert_repository_metadata(self, **kwargs) -> None:
        """Inserts or updates repository-level metadata keys."""
        pass

    @abstractmethod
    def get_repository_metadata(self, name: str) -> Any:
        """Retrieves a single repository-level metadata value by key.

        Returns:
            The stored value, or None if not found.
        """
        pass

    @abstractmethod
    def add_image(self, filepath: str, scene: str | None = None) -> ImageId:
        """Adds an image to the repository.

        Args:
            filepath: Path to the image file.
            scene: Scene this image belongs to. Scenes aren't a separate
                registry that must be created upfront — assigning an image
                to a scene name that hasn't been used before implicitly
                creates it. If None (default), the image is assigned to the
                default scene, `DEFAULT_SCENE` ("base").

        Returns:
            The unique identifier assigned to the image.
        """
        pass

    def add_images(
        self, filepaths: list[str], scene: str | None = None
    ) -> list[ImageId]:
        """Adds multiple images to the repository.

        Args:
            filepaths: Paths to the image files.
            scene: Scene these images belong to. See `add_image` for details
                on scene assignment. If None (default), the images are
                assigned to the default scene, `DEFAULT_SCENE` ("base").

        Returns:
            The unique identifiers assigned to the images, in the same
            order as `filepaths`.
        """
        return [self.add_image(filepath, scene=scene) for filepath in filepaths]

    @abstractmethod
    def add_scene(self, image_id: ImageId, scene: str) -> None:
        """Assigns (or reassigns) the scene an image belongs to.

        Args:
            image_id: The image identifier.
            scene: The scene name.
        """
        pass

    @abstractmethod
    def get_scene(self, image_id: ImageId) -> str | None:
        """Retrieves the scene assigned to an image.

        Args:
            image_id: The image identifier.

        Returns:
            The scene name, or None if the image has no scene assigned.
        """
        pass

    @abstractmethod
    def image_ids(self, scene: str | None = None) -> Generator[ImageId, None, None]:
        """Iterates over stored image IDs.

        Args:
            scene: If given, yields only image IDs assigned to this scene.
                If None (default), yields all image IDs regardless of scene.

        Yields:
            The identifier of each stored image.
        """
        pass

    @abstractmethod
    def images_num(self) -> int:
        """Returns the total number of stored images.

        Returns:
            The number of images in the repository.
        """
        pass

    @abstractmethod
    def get_image_id(self, filepath: str | Path) -> ImageId | None:
        """Retrieves the image ID corresponding to a filepath.

        Args:
            filepath: Path of the image.

        Returns:
            The image ID if found, otherwise None.
        """
        pass

    @abstractmethod
    def get_filepath(self, img_id: ImageId) -> str | None:
        """Retrieves the filepath for a given image ID.

        Args:
            img_id: The image identifier.

        Returns:
            The filepath if found, otherwise None.
        """
        pass

    @abstractmethod
    def image_filepaths(self) -> Generator[Path, None, None]:
        """Iterates over all stored image filepaths.

        Yields:
            The filepath of each stored image.
        """
        pass

    @abstractmethod
    def iterate_over_images(
        self,
    ) -> Generator[Tuple[ImageId, np.ndarray], None, None]:
        """Iterates over all images and their pixel data.

        Yields:
            A tuple containing the image ID and the loaded image array.
        """
        pass

    @abstractmethod
    def load_image(self, image_id: ImageId) -> np.ndarray:
        """Loads image pixel data by ID.

        Args:
            image_id: The image identifier.

        Returns:
            The image as a NumPy array.
        """
        pass

    def get_size_wh(self, image_id: ImageId) -> tuple[int, int]:
        """Returns (width, height) for an image, reading from disk only once."""
        cached = self.get_metadata(image_id)
        if cached and "width" in cached and "height" in cached:
            return cached["width"], cached["height"]
        filepath = self.get_filepath(image_id)
        with Image.open(filepath) as img:
            w, h = img.size
        self.upsert_metadata(image_id, width=w, height=h)
        return w, h

    def get_size_hw(self, image_id: ImageId) -> tuple[int, int]:
        """Returns (height, width) for an image, reading from disk only once."""
        w, h = self.get_size_wh(image_id)
        return h, w

    @abstractmethod
    def add_metadata(self, image_id: ImageId, **kwargs) -> None:
        """Adds metadata to a specific image.

        Args:
            image_id: The image identifier.
            **kwargs: Metadata key-value pairs to add.

        Raises:
            Exception: If a metadata key already exists (implementation-specific).
        """
        pass

    @abstractmethod
    def update_metadata(self, image_id: ImageId, **kwargs) -> None:
        """Updates existing metadata for a specific image.

        Args:
            image_id: The image identifier.
            **kwargs: Metadata key-value pairs to update.

        Raises:
            Exception: If a metadata key does not exist (implementation-specific).
        """
        pass

    @abstractmethod
    def upsert_metadata(self, image_id: ImageId, **kwargs) -> None:
        """Inserts or updates metadata for a specific image.

        Args:
            image_id: The image identifier.
            **kwargs: Metadata key-value pairs.
        """
        pass

    @abstractmethod
    def delete_metadata(self, name: str) -> None:
        """Removes a metadata field from all images.

        Args:
            name: The metadata key to remove.
        """
        pass

    @abstractmethod
    def delete_repo(self) -> None:
        """Delete the whole repo"""
        pass

    @abstractmethod
    def get_metadata(self, image_id: ImageId) -> dict[str, Any] | None:
        """Retrieves all metadata for a specific image.

        Args:
            image_id: The image identifier.

        Returns:
            A dictionary containing metadata key-value pairs.
        """
        pass

    @abstractmethod
    def get_metadata_values(self, image_id: ImageId, *args) -> Iterable[Any]:
        """Retrieves selected metadata values for an image.

        Args:
            image_id: The image identifier.
            *args: Metadata keys to retrieve.

        Returns:
            An iterable of metadata values corresponding to the requested keys.
        """
        pass

    @abstractmethod
    def add_pose(self, image_id: ImageId, pose: Any) -> None:
        """Stores a pose associated with an image.

        Args:
            image_id: The image identifier.
            pose: The pose representation (e.g., transformation object or matrix).
        """
        pass

    @abstractmethod
    def get_pose(self, image_id: ImageId) -> Any | None:
        """Retrieves the pose associated with an image.

        Args:
            image_id: The image identifier.

        Returns:
            The pose if present, otherwise None.
        """
        pass

    @abstractmethod
    def add_keypoints(self, img_id: ImageId, keypoints: np.ndarray, *, name: str = "keypoints") -> None:
        """Stores a named set of keypoints for an image.

        Args:
            img_id: The image identifier.
            keypoints: Array of detected keypoints.
            name: Name identifying the keypoint type (e.g. "sift", "superpoint").
        """
        pass

    @abstractmethod
    def get_keypoints(self, img_id: ImageId, *, name: str = "keypoints") -> np.ndarray | None:
        """Retrieves a named set of keypoints for an image.

        Args:
            img_id: The image identifier.
            name: Name identifying the keypoint type.

        Returns:
            The keypoints array if available, otherwise None.
        """
        pass

    @abstractmethod
    def add_descriptors(self, img_id: ImageId, descriptors: np.ndarray, *, name: str = "descriptors") -> None:
        """Stores a named set of local feature descriptors for an image.

        Args:
            img_id: The image identifier.
            descriptors: Array of local descriptors.
            name: Name identifying the descriptor type (e.g. "sift", "superpoint").
        """
        pass

    @abstractmethod
    def get_descriptors(self, img_id: ImageId, *, name: str = "descriptors") -> np.ndarray | None:
        """Retrieves a named set of local feature descriptors for an image.

        Args:
            img_id: The image identifier.
            name: Name identifying the descriptor type.

        Returns:
            The descriptor array if available, otherwise None.
        """
        pass

    @abstractmethod
    def add_global_descriptor(self, img_id: ImageId, descriptor: np.ndarray) -> None:
        """Stores a global descriptor for an image.

        Args:
            img_id: The image identifier.
            descriptor: Global descriptor vector.
        """
        pass

    @abstractmethod
    def get_global_descriptor(self, img_id: ImageId) -> np.ndarray | None:
        """Retrieves the global descriptor for an image.

        Args:
            img_id: The image identifier.

        Returns:
            The global descriptor if available, otherwise None.
        """
        pass

    @abstractmethod
    def add_matches(
        self,
        img_id1: ImageId,
        img_id2: ImageId,
        matches: np.ndarray,
        *,
        name: str = "matches",
    ) -> None:
        """Stores feature matches between two images.

        Args:
            img_id1: First image identifier.
            img_id2: Second image identifier.
            matches: Array of feature matches.
            name: Name identifying the match type (e.g. "sift", "mast3r").
        """
        pass

    @abstractmethod
    def get_matches(
        self,
        img_id1: ImageId,
        img_id2: ImageId,
        *,
        name: str = "matches",
    ) -> np.ndarray | None:
        """Retrieves feature matches between two images.

        Args:
            img_id1: First image identifier.
            img_id2: Second image identifier.
            name: Name identifying the match type.

        Returns:
            The matches array if available, otherwise None.
        """
        pass

    @abstractmethod
    def add_match_metadata(
        self, img_id1: ImageId, img_id2: ImageId, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def update_match_metadata(
        self, img_id1: ImageId, img_id2: ImageId, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def upsert_match_metadata(
        self, img_id1: ImageId, img_id2: ImageId, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def get_match_metadata(
        self, img_id1: ImageId, img_id2: ImageId
    ) -> dict[str, Any] | None:
        pass

    @abstractmethod
    def add_pairs(self, pairs: list[PairType[int]]) -> None:
        """Stores image ID pairs.

        Each pair's scene is resolved from its endpoints' assigned scene
        (see `add_scene`/`add_image`). A pair whose two endpoints belong to
        different scenes raises `RepositoryException` — pairing is only
        ever expected to happen within one scene.

        Args:
            pairs: A list of image ID pairs.
        """
        pass

    @abstractmethod
    def get_pairs(self, scene: str | None = None) -> list[PairType]:
        """Retrieves stored image ID pairs.

        Args:
            scene: If given, returns only pairs stored under this scene.
                If None (default), returns all pairs regardless of scene.

        Returns:
            A list of image ID pairs.
        """
        pass

    @abstractmethod
    def pair_num(self) -> int:
        """Returns the number of stored image pairs.

        Returns:
            The total number of image pairs.
        """
        pass

    @abstractmethod
    def store(self, name: str, data: Any) -> None:
        """Stores arbitrary data under a name.

        Args:
            name: Key to store the data under.
            data: Data to store. Numpy arrays are stored natively;
                  all other types must be JSON-serializable.
        """
        pass

    @abstractmethod
    def load(self, name: str) -> Any:
        """Retrieves data previously stored under a name.

        Args:
            name: The key to retrieve.

        Returns:
            The stored data, or None if not found.
        """
        pass

    @abstractmethod
    def store_pair(
        self,
        img_id1: ImageId,
        img_id2: ImageId,
        name: str,
        data: Any,
        *,
        directional: bool = False,
    ) -> None:
        """Stores arbitrary data associated with a pair of images.

        Args:
            img_id1: First image identifier.
            img_id2: Second image identifier.
            name: Namespace to store the data under, scoped per-pair (mirrors
                `store`'s `name`, but per-pair rather than repository-global).
            data: Data to store. Numpy arrays are stored natively, dicts are
                stored recursively (nested dict values become nested groups,
                ndarray leaves are stored natively), and all other leaves
                must be JSON-serializable.
            directional: If False (default), the pair is treated as
                order-independent — `(img_id1, img_id2)` and `(img_id2,
                img_id1)` resolve to the same entry, matching `add_matches`'
                sorted-pair convention. If True, the pair is treated as
                order-significant (e.g. asymmetric per-pair data where the
                two images play different roles) — `(img_id1, img_id2)` and
                `(img_id2, img_id1)` are distinct entries.
        """
        pass

    @abstractmethod
    def load_pair(
        self,
        img_id1: ImageId,
        img_id2: ImageId,
        *,
        name: str = "data",
        directional: bool = False,
        with_direction: bool = True,
    ) -> Any:
        """Retrieves data previously stored via `store_pair`.

        Args:
            img_id1: First image identifier.
            img_id2: Second image identifier.
            name: The namespace used in `store_pair`.
            directional: Must match the value passed to `store_pair` for the
                lookup to hit the same entry. See `store_pair` for details.
            with_direction: If True (default), returns `(data, direction)`,
                where `direction` is the exact `(img_id1, img_id2)` order
                that was passed to `store_pair` for this entry (or None if
                no entry is found). Since non-directional storage resolves
                `(a, b)` and `(b, a)` to the same entry, `direction` lets a
                caller detect whether the order they queried with matches
                the order the data was originally stored in — useful when
                the stored data itself is order-sensitive (e.g. the two
                images play different roles). If False, returns the bare
                stored data, matching the pre-existing return shape.

        Returns:
            `(data, direction)` by default, or bare `data` when
            `with_direction=False`. `data` is None if not found.
        """
        pass

    @abstractmethod
    def get_stored_pairs(self, name: str) -> list[PairType[ImageId]]:
        """Lists image ID pairs that have data stored via `store_pair`.

        Args:
            name: The namespace used in `store_pair`.

        Returns:
            A list of image ID pairs that have an entry under `name`.
        """
        pass

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        """Releases any resources held by the repository.

        Implementations using file handles, database connections,
        or external resources should close them here.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class RepositoryException(BaseException):
    """Repository related exception."""


class NotFoundException(RepositoryException):
    """Raised in case an item is not found inside the repository"""


class AlreadyExistsException(RepositoryException):
    """Raised in case an item already exists inside the repository"""
