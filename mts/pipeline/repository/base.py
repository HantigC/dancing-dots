from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, Iterable, Tuple

import numpy as np

from mts.core.types import ImageId, PairType


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
    def add_image(self, filepath: str) -> ImageId:
        """Adds an image to the repository.

        Args:
            filepath: Path to the image file.

        Returns:
            The unique identifier assigned to the image.
        """
        pass

    @abstractmethod
    def image_ids(self) -> Generator[ImageId, None, None]:
        """Iterates over all stored image IDs.

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
    def get_metadata(self, image_id: ImageId) -> dict[str, Any]:
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
    def add_keypoints(self, img_id: ImageId, keypoints: np.ndarray) -> None:
        """Stores keypoints for an image.

        Args:
            img_id: The image identifier.
            keypoints: Array of detected keypoints.
        """
        pass

    @abstractmethod
    def get_keypoints(self, img_id: ImageId) -> np.ndarray | None:
        """Retrieves keypoints for an image.

        Args:
            img_id: The image identifier.

        Returns:
            The keypoints array if available, otherwise None.
        """
        pass

    @abstractmethod
    def add_descriptors(self, img_id: ImageId, descriptors: np.ndarray) -> None:
        """Stores local feature descriptors for an image.

        Args:
            img_id: The image identifier.
            descriptors: Array of local descriptors.
        """
        pass

    @abstractmethod
    def get_descriptors(self, img_id: ImageId) -> np.ndarray | None:
        """Retrieves local feature descriptors for an image.

        Args:
            img_id: The image identifier.

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
    ) -> None:
        """Stores feature matches between two images.

        Args:
            img_id1: First image identifier.
            img_id2: Second image identifier.
            matches: Array of feature matches.
        """
        pass

    @abstractmethod
    def get_matches(
        self,
        img_id1: ImageId,
        img_id2: ImageId,
    ) -> np.ndarray | None:
        """Retrieves feature matches between two images.

        Args:
            img_id1: First image identifier.
            img_id2: Second image identifier.

        Returns:
            The matches array if available, otherwise None.
        """
        pass

    @abstractmethod
    def add_pairs(self, pairs: list[PairType]) -> None:
        """Stores image ID pairs.

        Args:
            pairs: A list of image ID pairs.
        """
        pass

    @abstractmethod
    def get_pairs(self) -> list[PairType]:
        """Retrieves all stored image ID pairs.

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
    def close(self) -> None:
        """Releases any resources held by the repository.

        Implementations using file handles, database connections,
        or external resources should close them here.
        """
        pass


class RepositoryException(BaseException):
    """Repository related exception."""


class NotFoundException(RepositoryException):
    """Raised in case an item is not found inside the repository"""


class AlreadyExistsException(RepositoryException):
    """Raised in case an item already exists inside the repository"""
