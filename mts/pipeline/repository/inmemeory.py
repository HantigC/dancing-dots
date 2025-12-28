from typing import Any, Generator

import numpy as np

from mts.core.geometry.rigid3d import Rigid3D
from mts.core.types import ImageId, PairType
from mts.utils.image import imread_rgb

class NotFoundException(BaseException):
    """Raised in case an item is not found"""


class ImageRepository:
    def __init__(self):
        # filepath → integer ID
        self._images_filepath = {}
        # integer ID → filepath (reverse lookup)
        self._id_to_filepath = {}
        self._next_id = 0

        # image ID → keypoints/descriptors
        self._keypoints = {}
        self._global_descriptors = {}
        self._descriptors = {}

        # (id1, id2) → matches array
        self._matches = {}
        self._pairs_map = {}
        self._pairs = []
        self._poses: dict[ImageId, Rigid3D] = {}
        self._metadata: dict[ImageId, dict[str, Any]] = {}
        

    def pair_num(self) -> int:
        return len(self._pairs)

    def add_pairs(self, pairs: list[PairType[ImageId]]) -> None:
        for st_id, nd_id in pairs:
            self._pairs_map.setdefault(st_id, []).append(nd_id)
            self._pairs_map.setdefault(nd_id, []).append(st_id)
        self._pairs.extend(pairs)

    def get_pairs(self) -> list[PairType[ImageId]]:
        return self._pairs

    def iterate_over_images(self) -> Generator[np.ndarray, None, None]:
        for id in self._id_to_filepath:
            yield id, self.load_image(id)

    def image_ids(self) -> Generator[int, None, None]:
        for id in self._id_to_filepath:
            yield id

    def load_image(self, image_id: int) -> np.ndarray:
        return imread_rgb(self._id_to_filepath[image_id])

    def add_image(self, filepath: str) -> int:
        """Add an image filepath and return its integer ID."""
        if filepath in self._images_filepath:
            return self._images_filepath[filepath]

        img_id = self._next_id
        self._images_filepath[filepath] = img_id
        self._id_to_filepath[img_id] = filepath
        self._next_id += 1
        return img_id

    def get_image_id(self, filepath: str) -> int | None:
        """Return the ID for a filepath, or None if not found."""
        return self._images_filepath.get(filepath)

    def get_filepath(self, img_id: int) -> str | None:
        """Return the filepath for a given image ID."""
        return self._id_to_filepath.get(img_id)

    def images_num(self):
        return len(self._images_filepath)
    
    def add_metadata(self, image_id: ImageId, **kwargs):
        image_metadata_dict = self._metadata.setdefault(image_id, {})
        for k, v in kwargs.items():
            if k in image_metadata_dict:
                raise ValueError(f"metadata `{k}` already exists")
            image_metadata_dict[k] = v

    def update_metadata(self, image_id: ImageId, **kwargs):
        image_metadata_dict = self._metadata.setdefault(image_id, {})
        for k, v in kwargs.items():
            if k not in image_metadata_dict:
                raise ValueError(f"metadata `{k}` does not exist")
            image_metadata_dict[k] = v

    def upsert_metadata(self, image_id: ImageId, **kwargs):
        image_metadata_dict = self._metadata.setdefault(image_id, {})
        for k, v in kwargs.items():
            image_metadata_dict[k] = v

    def get_metadata(self, image_id: ImageId, *args):
        image_metadata_dict = self._metadata.setdefault(image_id, {})
        return {image_metadata_dict[k] for k in args}

    def add_pose(self, image_id: ImageId, pose: Rigid3D):
        self._poses[image_id] = pose

    def get_pose(self, image_id: ImageId) -> Rigid3D:
        return self._poses[image_id]

    def add_keypoints(self, img_id: int, keypoints: np.ndarray):
        self._keypoints[img_id] = keypoints

    def get_keypoints(self, img_id: int) -> np.ndarray | None:
        return self._keypoints.get(img_id)

    def add_global_descriptor(self, img_id: int, keypoints: np.ndarray):
        self._global_descriptors[img_id] = keypoints

    def get_global_descriptor(self, img_id: int) -> np.ndarray | None:
        return self._global_descriptors.get(img_id)

    def add_descriptors(self, img_id: int, descriptors: np.ndarray):
        self._descriptors[img_id] = descriptors

    def get_descriptors(self, img_id: int) -> np.ndarray | None:
        return self._descriptors.get(img_id)

    def add_matches(self, img_id1: int, img_id2: int, matches: np.ndarray):
        key = tuple(sorted((img_id1, img_id2)))
        self._matches[key] = matches

    def get_matches(self, img_id1: int, img_id2: int) -> np.ndarray | None:
        key = tuple(sorted((img_id1, img_id2)))
        return self._matches.get(key)
