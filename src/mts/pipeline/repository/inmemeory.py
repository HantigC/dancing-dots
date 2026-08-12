from pathlib import Path
from typing import Any, Generator

import numpy as np

from mts.core.geometry.rigid3d import Rigid3D
from mts.core.types import ImageId, PairType, PathLike
from mts.utils.image import imread_rgb

from .base import AlreadyExistsException, BaseImageRepository, NotFoundException




class ImageRepository(BaseImageRepository):
    def __init__(self):
        self._repository_metadata = {}
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
        self._match_metadata: dict[tuple, dict[str, Any]] = {}
        self._pairs_map = {}
        self._pairs = []
        self._poses: dict[ImageId, Rigid3D] = {}
        self._metadata: dict[ImageId, dict[str, Any]] = {}
        self._store: dict[str, Any] = {}
        self._pair_store: dict[str, dict[tuple, Any]] = {}
        self._pair_direction: dict[str, dict[tuple, tuple]] = {}

    def add_repository_metadata(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._repository_metadata:
                raise AlreadyExistsException(
                    f"{k} already exists inside repository metadata"
                )
            self._repository_metadata[k] = v

    def update_repository_metadata(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._repository_metadata:
                raise NotFoundException(f"{k} does not exist inside repository metadata")
            self._repository_metadata[k] = v

    def upsert_repository_metadata(self, **kwargs):
        self._repository_metadata.update(kwargs)

    def get_repository_metadata(self, name: str):
        return self._repository_metadata.get(name)

    def add_image(self, filepath: str) -> int:
        """Add an image filepath and return its integer ID."""
        if filepath in self._images_filepath:
            return self._images_filepath[filepath]

        img_id = self._next_id
        self._images_filepath[filepath] = img_id
        self._id_to_filepath[img_id] = filepath
        self._next_id += 1
        return img_id

    def image_ids(self) -> Generator[int, None, None]:
        for id in self._id_to_filepath:
            yield id

    def images_num(self):
        return len(self._images_filepath)

    def get_image_id(self, filepath: PathLike) -> int | None:
        """Return the ID for a filepath, or None if not found."""
        image_id = self._images_filepath.get(filepath)
        if image_id is None:
            image_id = self._images_filepath.get(Path(filepath))
        return image_id

    def get_filepath(self, img_id: int) -> str | None:
        """Return the filepath for a given image ID."""
        return self._id_to_filepath.get(img_id)

    def image_filepaths(self) -> Generator[Path, None, None]:
        return (self._id_to_filepath[image_id] for image_id in self.image_ids())

    def iterate_over_images(self) -> Generator[tuple[ImageId, np.ndarray], None, None]:
        for id in self._id_to_filepath:
            yield id, self.load_image(id)

    def load_image(self, image_id: int) -> np.ndarray:
        return imread_rgb(self._id_to_filepath[image_id])

    def add_metadata(self, image_id: ImageId, **kwargs):
        image_metadata_dict = self._metadata.setdefault(image_id, {})
        for k, v in kwargs.items():
            if k in image_metadata_dict:
                raise AlreadyExistsException(f"metadata `{k}` already exists")
            image_metadata_dict[k] = v

    def update_metadata(self, image_id: ImageId, **kwargs):
        image_metadata_dict = self._metadata.setdefault(image_id, {})
        for k, v in kwargs.items():
            if k not in image_metadata_dict:
                raise NotFoundException(f"metadata `{k}` does not exist")
            image_metadata_dict[k] = v

    def upsert_metadata(self, image_id: ImageId, **kwargs):
        image_metadata_dict = self._metadata.setdefault(image_id, {})
        for k, v in kwargs.items():
            image_metadata_dict[k] = v

    def delete_metadata(self, name: str) -> None:
        for k, metadata_dict in self._metadata.items():
            if name in metadata_dict:
                del metadata_dict[name]

    def get_metadata(self, image_id: ImageId):
        image_metadata_dict = self._metadata.setdefault(image_id, {})
        return image_metadata_dict

    def get_metadata_values(self, image_id: ImageId, *args):
        image_metadata_dict = self._metadata.setdefault(image_id, {})
        return {image_metadata_dict[k] for k in args}

    def add_pose(self, image_id: ImageId, pose: Rigid3D):
        self._poses[image_id] = pose

    def get_pose(self, image_id: ImageId) -> Rigid3D | None:
        return self._poses.get(image_id)

    def add_keypoints(self, img_id: int, keypoints: np.ndarray, *, name: str = "keypoints"):
        self._keypoints.setdefault(img_id, {})[name] = keypoints

    def get_keypoints(self, img_id: int, *, name: str = "keypoints") -> np.ndarray | None:
        return self._keypoints.get(img_id, {}).get(name)

    def add_descriptors(self, img_id: int, descriptors: np.ndarray, *, name: str = "descriptors"):
        self._descriptors.setdefault(img_id, {})[name] = descriptors

    def get_descriptors(self, img_id: int, *, name: str = "descriptors") -> np.ndarray | None:
        return self._descriptors.get(img_id, {}).get(name)

    def add_global_descriptor(self, img_id: int, keypoints: np.ndarray):
        self._global_descriptors[img_id] = keypoints

    def get_global_descriptor(self, img_id: int) -> np.ndarray | None:
        return self._global_descriptors.get(img_id)

    def add_matches(self, img_id1: int, img_id2: int, matches: np.ndarray, *, name: str = "matches"):
        key = tuple(sorted((img_id1, img_id2)))
        self._matches.setdefault(key, {})[name] = matches

    def get_matches(self, img_id1: int, img_id2: int, *, name: str = "matches") -> np.ndarray | None:
        key = tuple(sorted((img_id1, img_id2)))
        return self._matches.get(key, {}).get(name)

    def add_match_metadata(self, img_id1: int, img_id2: int, **kwargs):
        key = tuple(sorted((img_id1, img_id2)))
        pair_meta = self._match_metadata.setdefault(key, {})
        for k, v in kwargs.items():
            if k in pair_meta:
                raise AlreadyExistsException(
                    f"match metadata `{k}` already exists for pair ({img_id1}, {img_id2})"
                )
            pair_meta[k] = v

    def update_match_metadata(self, img_id1: int, img_id2: int, **kwargs):
        key = tuple(sorted((img_id1, img_id2)))
        pair_meta = self._match_metadata.setdefault(key, {})
        for k, v in kwargs.items():
            if k not in pair_meta:
                raise NotFoundException(
                    f"match metadata `{k}` does not exist for pair ({img_id1}, {img_id2})"
                )
            pair_meta[k] = v

    def upsert_match_metadata(self, img_id1: int, img_id2: int, **kwargs):
        key = tuple(sorted((img_id1, img_id2)))
        pair_meta = self._match_metadata.setdefault(key, {})
        for k, v in kwargs.items():
            pair_meta[k] = v

    def get_match_metadata(self, img_id1: int, img_id2: int) -> dict | None:
        key = tuple(sorted((img_id1, img_id2)))
        return self._match_metadata.get(key, {})

    def add_pairs(self, pairs: list[PairType[ImageId]]) -> None:
        for st_id, nd_id in pairs:
            self._pairs_map.setdefault(st_id, []).append(nd_id)
            self._pairs_map.setdefault(nd_id, []).append(st_id)
        self._pairs.extend([tuple(sorted(pair)) for pair in pairs])

    def get_pairs(self) -> list[PairType[ImageId]]:
        return self._pairs

    def pair_num(self) -> int:
        return len(self._pairs)

    def store(self, name: str, data: Any) -> None:
        self._store[name] = data

    def load(self, name: str) -> Any:
        return self._store.get(name)

    @staticmethod
    def _pair_store_key(
        img_id1: ImageId, img_id2: ImageId, *, directional: bool
    ) -> tuple:
        if directional:
            return (img_id1, img_id2)
        return tuple(sorted((img_id1, img_id2)))

    def store_pair(
        self,
        img_id1: ImageId,
        img_id2: ImageId,
        name: str,
        data: Any,
        *,
        directional: bool = False,
    ) -> None:
        key = self._pair_store_key(img_id1, img_id2, directional=directional)
        self._pair_store.setdefault(name, {})[key] = data
        self._pair_direction.setdefault(name, {})[key] = (img_id1, img_id2)

    def load_pair(
        self,
        img_id1: ImageId,
        img_id2: ImageId,
        *,
        name: str = "data",
        directional: bool = False,
        with_direction: bool = True,
    ) -> Any:
        key = self._pair_store_key(img_id1, img_id2, directional=directional)
        data = self._pair_store.get(name, {}).get(key)
        if not with_direction:
            return data
        direction = self._pair_direction.get(name, {}).get(key)
        return data, direction

    def get_stored_pairs(self, name: str) -> list[PairType[ImageId]]:
        return list(self._pair_store.get(name, {}).keys())

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass
