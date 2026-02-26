from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Generator

import h5py
import numpy as np

from mts.core.geometry.rigid3d import Rigid3D
from mts.core.types import ImageId, PathLike
from mts.utils.image import imread_rgb

from .base import BaseImageRepository

LOGGER = logging.getLogger(__name__)


class H5ImageRepository(BaseImageRepository):
    def __init__(self, h5_path: str):
        self._h5_path = h5_path
        self._h5 = h5py.File(h5_path, "a")

        self._image_id_to_filepath = {}
        self._filepath_to_image_id = {}

        self._repo_meta = self._h5.require_group("repository_metadata")
        self._images_grp = self._h5.require_group("images")
        self._features_grp = self._h5.require_group("features")
        self._matches_grp = self._h5.require_group("matches")
        self._poses_grp = self._h5.require_group("poses")
        self._metadata_grp = self._h5.require_group("metadata")
        self._pairs_grp = self._h5.require_group("pairs")

        if "next_id" not in self._h5.attrs:
            self._h5.attrs["next_id"] = 0

        self._next_id = self._h5.attrs["next_id"]
        for image_id in self._images_grp.keys():
            filepath = self._images_grp[image_id].attrs["filepath"]
            self._image_id_to_filepath[int(image_id)] = filepath
            self._image_id_to_filepath[filepath] = int(image_id)

    def add_repository_metadata(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._repo_meta.attrs:
                raise ValueError(f"{k} already exists")
            self._repo_meta.attrs[k] = json.dumps(v)

    def add_image(self, filepath: PathLike) -> int:
        filepath = str(filepath)
        image_id = self._filepath_to_image_id.get(filepath)
        if image_id is not None:
            LOGGER.warning("`%s` already exists", filepath)

        img_id = self._h5.attrs["next_id"]
        grp = self._images_grp.create_group(str(img_id))

        self._image_id_to_filepath[int(img_id)] = filepath
        self._filepath_to_image_id[filepath] = img_id
        grp.attrs["filepath"] = filepath

        self._h5.attrs["next_id"] += 1
        return img_id

    def image_ids(self) -> Generator[int, None, None]:
        for k in self._images_grp.keys():
            yield int(k)

    def images_num(self):
        return len(self._images_grp)

    def get_image_id(self, filepath: PathLike) -> ImageId:
        filepath = str(filepath)
        image_id = self._filepath_to_image_id.get(filepath)
        return image_id

    def get_filepath(self, img_id: int | np.int64) -> str | None:
        grp = self._images_grp.get(str(img_id))
        if grp is None:
            return None
        return grp.attrs["filepath"]

    def image_filepaths(self) -> Generator[Path, None, None]:
        return (
            Path(self._image_id_to_filepath[image_id]) for image_id in self.image_ids()
        )

    def iterate_over_images(self) -> Generator[tuple[ImageId, np.ndarray], None, None]:
        for image_id in self._images_grp:
            yield image_id, self.load_image(image_id)

    def load_image(self, image_id: ImageId) -> np.ndarray:
        return imread_rgb(self._images_grp[str(image_id)].attrs["filepath"])

    def add_metadata(self, image_id: int, **kwargs):
        grp = self._metadata_grp.require_group(str(image_id))
        for k, v in kwargs.items():
            if k in grp.attrs:
                raise ValueError(f"metadata `{k}` already exists")
            grp.attrs[k] = json.dumps(v)

    def update_metadata(self, image_id: int, **kwargs):
        grp = self._metadata_grp.require_group(str(image_id))
        for k, v in kwargs.items():
            if k not in grp.attrs:
                raise ValueError(f"metadata `{k}` does not exist")
            grp.attrs[k] = json.dumps(v)

    def upsert_metadata(self, image_id: int, **kwargs):
        grp = self._metadata_grp.require_group(str(image_id))
        for k, v in kwargs.items():
            grp.attrs[k] = json.dumps(v)

    def delete_metadata(self, name: str) -> None:
        for metadata in self._metadata_grp.values():
            if name in metadata.attrs:
                del metadata.attrs[name]

    def get_metadata(self, image_id: ImageId) -> dict[str, Any]:
        grp = self._metadata_grp.require_group(str(image_id))
        return {k: json.loads(v) for k, v in grp.attrs.items()}

    def get_metadata_values(self, image_id: ImageId, *args):
        metadata_grp = self._metadata_grp[str(image_id)]
        return {json.loads(metadata_grp[k]) for k in args}

    def add_pose(self, image_id: int, pose: Rigid3D, force: bool = True):
        rigid_grp = self._poses_grp.get(str(image_id))
        if rigid_grp is None:
            rigid_grp = self._poses_grp.create_group(str(image_id))
        elif not force:
            return

        rigid_grp.attrs["rotation"] = pose.rotation
        rigid_grp.attrs["translation"] = pose.translation

    def get_pose(self, image_id: int):
        if str(image_id) not in self._poses_grp:
            return None
        pose_grp = self._poses_grp[str(image_id)]
        rigid_3d = Rigid3D(pose_grp["rotation"], pose_grp["translation"])
        return rigid_3d

    def add_keypoints(self, img_id: int, keypoints: np.ndarray):
        grp = self._features_grp.require_group(str(img_id))
        if "keypoints" in grp:
            del grp["keypoints"]
        grp.create_dataset("keypoints", data=keypoints)

    def get_keypoints(self, img_id: int):
        grp = self._features_grp.get(str(img_id))
        if grp is None or "keypoints" not in grp:
            return None
        return grp["keypoints"][:]

    def add_descriptors(self, img_id: int, descriptors: np.ndarray):
        grp = self._features_grp.require_group(str(img_id))
        if "descriptors" in grp:
            del grp["descriptors"]
        grp.create_dataset("descriptors", data=descriptors)

    def get_descriptors(self, img_id: int):
        grp = self._features_grp.get(str(img_id))
        if grp is None or "descriptors" not in grp:
            return None
        return grp["descriptors"][:]

    def add_global_descriptor(self, img_id: int, descriptor: np.ndarray):
        grp = self._features_grp.require_group(str(img_id))
        if "global_descriptor" in grp:
            del grp["global_descriptor"]
        grp.create_dataset("global_descriptor", data=descriptor)

    def get_global_descriptor(self, img_id: int):
        grp = self._features_grp.get(str(img_id))
        if grp is None or "global_descriptor" not in grp:
            return None
        return grp["global_descriptor"][:]

    def add_matches(self, img_id1: int, img_id2: int, matches: np.ndarray):
        key = f"{min(img_id1, img_id2)}_{max(img_id1, img_id2)}"
        if key in self._matches_grp:
            del self._matches_grp[key]
        self._matches_grp.create_dataset(key, data=matches)

    def get_matches(self, img_id1: int, img_id2: int):
        key = f"{min(img_id1, img_id2)}_{max(img_id1, img_id2)}"
        if key not in self._matches_grp:
            return None
        return self._matches_grp[key][:]

    def add_pairs(self, pairs):
        ds = self._h5.require_dataset(
            "pairs",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype=np.int64,
        )
        n = len(ds)
        ds.resize((n + len(pairs), 2))
        ds[n:] = np.array([sorted(p) for p in pairs])

    def get_pairs(self):
        if "pairs" not in self._h5:
            return []
        return [tuple(row) for row in self._pairs_grp["pairs"][:]]

    def pair_num(self):
        return len(self._h5["pairs"][:])

    def close(self):
        self._h5.close()

    @classmethod
    def from_filename(cls, dirpath: PathLike, filename: str) -> H5ImageRepository:
        dirpath = Path(dirpath)
        return cls(str(dirpath / filename))
