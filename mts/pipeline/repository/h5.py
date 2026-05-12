from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import h5py
import numpy as np

from mts.core.geometry.rigid3d import Rigid3D
from mts.core.types import ImageId, PairType, PathLike
from mts.utils.image import imread_rgb

from .base import BaseImageRepository

LOGGER = logging.getLogger(__name__)


class H5ImageRepository(BaseImageRepository):
    def __init__(self, h5_path: PathLike, create_open: bool = False):
        self._h5_path = Path(h5_path)
        self._h5 = None
        self._image_id_to_filepath = {}
        self._filepath_to_image_id = {}
        self._init_maps()
        if create_open:
            self.open()

    def _init_maps(self):
        if not self._h5_path.exists():
            return
        with self._reading() as h5_read:
            for image_id in h5_read.get("images", {}).keys():
                filepath = h5_read["images"][image_id].attrs["filepath"]
                self._image_id_to_filepath[int(image_id)] = filepath
                self._filepath_to_image_id[filepath] = int(image_id)

    @staticmethod
    def _pair_key(img_id1: int, img_id2: int) -> str:
        return f"{min(img_id1, img_id2)}_{max(img_id1, img_id2)}"

    def open(self):
        self._open()

    def _open(self):
        self._h5 = h5py.File(self._h5_path, "a")

        if "next_id" not in self._h5.attrs:
            self._h5.attrs["next_id"] = 0

        self._next_id = self._h5.attrs["next_id"]
        # self._h5.require_group("repository_metadata")
        # self._h5.require_group("images")
        # self._h5.require_group("features")
        # self._h5.require_group("matches")
        # self._h5.require_group("matches_metadata")
        # self._h5.require_group("poses")
        # self._h5.require_group("metadata")
        # self._h5.require_group("store")

    @property
    def _repo_meta(self):
        return self._h5.require_group("repository_metadata")

    @property
    def _images_grp(self):
        return self._h5.require_group("images")

    @property
    def _features_grp(self):
        return self._h5.require_group("features")

    @property
    def _matches_grp(self):
        return self._h5.require_group("matches")

    @property
    def _poses_grp(self):
        return self._h5.require_group("poses")

    @property
    def _metadata_grp(self):
        return self._h5.require_group("metadata")

    @property
    def _matches_metadata_grp(self):
        return self._h5.require_group("matches_metadata")

    @property
    def _store_grp(self):
        return self._h5.require_group("store")

    def _get_file_for_reading(self):
        already_open = self._h5 is not None and self._h5.id.valid
        h5 = self._h5 if already_open else h5py.File(self._h5_path, "r")
        return h5

    @contextmanager
    def _reading(self):
        already_open = self._h5 is not None and self._h5.id.valid
        h5 = self._h5 if already_open else h5py.File(self._h5_path, "r")
        try:
            yield h5
        finally:
            if not already_open:
                h5.close()

    def __enter__(self):
        self._was_open = self._h5 is not None and self._h5.id.valid
        if not self._was_open:
            self._open()
        return self

    def __exit__(self, *_):
        if self._was_open:
            self._h5.flush()
        else:
            self._h5.close()

    def get_repository_metadata(self, name: str):
        with self._reading() as h5_read:
            value = h5_read["repository_metadata"].attrs.get(name)
            if value is not None:
                value = json.loads(value)
        return value

    def add_repository_metadata(self, **kwargs):
        with self:
            for k, v in kwargs.items():
                if k in self._repo_meta.attrs:
                    raise ValueError(f"{k} already exists")
                self._repo_meta.attrs[k] = json.dumps(v)

    def add_image(self, filepath: PathLike) -> int:
        with self:
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
        with self._reading() as h5_read:
            yield from (int(k) for k in h5_read["images"].keys())

    def images_num(self):
        with self._reading() as h5_read:
            return len(h5_read["images"])

    def get_image_id(self, filepath: PathLike) -> ImageId:
        filepath = str(filepath)
        return self._filepath_to_image_id.get(filepath)

    def get_filepath(self, img_id: int | np.int64) -> str | None:
        return self._image_id_to_filepath[int(img_id)]

    def image_filepaths(self) -> Generator[Path, None, None]:
        return (
            Path(self._image_id_to_filepath[image_id]) for image_id in self.image_ids()
        )

    def iterate_over_images(self) -> Generator[tuple[ImageId, np.ndarray], None, None]:
        with self._reading() as h5_read:
            for image_id in h5_read["images"]:
                yield int(image_id), self.load_image(image_id)

    def load_image(self, image_id: ImageId) -> np.ndarray:
        with self._reading() as h5_read:
            return imread_rgb(h5_read["images"][str(image_id)].attrs["filepath"])

    def add_metadata(self, image_id: int, **kwargs):
        with self:
            grp = self._metadata_grp.require_group(str(image_id))
            for k, v in kwargs.items():
                if k in grp.attrs:
                    LOGGER.warning(
                        "metadata `%s` already exists for image_id `%d`", k, image_id
                    )
                    LOGGER.debug("updating `%s` with %s", k, str(v))
                grp.attrs[k] = json.dumps(v)

    def update_metadata(self, image_id: int, **kwargs):
        with self:
            grp = self._metadata_grp.require_group(str(image_id))
            for k, v in kwargs.items():
                if k not in grp.attrs:
                    raise ValueError(f"metadata `{k}` does not exist")
                grp.attrs[k] = json.dumps(v)

    def upsert_metadata(self, image_id: int, **kwargs):
        with self:
            grp = self._metadata_grp.require_group(str(image_id))
            for k, v in kwargs.items():
                grp.attrs[k] = json.dumps(v)

    def delete_metadata(self, name: str) -> None:
        with self:
            for metadata in self._metadata_grp.values():
                if name in metadata.attrs:
                    del metadata.attrs[name]

    def get_metadata(self, image_id: ImageId) -> dict[str, Any] | None:
        with self._reading() as h5_read:
            grp = h5_read.get("metadata", {}).get(str(image_id))
            if grp is None:
                return {}
            attribute_dict = {k: json.loads(v) for k, v in grp.attrs.items()}
        return attribute_dict

    def get_metadata_values(self, image_id: ImageId, *args):
        with self._reading() as h5_read:
            metadata_grp = h5_read["metadata"][str(image_id)]
            return {json.loads(metadata_grp[k]) for k in args}

    def add_pose(self, image_id: int, pose: Rigid3D) -> None:
        with self:
            rigid_grp = self._poses_grp.get(str(image_id))
            if rigid_grp is None:
                rigid_grp = self._poses_grp.create_group(str(image_id))

            rigid_grp.attrs["rotation"] = pose.rotation
            rigid_grp.attrs["translation"] = pose.translation

    def get_pose(self, image_id: int):
        with self._reading() as h5_read:
            if str(image_id) not in h5_read["poses"]:
                return None
            pose_grp = h5_read["poses"][str(image_id)]
            return Rigid3D(pose_grp.attrs["rotation"], pose_grp.attrs["translation"])

    def add_keypoints(self, img_id: int, keypoints: np.ndarray):
        with self:
            grp = self._features_grp.require_group(str(img_id))
            if "keypoints" in grp:
                del grp["keypoints"]
            grp.create_dataset("keypoints", data=keypoints)

    def get_keypoints(self, img_id: int):
        with self._reading() as h5_read:
            grp = h5_read["features"].get(str(img_id))
            if grp is None or "keypoints" not in grp:
                return None
            return grp["keypoints"][:]

    def add_descriptors(self, img_id: int, descriptors: np.ndarray):
        with self:
            grp = self._features_grp.require_group(str(img_id))
            if "descriptors" in grp:
                del grp["descriptors"]

    def get_descriptors(self, img_id: int):
        with self._reading() as h5_read:
            grp = h5_read["features"].get(str(img_id))
            if grp is None or "descriptors" not in grp:
                return None
            return grp["descriptors"][:]

    def add_global_descriptor(self, img_id: int, descriptor: np.ndarray):
        with self:
            grp = self._features_grp.require_group(str(img_id))
            if "global_descriptor" in grp:
                del grp["global_descriptor"]
            grp.create_dataset("global_descriptor", data=descriptor)

    def get_global_descriptor(self, img_id: int):
        with self._reading() as h5_read:
            grp = h5_read["features"].get(str(img_id))
            if grp is None or "global_descriptor" not in grp:
                return None
            return grp["global_descriptor"][:]

    def add_match_metadata(self, img_id1: int, img_id2: int, **kwargs):
        with self:
            key = self._pair_key(img_id1, img_id2)
            grp = self._matches_metadata_grp.require_group(key)
            for k, v in kwargs.items():
                if k in grp.attrs:
                    raise ValueError(
                        f"match metadata `{k}` already exists for pair ({img_id1}, {img_id2})"
                    )
                grp.attrs[k] = json.dumps(v)

    def update_match_metadata(self, img_id1: int, img_id2: int, **kwargs):
        with self:
            key = self._pair_key(img_id1, img_id2)
            grp = self._matches_metadata_grp.require_group(key)
            for k, v in kwargs.items():
                if k not in grp.attrs:
                    raise ValueError(
                        f"match metadata `{k}` does not exist for pair ({img_id1}, {img_id2})"
                    )
                grp.attrs[k] = json.dumps(v)

    def upsert_match_metadata(self, img_id1: int, img_id2: int, **kwargs):
        with self:
            key = self._pair_key(img_id1, img_id2)
            grp = self._matches_metadata_grp.require_group(key)
            for k, v in kwargs.items():
                grp.attrs[k] = json.dumps(v)

    def get_match_metadata(self, img_id1: int, img_id2: int) -> dict | None:
        with self._reading() as h5_read:
            key = self._pair_key(img_id1, img_id2)
            grp = h5_read.get("matches_metadata", {}).get(key)
            if grp is None:
                return {}
            return {k: json.loads(v) for k, v in grp.attrs.items()}

    def add_matches(self, img_id1: int, img_id2: int, matches: np.ndarray):
        with self:
            key = self._pair_key(img_id1, img_id2)
            if key in self._matches_grp:
                del self._matches_grp[key]

            if img_id1 > img_id2:
                matches = matches[:, ::-1]
            self._matches_grp.create_dataset(key, data=matches)

    def get_matches(self, img_id1: int, img_id2: int):
        with self._reading() as h5_read:
            key = self._pair_key(img_id1, img_id2)
            if key not in h5_read["matches"]:
                return None
            matches = h5_read["matches"][key][:]
            if img_id1 > img_id2:
                matches = matches[:, ::-1]
            return matches

    def iterate_over_matches(
        self,
    ) -> Generator[
        tuple[PairType[int], np.ndarray],
        None,
        None,
    ]:
        with self._reading() as h_reading:
            matches = h_reading["matches"]
            for pair_id, match in matches.items():
                st_image_id, nd_image_id = pair_id.split("_")
                st_image_id, nd_image_id = int(st_image_id), int(nd_image_id)
                yield (st_image_id, nd_image_id), match[:]

    def store(self, name: str, data: Any):
        with self:
            grp = self._store_grp
            if isinstance(data, np.ndarray):
                if name in grp:
                    del grp[name]
                grp.create_dataset(name, data=data)
            else:
                grp.attrs[name] = json.dumps(data)

    def load(self, name: str) -> Any:
        with self._reading() as h5_read:
            grp = h5_read.get("store", {})
            if name in grp:
                return grp[name][:]
            if name in grp.attrs:
                return json.loads(grp.attrs[name])
            return None

    def add_pairs(self, pairs):
        with self:
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
        with self._reading() as h5_read:
            if "pairs" not in h5_read:
                return []
            return h5_read["pairs"][:].tolist()

    def pair_num(self):
        with self._reading() as h5_read:
            return len(h5_read["pairs"][:])

    def close(self):
        self._h5.close()

    def clone(self, dest: PathLike) -> H5ImageRepository:
        import shutil

        dest = Path(dest)
        shutil.copy2(self._h5_path, dest)
        return H5ImageRepository(dest)

    @classmethod
    def from_filename(cls, dirpath: PathLike, filename: str) -> H5ImageRepository:
        dirpath = Path(dirpath)
        return cls(str(dirpath / filename))
