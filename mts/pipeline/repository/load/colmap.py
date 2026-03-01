import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pycolmap

from mts.core.geometry.rigid3d import Rigid3D
from mts.core.types import PathLike
from mts.helpers.colmap.database import COLMAPDatabase, pair_id_to_image_ids
from mts.pipeline.repository.inmemeory import AlreadyExistsException, ImageRepository

LOGGER = logging.getLogger(__name__)


def import_from_colmap(
    image_repository: ImageRepository,
    db_filepath: PathLike,
    reconstructions_dirpath: PathLike | None = None,
) -> None:
    db_filepath = Path(db_filepath)
    db = COLMAPDatabase(db_filepath)

    LOGGER.info("Start exporting to colmap db")
    db_to_repo_ids = import_images(
        image_repository,
        db,
    )
    import_keypoints(
        image_repository,
        db,
        db_to_repo_ids,
    )
    import_matches(
        image_repository,
        db,
        db_to_repo_ids,
    )
    reconstructions_dirpath = reconstructions_dirpath or db_filepath.parent
    if reconstructions_dirpath is not None:
        import_reconstructions_dirpath(
            image_repository,
            reconstructions_dirpath,
        )


def import_images(
    image_repository: ImageRepository,
    db: COLMAPDatabase,
) -> dict[int, int]:
    db_to_repo_ids = {}
    for image_dict in db.fetch_images(as_dict=True):
        image_id = image_repository.add_image(image_dict["name"])
        db_to_repo_ids[image_dict["image_id"]] = image_id
    return db_to_repo_ids


def import_keypoints(
    image_repository: ImageRepository,
    db: COLMAPDatabase,
    db_to_repos_ids: dict[int, int],
) -> None:
    for db_id, repo_id in db_to_repos_ids.items():
        keypoints = db.select_kp(db_id)
        descriptors = db.select_descriptors(db_id)
        image_repository.add_descriptors(repo_id, descriptors)
        image_repository.add_keypoints(repo_id, keypoints)


def import_matches(
    image_repository: ImageRepository,
    db: COLMAPDatabase,
    db_to_repos_ids: dict[int, int],
) -> dict[int, int]:
    LOGGER.info("Export keypoints and descriptors")
    id_to_db_id = {}
    for match_dict in db.fetch_matches(as_dict=True):
        data = match_dict.get("data")
        if data is None:
            continue
        st_image_db_id, nd_image_db_id = pair_id_to_image_ids(
            match_dict["pair_id"],
        )
        matches = np.frombuffer(data, dtype=np.uint32).reshape(
            match_dict["rows"],
            match_dict["cols"],
        )
        st_image_db_id = db_to_repos_ids[st_image_db_id]
        nd_image_db_id = db_to_repos_ids[nd_image_db_id]
        image_repository.add_matches(st_image_db_id, nd_image_db_id, matches)

    return id_to_db_id


def import_reconstructions_dirpath(
    image_repository: ImageRepository,
    reconstructions_dirpath: PathLike,
) -> None:
    reconstructions_dirpath = Path(reconstructions_dirpath)
    reconstructions = []
    for reconstruction_dirpath in reconstructions_dirpath.glob("*"):
        if reconstruction_dirpath.is_dir():
            try:
                reconstruction = pycolmap.Reconstruction(reconstruction_dirpath)
            except ValueError:
                continue
            else:
                reconstructions.append(reconstruction)
    import_reconstructions(image_repository, reconstructions)


def import_reconstructions(
    image_repository: ImageRepository,
    reconstructions: list[pycolmap.Reconstruction],
) -> None:
    for map_index, reconstruction in enumerate(reconstructions):
        for index, image in reconstruction.images.items():
            rigid3d = image.cam_from_world()
            image_id = image_repository.get_image_id(image.name)
            image_repository.add_pose(
                image_id,
                Rigid3D(
                    deepcopy(rigid3d.rotation.matrix()),
                    deepcopy(rigid3d.translation),
                ),
            )
            try:
                image_repository.add_metadata(image_id, cluster=map_index)
            except AlreadyExistsException:
                LOGGER.warning("`%d` already has 'cluster' metadata", image_id)
                image_repository.update_metadata(image_id, cluster=map_index)
