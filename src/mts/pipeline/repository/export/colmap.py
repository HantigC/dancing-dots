import logging
from sqlite3 import IntegrityError

from tqdm.auto import tqdm

from mts.core.types import PathLike
from mts.helpers.colmap.database import COLMAPDatabase
from mts.helpers.colmap.h5_to_db import CameraModel, create_camera
from mts.pipeline.repository.inmemeory import ImageRepository

LOGGER = logging.getLogger(__name__)


def export_to_colmap(
    image_repository: ImageRepository,
    db_filepath: PathLike,
    single_camera: bool = False,
    camera_model: str = CameraModel.PINHOLE,
    keypoints_name: str = "keypoints",
    descriptors_name: str = "descriptors",
    matches_name: str = "matches",
) -> COLMAPDatabase:
    db = COLMAPDatabase(db_filepath)
    db.create_tables()

    LOGGER.info("Start exporting to colmap db")
    try:
        id_to_db_id = add_keypoints(
            image_repository,
            db,
            camera_model,
            single_camera,
            keypoints_name=keypoints_name,
            descriptors_name=descriptors_name,
        )
        add_matches(
            id_to_db_id,
            image_repository,
            db,
            matches_name=matches_name,
        )
        db.commit()

        LOGGER.info("Commit the exporting to colmap db")
    except BaseException:
        raise
    finally:
        db.close()
    return db


def add_keypoints(
    image_repository: ImageRepository,
    db: COLMAPDatabase,
    camera_model: str,
    single_camera: bool = False,
    keypoints_name: str = "keypoints",
    descriptors_name: str = "descriptors",
) -> dict[int, int]:
    LOGGER.info("Export keypoints and descriptors")
    camera_id = None
    id_to_db_id = {}
    for image_id in tqdm(
        image_repository.image_ids(),
        total=image_repository.images_num(),
        desc="Add Keypoints",
    ):
        keypoints = image_repository.get_keypoints(image_id, name=keypoints_name)
        descriptors = image_repository.get_descriptors(image_id, name=descriptors_name)
        image_filepath = image_repository.get_filepath(image_id)
        # TODO: add more methods to to colmap-db: e.g. exists

        if camera_id is None or not single_camera:
            camera_id = create_camera(
                db,
                str(image_filepath),
                camera_model,
            )
        db_image_id = db.add_image(str(image_filepath), camera_id)

        if keypoints is not None:
            db.add_keypoints(db_image_id, keypoints)

        if descriptors is not None:
            db.add_descriptors(image_id, descriptors)
        id_to_db_id[image_id] = db_image_id

    return id_to_db_id


def add_matches(
    id_to_db_id: dict[int, int],
    image_repository: ImageRepository,
    db: COLMAPDatabase,
    matches_name: str = "matches",
) -> None:
    LOGGER.info("Export matches")
    for from_idx, to_idx in tqdm(
        image_repository.get_pairs(),
        total=image_repository.pair_num(),
        desc="Add matches",
    ):
        from_db_id = id_to_db_id[from_idx]
        to_db_id = id_to_db_id[to_idx]

        matches = image_repository.get_matches(from_idx, to_idx, name=matches_name)
        if matches is not None:
            try:
                db.add_matches(from_db_id, to_db_id, matches)
            except IntegrityError:
                LOGGER.warning("(%d, %d) is already inserted", from_idx, to_idx)
