
from mts.core.types import ImageId
from mts.pipeline.repository.base import BaseImageRepository


def get_cluster_map(image_repository: BaseImageRepository) -> dict[int, list[ImageId]]:
    cluster_map = {}
    for image_id in image_repository.image_ids():
        cluster_id = image_repository.get_metadata(image_id).get("cluster")
        cluster_map.setdefault(cluster_id, []).append(image_id)
    return cluster_map
