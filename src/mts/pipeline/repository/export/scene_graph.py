import networkx as nx

from mts.core.scene_graph.model import Image
from mts.pipeline.repository.base import BaseImageRepository


def as_minimal(image_repository: BaseImageRepository) -> nx.Graph:
    scene_graph = nx.Graph().to_undirected()
    for image_id, image in image_repository.iterate_over_images():
        height, width = image.shape[:2]
        image = Image(
            height=height,
            width=width,
        )
        scene_graph.add_node(
            image_repository.get_filepath(image_id),
            image=image,
        )

    for (st_image_id, nd_image_id), matches in image_repository.iterate_over_matches():
        st_filepath = image_repository.get_filepath(st_image_id)
        nd_filepath = image_repository.get_filepath(nd_image_id)
        scene_graph.add_edge(
            st_filepath,
            nd_filepath,
            weight=len(matches),
        )
    return scene_graph
