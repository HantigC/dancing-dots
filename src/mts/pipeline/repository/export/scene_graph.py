import logging

import networkx as nx

from mts.core.scene_graph.model import Image, MatchKind, TwoViewEdge
from mts.pipeline.repository.base import BaseImageRepository

LOGGER = logging.getLogger(__name__)


def _add_image_nodes(scene_graph: nx.Graph, image_repository: BaseImageRepository) -> None:
    for image_id in image_repository.image_ids():
        height, width = image_repository.get_size_hw(image_id)
        filepath = image_repository.get_filepath(image_id)
        scene_graph.add_node(filepath, image=Image(height=height, width=width))


def as_minimal(image_repository: BaseImageRepository) -> nx.Graph:
    scene_graph = nx.Graph().to_undirected()
    _add_image_nodes(scene_graph, image_repository)

    for st_image_id, nd_image_id in image_repository.get_pairs():
        matches = image_repository.get_matches(st_image_id, nd_image_id)
        if matches is None or len(matches) == 0:
            continue
        st_filepath = image_repository.get_filepath(st_image_id)
        nd_filepath = image_repository.get_filepath(nd_image_id)
        scene_graph.add_edge(
            st_filepath,
            nd_filepath,
            weight=len(matches),
        )
    return scene_graph


def from_matches(
    image_repository: BaseImageRepository,
    matches_name: str = "matches",
    keypoints_name: str = "keypoints",
) -> nx.Graph:
    scene_graph = nx.Graph().to_undirected()
    _add_image_nodes(scene_graph, image_repository)

    for st_image_id, nd_image_id in image_repository.get_pairs():
        matches = image_repository.get_matches(st_image_id, nd_image_id, name=matches_name)
        if matches is None or len(matches) == 0:
            continue

        st_kpts_all = image_repository.get_keypoints(st_image_id, name=keypoints_name)
        nd_kpts_all = image_repository.get_keypoints(nd_image_id, name=keypoints_name)
        if st_kpts_all is None or nd_kpts_all is None:
            LOGGER.warning(
                "Skipping pair (%s, %s): matches present but keypoints missing",
                st_image_id,
                nd_image_id,
            )
            continue

        st_filepath = image_repository.get_filepath(st_image_id)
        nd_filepath = image_repository.get_filepath(nd_image_id)
        st_kpts = st_kpts_all[matches[:, 0]]
        nd_kpts = nd_kpts_all[matches[:, 1]]
        num_matches = len(matches)

        scene_graph.add_edge(
            st_filepath,
            nd_filepath,
            two_view=TwoViewEdge(
                st_filepath=st_filepath,
                nd_filepath=nd_filepath,
                kpts_for={st_filepath: st_kpts, nd_filepath: nd_kpts},
                match_kind=MatchKind.MATCHED,
                num_matches=num_matches,
            ),
            weight=num_matches,
        )
    return scene_graph
