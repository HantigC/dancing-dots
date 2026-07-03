from collections import Counter

import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx import edge_betweenness_centrality as betweenness
import numpy as np

from mts.core.matching.utils.merging import match_kpts
from mts.core.matching.utils.validation import validate_kps_matches
from mts.core.scene_graph.model import MatchKind, TwoViewEdge
from mts.core.types import DistancedTriple


def graph_from_distance_matrix(
    distance_matrix: np.ndarray,
    labels: list[str],
    threshold: float = 1,
) -> nx.Graph:
    n = distance_matrix.shape[0]
    G = nx.Graph()

    # add nodes
    if labels is None:
        labels = list(range(n))
    G.add_nodes_from(labels)

    # add weighted edges
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] <= threshold:
                G.add_edge(labels[i], labels[j], weight=distance_matrix[i, j])
    return G


def mst_from_distance_matrix(
    distance_matrix: np.ndarray,
    labels: list[str],
    threshold: float = 1,
) -> nx.Graph:
    distance_graph = graph_from_distance_matrix(
        distance_matrix,
        labels,
        threshold,
    )

    mst = nx.minimum_spanning_tree(distance_graph, weight="weight")
    return mst


def graph_from_distanced_triple(
    triples: list[DistancedTriple],
    labels: list[str],
    upper_threshold: float,
) -> nx.Graph:

    G = nx.Graph()

    G.add_nodes_from(labels)
    for distanced_triple in triples:
        if distanced_triple.distance <= upper_threshold:
            G.add_edge(
                labels[distanced_triple.st],
                labels[distanced_triple.nd],
                weight=distanced_triple.distance,
            )
    return G


def mst_pair_distanced_triple(
    triples: list[DistancedTriple],
    labels: list[str],
    upper_threshold: float,
) -> nx.Graph:
    distance_graph = graph_from_distanced_triple(
        triples,
        labels,
        upper_threshold,
    )

    mst = nx.minimum_spanning_tree(distance_graph, weight="weight")
    return mst


def match_path(
    kpts_graph: nx.Graph,
    kpts_path: tuple[str, str, str],
    return_kpts: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    start_from, intermediary_from = kpts_path[0], kpts_path[1]
    intermediary_to, end_to = kpts_path[1], kpts_path[2]
    edge_from = kpts_graph[start_from][intermediary_from]
    edge_to = kpts_graph[intermediary_to][end_to]

    two_view_from: TwoViewEdge = edge_from["two_view"]
    two_view_to: TwoViewEdge = edge_to["two_view"]

    kpts_from, kpts_intermediary_from = (
        two_view_from.kpts_for[start_from],
        two_view_from.kpts_for[intermediary_from],
    )
    kpts_intermediary_to, kpts_to = (
        two_view_to.kpts_for[intermediary_to],
        two_view_to.kpts_for[end_to],
    )

    indices_from, indices_to = match_kpts(
        kpts_intermediary_from,
        kpts_intermediary_to,
    )
    if return_kpts:
        return kpts_from, kpts_to, indices_from, indices_to

    return indices_from, indices_to


def merge_path(
    kpts_graph: nx.Graph,
    kpts_path: tuple[str, str, str],
    min_matches: int = 500,
) -> TwoViewEdge:
    kpts_from, kpts_to, indices_from, indices_to = match_path(
        kpts_graph,
        kpts_path,
        return_kpts=True,
    )

    if len(indices_from) < min_matches:
        return None
    node_from, via, node_to = kpts_path
    from_matched_kpts = kpts_from[indices_from]
    to_matched_kpts = kpts_to[indices_to]

    inlier_indices = validate_kps_matches(
        kpts_from[indices_from],
        kpts_to[indices_to],
        kpts_graph.nodes[node_from]["image"].hw,
        kpts_graph.nodes[node_to]["image"].hw,
    )
    if len(inlier_indices) < min_matches:
        return None
    from_inlier_kpts = from_matched_kpts[inlier_indices[:, 0]]
    to_inlier_kpts = to_matched_kpts[inlier_indices[:, 1]]

    return TwoViewEdge(
        st_filepath=node_from,
        nd_filepath=node_to,
        kpts_for={
            node_from: from_inlier_kpts,
            node_to: to_inlier_kpts,
        },
        match_kind=MatchKind.MERGED,
        num_matches=len(from_inlier_kpts),
    )


def prune_connections(scene_graph: nx.Graph) -> nx.Graph:
    def most_central_edge(G):
        centrality = betweenness(G, weight="weight")
        return max(centrality, key=centrality.get)

    comp = nx_comm.girvan_newman(
        scene_graph,
        most_valuable_edge=most_central_edge,
    )

    # Get the first split (top level)
    top_level = next(comp)
    # Map each node to its community index
    node_to_comm = {}
    for i, community in enumerate(top_level):
        for node in community:
            node_to_comm[node] = i

    # Copy the graph and remove edges that cross communities
    pruned_graph = scene_graph.copy()
    edges_to_remove = [
        (u, v) for u, v in pruned_graph.edges() if node_to_comm[u] != node_to_comm[v]
    ]
    pruned_graph.remove_edges_from(edges_to_remove)
    return pruned_graph


def nums(scene_graph: nx.Graph) -> Counter:
    return Counter(
        [
            edge_data["two_view"].match_kind.value
            for edge_data in scene_graph.edges.values()
        ]
    )


def extract_matches(
    scene_graph: nx.Graph,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[tuple[str, str], MatchKind]]:
    matches_dict = {}
    match_kinds = {}
    for (img1, img2), edge_data in scene_graph.edges.items():
        two_view: TwoViewEdge = edge_data["two_view"]
        matches = np.concatenate(
            [
                two_view.kpts_for[img1],
                two_view.kpts_for[img2],
            ],
            axis=1,
        )
        matches_dict.setdefault(img1, {})[img2] = matches
        match_kinds[(img1, img2)] = two_view.match_kind
    return matches_dict, match_kinds
