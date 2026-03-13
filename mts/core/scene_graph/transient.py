import logging
from typing import Callable

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from mts.core.scene_graph.model import MatchKind, TwoViewEdge
from mts.core.scene_graph.nx import merge_path, nums


LOGGER = logging.getLogger(__name__)


def match_densely(
    st_fpath: str,
    nd_fpath: str,
    scene_graph: nx.Graph,
    extract_dense_matches: Callable[[str, str], np.ndarray],
) -> None:
    st_kpts, nd_kpts = extract_dense_matches(st_fpath, nd_fpath)
    if len(st_kpts) >= 500:
        LOGGER.info("Match densely (%s, %s)", st_fpath, nd_fpath,)
        scene_graph.add_edge(
            st_fpath,
            nd_fpath,
            two_view=TwoViewEdge(
                st_filepath=st_fpath,
                nd_filepath=nd_fpath,
                kpts_for={
                    st_fpath: st_kpts,
                    nd_fpath: nd_kpts,
                },
                match_kind=MatchKind.MATCHED,
                num_matches=len(st_fpath),
            ),
            weight=len(st_kpts),
        )


def grow_from_pairs(
    scene_graph: nx.Graph,
    pairs: list[tuple[str, str]],
    extract_dense_matches: Callable[[str, str], np.ndarray],
) -> None:
    for st_fpath, nd_fpath in tqdm(pairs):
        if not scene_graph.has_node(st_fpath):
            print(f"`{st_fpath}` not in the graph")
            continue
        if not scene_graph.has_node(nd_fpath):
            print(f"`{nd_fpath}` not in the graph")
            continue
        merged = False
        if not scene_graph.has_edge(st_fpath, nd_fpath):
            for kpts_path in list(
                nx.all_simple_paths(scene_graph, st_fpath, nd_fpath, cutoff=2)
            ):
                # TODO: Add a check if there are enough merges, if so, then merge, otherwise compute the matchings - which is more costly
                # TODO: Check if it can be reconstructed from different paths
                two_view: TwoViewEdge | None = merge_path(scene_graph, kpts_path)
                if two_view is not None and two_view.num_matches > 500:
                    node_from, via, node_to = kpts_path
                    scene_graph.add_edge(
                        node_from,
                        node_to,
                        two_view=two_view,
                        weight=two_view.num_matches,
                    )
                    count = nums(scene_graph)

                    LOGGER.info(
                        "merge new edge (%s, %s) via `%s`; (matched, merged) = (%d, %d);",
                        node_from,
                        node_to,
                        via,
                        count[MatchKind.MATCHED.value],
                        count[MatchKind.MERGED.value],
                    )
                    merged = True
                    break
        else:
            merged = True
        if not merged:
            match_densely(
                st_fpath,
                nd_fpath,
                scene_graph,
                extract_dense_matches,
            )
