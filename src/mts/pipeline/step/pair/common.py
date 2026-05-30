

from mts.core.types import DistancedTriple, PairType


def extract_possible_pairs(
    filtered_triples: list[DistancedTriple],
    image_ids: list[int],
) -> list[PairType[int]]:
    possible_pairs = []
    for distance_triple in filtered_triples:
        st_idx, nd_idx = distance_triple.st, distance_triple.nd
        st_image_id = image_ids[st_idx]
        nd_image_id = image_ids[nd_idx]
        st_image_id, nd_image_id = sorted((st_image_id, nd_image_id))
        possible_pairs.append((st_image_id, nd_image_id))
    return possible_pairs