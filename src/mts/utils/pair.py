from mts.core.types import PairType
import numpy as np


def generate_exhaustive(no):
    return [(i, j) for i in range(no) for j in range(i + 1, no)]


def from_distance_matrix(
    distance_matrix: np.ndarray,
    threshold: float = 0.99,
) -> list[PairType[int]]:
    distance_mask = distance_matrix <= threshold

    possible_pairs = []
    for num, (sim_row, sim_mask_row) in enumerate(zip(distance_matrix, distance_mask)):
        sorted_indices = np.argsort(sim_row)
        sorted_mask = sim_mask_row[sorted_indices]
        for idx in sorted_indices[sorted_mask]:
            if num == idx:
                continue
            possible_pairs.append(tuple(sorted([num, int(idx)])))
    return possible_pairs
