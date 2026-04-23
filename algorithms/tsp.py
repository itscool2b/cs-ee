"""TSPLIB parser, distance matrix builder, and tour cost evaluator."""

import re
from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class TSPInstance:
    name: str
    dimension: int
    coords: np.ndarray      # (n, 2) float64
    dist_matrix: np.ndarray  # (n, n) int32
    optimal: int


def parse_tsplib(filepath: str, optimal: int) -> TSPInstance:
    """Parse a TSPLIB file in EUC_2D format and build the distance matrix."""
    with open(filepath) as f:
        lines = f.readlines()

    name = ""
    dimension = 0
    coords = []
    reading_coords = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line == "EOF":
            break

        if line == "NODE_COORD_SECTION":
            reading_coords = True
            continue

        if reading_coords:
            parts = line.split()
            # node_id x y — skip node_id
            coords.append((float(parts[1]), float(parts[2])))
        else:
            # Parse header fields — handle "KEY: val" and "KEY : val"
            m = re.match(r"^(\w+)\s*:\s*(.+)$", line)
            if m:
                key, val = m.group(1), m.group(2).strip()
                if key == "NAME":
                    name = val
                elif key == "DIMENSION":
                    dimension = int(val)

    coords_arr = np.array(coords, dtype=np.float64)
    assert coords_arr.shape == (dimension, 2), (
        f"Expected {dimension} coords, got {coords_arr.shape[0]}"
    )

    dist_matrix = _build_distance_matrix(coords_arr)

    return TSPInstance(
        name=name,
        dimension=dimension,
        coords=coords_arr,
        dist_matrix=dist_matrix,
        optimal=optimal,
    )


def _build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Build symmetric distance matrix using TSPLIB EUC_2D rounding: nint(sqrt(dx^2+dy^2))."""
    n = len(coords)
    dx = coords[:, 0].reshape(n, 1) - coords[:, 0].reshape(1, n)
    dy = coords[:, 1].reshape(n, 1) - coords[:, 1].reshape(1, n)
    raw = np.sqrt(dx ** 2 + dy ** 2)
    # TSPLIB EUC_2D: round to nearest integer
    dist = np.rint(raw).astype(np.int32)
    np.fill_diagonal(dist, 0)
    return dist


@njit(cache=True)
def tour_cost(dist_matrix, tour):
    """Compute total tour cost by summing edge weights around the cycle."""
    n = len(tour)
    cost = 0
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        cost += dist_matrix[tour[i], tour[j]]
    return cost


@njit(cache=True)
def nearest_neighbor_cost(dist_matrix):
    """Multi-start nearest-neighbor baseline: minimum tour cost over all starting cities."""
    n = len(dist_matrix)
    best = np.iinfo(np.int64).max
    for start in range(n):
        visited = np.zeros(n, dtype=np.bool_)
        tour = np.empty(n, dtype=np.int32)
        tour[0] = start
        visited[start] = True
        for step in range(1, n):
            current = tour[step - 1]
            best_dist = np.iinfo(np.int32).max
            best_city = -1
            for city in range(n):
                if not visited[city] and dist_matrix[current, city] < best_dist:
                    best_dist = dist_matrix[current, city]
                    best_city = city
            tour[step] = best_city
            visited[best_city] = True
        cost = tour_cost(dist_matrix, tour)
        if cost < best:
            best = cost
    return int(best)
