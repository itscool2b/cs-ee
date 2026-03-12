"""Simulated Annealing for TSP with 2-opt neighborhood and auto-calibrated temperature."""

import time

import numpy as np
from numba import njit

from tsp import tour_cost


@njit(cache=True)
def _sa_inner_loop(dist_matrix, tour, current_cost, T, cooling_rate, max_fe,
                   record_interval, rand_vals, rand_ij):
    """JIT-compiled SA inner loop for maximum performance.

    rand_vals: pre-generated uniform random values for Metropolis acceptance.
    rand_ij: pre-generated random (i, j) pairs for 2-opt moves.
    """
    n = len(tour)
    best_cost = current_cost
    best_tour = tour.copy()

    # Pre-allocate convergence arrays (worst case: one record per interval)
    max_records = max_fe // record_interval + 2
    convergence_fe = np.empty(max_records, dtype=np.int64)
    convergence_cost = np.empty(max_records, dtype=np.int64)
    convergence_fe[0] = 1
    convergence_cost[0] = best_cost
    n_records = 1
    next_record = record_interval

    fe_count = 1
    idx = 0

    while fe_count < max_fe:
        i = rand_ij[idx, 0]
        j = rand_ij[idx, 1]
        idx += 1
        if idx >= len(rand_ij):
            idx = 0

        # Ensure valid i < j with i >= 1
        if i >= j:
            i, j = 1, n - 1
        if i < 1:
            i = 1
        if j >= n:
            j = n - 1

        # O(1) 2-opt delta
        a, b = tour[i - 1], tour[i]
        j1 = j + 1
        if j1 == n:
            j1 = 0
        c, d = tour[j], tour[j1]
        delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])

        fe_count += 1

        # Metropolis acceptance
        accept = False
        if delta < 0:
            accept = True
        elif T > 1e-12:
            if rand_vals[idx - 1] < np.exp(-delta / T):
                accept = True

        if accept:
            # Reverse segment tour[i:j+1]
            left = i
            right = j
            while left < right:
                tour[left], tour[right] = tour[right], tour[left]
                left += 1
                right -= 1
            current_cost += delta

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = tour.copy()

        T *= cooling_rate

        if fe_count >= next_record:
            convergence_fe[n_records] = fe_count
            convergence_cost[n_records] = best_cost
            n_records += 1
            next_record += record_interval

    # Final record
    if convergence_fe[n_records - 1] != fe_count:
        convergence_fe[n_records] = fe_count
        convergence_cost[n_records] = best_cost
        n_records += 1

    return best_cost, best_tour, convergence_fe[:n_records], convergence_cost[:n_records]


def _calibrate_temperature(dist_matrix, tour, rng, target_accept=0.8, n_samples=1000):
    """Binary search for initial temperature giving ~target_accept acceptance of worse moves."""
    n = len(tour)
    deltas = []
    attempts = 0
    while len(deltas) < n_samples and attempts < n_samples * 20:
        i = rng.integers(1, n)
        j = rng.integers(i + 1, n + (1 if i > 1 else 0))
        if j >= n:
            j = n - 1
        if i >= j:
            continue
        a, b = tour[i - 1], tour[i]
        c, d = tour[j], tour[(j + 1) % n]
        delta = int((dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d]))
        if delta > 0:
            deltas.append(delta)
        attempts += 1

    if not deltas:
        return 1000.0

    deltas = np.array(deltas, dtype=np.float64)
    lo, hi = 1e-6, float(np.max(deltas)) * 10
    for _ in range(50):
        mid = (lo + hi) / 2
        accept_rate = np.mean(np.exp(-deltas / mid))
        if accept_rate < target_accept:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_sa(dist_matrix, n_cities, max_fe, record_interval=1000, rng=None, cooling_rate=None):
    """Run Simulated Annealing on TSP."""
    if rng is None:
        rng = np.random.default_rng()

    start_time = time.perf_counter()

    # Random initial tour
    tour = np.arange(n_cities, dtype=np.int32)
    rng.shuffle(tour)
    current_cost = int(tour_cost(dist_matrix, tour))

    # Auto-calibrate temperature (not counted as FEs)
    T = _calibrate_temperature(dist_matrix, tour, rng)

    # Compute cooling rate from budget so T reaches ~1e-3 at the end
    if cooling_rate is None:
        T_final = 1e-3
        cooling_rate = (T_final / T) ** (1.0 / max_fe)

    # Pre-generate random values for the inner loop
    rand_vals = rng.random(max_fe).astype(np.float64)
    # Uniform (i, j) generation: two distinct values from [1, n), sorted
    pos_a = rng.integers(1, n_cities, size=max_fe).astype(np.int32)
    pos_b = rng.integers(1, n_cities - 1, size=max_fe).astype(np.int32)
    pos_b += (pos_b >= pos_a).astype(np.int32)  # shift to avoid duplicates
    rand_ij = np.empty((max_fe, 2), dtype=np.int32)
    rand_ij[:, 0] = np.minimum(pos_a, pos_b)
    rand_ij[:, 1] = np.maximum(pos_a, pos_b)

    best_cost, best_tour, conv_fe, conv_cost = _sa_inner_loop(
        dist_matrix, tour, current_cost, T, cooling_rate, max_fe,
        record_interval, rand_vals, rand_ij,
    )

    wall_time = time.perf_counter() - start_time

    return {
        "best_cost": int(best_cost),
        "best_tour": best_tour,
        "convergence_fe": conv_fe,
        "convergence_cost": conv_cost,
        "wall_time": wall_time,
    }
