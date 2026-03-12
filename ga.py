"""Genetic Algorithm for TSP with OX crossover, inversion mutation, and tournament selection."""

import time

import numpy as np
from numba import njit

from tsp import tour_cost


@njit(cache=True)
def _order_crossover(parent1, parent2, c1, c2):
    """Order Crossover (OX): preserves relative ordering of cities."""
    n = len(parent1)
    child = np.full(n, -1, dtype=np.int32)
    # Copy segment from parent1
    for k in range(c1, c2 + 1):
        child[k] = parent1[k]

    # Build membership array instead of set
    in_child = np.zeros(n, dtype=np.bool_)
    for k in range(c1, c2 + 1):
        in_child[parent1[k]] = True

    # Fill remaining positions with cities from parent2 in order
    pos = (c2 + 1) % n
    start = (c2 + 1) % n
    for offset in range(n):
        idx = (start + offset) % n
        city = parent2[idx]
        if not in_child[city]:
            child[pos] = city
            in_child[city] = True
            pos = (pos + 1) % n

    return child


@njit(cache=True)
def _inversion_mutation(tour, i, j):
    """Inversion mutation: reverse a random subsequence."""
    mutant = tour.copy()
    left = i
    right = j
    while left < right:
        mutant[left], mutant[right] = mutant[right], mutant[left]
        left += 1
        right -= 1
    return mutant


def run_ga(
    dist_matrix, n_cities, max_fe, record_interval=1000, rng=None,
    pop_size=100, crossover_rate=0.9, mutation_rate=0.3,
    elite_count=2, tournament_k=5,
):
    """Run Genetic Algorithm on TSP."""
    if rng is None:
        rng = np.random.default_rng()

    start_time = time.perf_counter()

    # Initialize population
    population = np.array([rng.permutation(n_cities).astype(np.int32) for _ in range(pop_size)])
    fitness = np.array([tour_cost(dist_matrix, ind) for ind in population])
    fe_count = pop_size

    best_idx = np.argmin(fitness)
    best_cost = int(fitness[best_idx])
    best_tour = population[best_idx].copy()

    convergence_fe = [fe_count]
    convergence_cost = [best_cost]
    next_record = record_interval

    while fe_count < max_fe:
        new_population = []
        new_fitness = []

        # Elitism
        elite_indices = np.argsort(fitness)[:elite_count]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
            new_fitness.append(fitness[idx])

        # Generate offspring
        while len(new_population) < pop_size:
            if fe_count >= max_fe:
                break

            # Tournament selection
            idx1 = rng.choice(len(population), size=tournament_k, replace=False)
            p1 = population[idx1[np.argmin(fitness[idx1])]].copy()
            idx2 = rng.choice(len(population), size=tournament_k, replace=False)
            p2 = population[idx2[np.argmin(fitness[idx2])]].copy()

            # Crossover
            if rng.random() < crossover_rate:
                cuts = sorted(rng.choice(n_cities, size=2, replace=False))
                child = _order_crossover(p1, p2, cuts[0], cuts[1])
            else:
                child = p1.copy()

            # Mutation
            if rng.random() < mutation_rate:
                cuts = sorted(rng.choice(n_cities, size=2, replace=False))
                child = _inversion_mutation(child, cuts[0], cuts[1])

            child_cost = tour_cost(dist_matrix, child)
            fe_count += 1

            new_population.append(child)
            new_fitness.append(child_cost)

            if child_cost < best_cost:
                best_cost = child_cost
                best_tour = child.copy()

            if fe_count >= next_record:
                convergence_fe.append(fe_count)
                convergence_cost.append(best_cost)
                next_record += record_interval

        population = np.array(new_population[:pop_size])
        fitness = np.array(new_fitness[:pop_size])

    if convergence_fe[-1] != fe_count:
        convergence_fe.append(fe_count)
        convergence_cost.append(best_cost)

    wall_time = time.perf_counter() - start_time

    return {
        "best_cost": best_cost,
        "best_tour": best_tour,
        "convergence_fe": np.array(convergence_fe),
        "convergence_cost": np.array(convergence_cost),
        "wall_time": wall_time,
    }
