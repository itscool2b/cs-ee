"""Trial runner: 30 seeds x 2 algorithms x 4 instances, with progress output and CSV saving."""

import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from tsp import parse_tsplib
from ga import run_ga
from sa import run_sa


# Instance configurations: (filename, optimal_cost, fe_budget)
INSTANCES = {
    "berlin52": {"file": "berlin52.tsp", "optimal": 7542, "fe_budget": 500_000},
    "pr76": {"file": "pr76.tsp", "optimal": 108159, "fe_budget": 750_000},
    "kroA100": {"file": "kroA100.tsp", "optimal": 21282, "fe_budget": 1_000_000},
    "d198": {"file": "d198.tsp", "optimal": 15780, "fe_budget": 2_000_000},
}

ALGORITHMS = {"GA": run_ga, "SA": run_sa}
RECORD_INTERVAL = 1000


@dataclass
class TrialResult:
    instance: str
    algorithm: str
    seed: int
    best_cost: int
    optimal: int
    gap_pct: float
    wall_time: float


def run_all_experiments(
    n_trials: int = 30,
    fe_scale: float = 1.0,
    base_dir: str = ".",
    results_dir: str = "results",
) -> list[TrialResult]:
    """Run all experiments and save results.

    Args:
        n_trials: Number of independent trials per (instance, algorithm) pair.
        fe_scale: Scale factor for FE budgets (0.1 for quick mode).
        base_dir: Directory containing .tsp files.
        results_dir: Directory to save results.

    Returns:
        List of TrialResult objects.
    """
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    convergence_data = {}  # (instance, algorithm) -> list of (fe, cost) arrays

    total_trials = len(INSTANCES) * len(ALGORITHMS) * n_trials
    trial_num = 0

    for inst_name, inst_config in INSTANCES.items():
        filepath = os.path.join(base_dir, inst_config["file"])
        instance = parse_tsplib(filepath, inst_config["optimal"])
        max_fe = int(inst_config["fe_budget"] * fe_scale)

        print(f"\n{'='*60}")
        print(f"Instance: {inst_name} (n={instance.dimension}, optimal={inst_config['optimal']}, FE budget={max_fe:,})")
        print(f"{'='*60}")

        for alg_name, alg_func in ALGORITHMS.items():
            conv_list = []

            for seed in range(n_trials):
                trial_num += 1
                rng = np.random.default_rng(seed)

                result = alg_func(
                    dist_matrix=instance.dist_matrix,
                    n_cities=instance.dimension,
                    max_fe=max_fe,
                    record_interval=RECORD_INTERVAL,
                    rng=rng,
                )

                gap = (result["best_cost"] - inst_config["optimal"]) / inst_config["optimal"] * 100

                trial = TrialResult(
                    instance=inst_name,
                    algorithm=alg_name,
                    seed=seed,
                    best_cost=result["best_cost"],
                    optimal=inst_config["optimal"],
                    gap_pct=round(gap, 4),
                    wall_time=round(result["wall_time"], 3),
                )
                all_results.append(trial)

                conv_list.append({
                    "fe": result["convergence_fe"],
                    "cost": result["convergence_cost"],
                    "best_tour": result["best_tour"],
                })

                print(
                    f"  [{trial_num}/{total_trials}] {alg_name} seed={seed:2d}: "
                    f"cost={result['best_cost']:>8,}  gap={gap:6.2f}%  "
                    f"time={result['wall_time']:.1f}s"
                )

            convergence_data[(inst_name, alg_name)] = conv_list

    # Save summary CSV
    df = pd.DataFrame([asdict(r) for r in all_results])
    df.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
    print(f"\nResults saved to {results_dir}/summary.csv")

    # Save convergence data as compressed npz
    conv_save = {}
    for (inst, alg), trials in convergence_data.items():
        for i, trial in enumerate(trials):
            conv_save[f"{inst}_{alg}_{i}_fe"] = trial["fe"]
            conv_save[f"{inst}_{alg}_{i}_cost"] = trial["cost"]
            conv_save[f"{inst}_{alg}_{i}_tour"] = trial["best_tour"]
    np.savez_compressed(os.path.join(results_dir, "convergence.npz"), **conv_save)
    print(f"Convergence data saved to {results_dir}/convergence.npz")

    return all_results
