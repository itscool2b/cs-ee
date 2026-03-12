"""Single entry point for GA vs SA on TSP experiment.

Usage:
    python main.py          # Full experiment (30 trials, full FE budgets)
    python main.py --quick  # Quick test (1 trial, 10% FE budgets)
"""

import argparse
import time

from experiment import run_all_experiments
from analysis import run_analysis


def main():
    parser = argparse.ArgumentParser(
        description="GA vs SA on TSP — IB Extended Essay Experiment"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 1 trial with 10%% FE budget (~30 sec)",
    )
    args = parser.parse_args()

    if args.quick:
        n_trials = 1
        fe_scale = 0.1
        results_dir = "results_quick"
        print("QUICK MODE: 1 trial, 10% FE budget (output: results_quick/)")
    else:
        n_trials = 30
        fe_scale = 1.0
        results_dir = "results"
        print("FULL MODE: 30 trials, full FE budgets (output: results/)")

    print()
    start = time.perf_counter()

    # Phase 1: Run experiments
    run_all_experiments(
        n_trials=n_trials,
        fe_scale=fe_scale,
        results_dir=results_dir,
    )

    # Phase 2: Analysis and figures
    print("\n" + "=" * 60)
    print("GENERATING ANALYSIS AND FIGURES")
    print("=" * 60)
    run_analysis(results_dir=results_dir)

    elapsed = time.perf_counter() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
