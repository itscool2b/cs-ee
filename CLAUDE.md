# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IB Computer Science Extended Essay comparing Genetic Algorithm (GA) vs Simulated Annealing (SA) for the Traveling Salesman Problem on four TSPLIB instances (berlin52, pr76, kroA100, d198). Uses function evaluations (FEs) as the hardware-independent comparison metric.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pandas numba

# Full experiment: 30 trials × 2 algorithms × 4 instances (~30-75 min)
python main.py

# Quick test: 1 trial, 10% FE budgets (~20 sec)
python main.py --quick
```

Results go to `results/` (full) or `results_quick/` (quick mode).

## Architecture

Three-phase pipeline: **experiment → algorithms → analysis**.

- **main.py** — Entry point, dispatches `--quick` flag to experiment runner then analysis
- **experiment.py** — Trial orchestrator. Runs 30 seeded trials per (instance, algorithm) pair. Defines `INSTANCES` dict with file paths, known optima, and FE budgets. Saves `summary.csv` and `convergence.npz`
- **analysis.py** — Generates 6 publication figures (fig1–fig4, fig6–fig7; fig5 was the old stats-table PNG, now removed in favor of the CSV) and statistical tests (Shapiro-Wilk, Mann-Whitney U, rank-biserial effect size, Bonferroni-corrected). Convergence curves use median + IQR band to match the non-parametric framework. Writes `summary.csv`, `convergence.npz`, `statistical_summary.csv`, `convergence_efficiency.csv` to `results/`
- **algorithms/tsp.py** — TSPLIB EUC_2D parser, distance matrix builder (nint rounding), Numba-JIT `tour_cost()`, and `nearest_neighbor_cost()` multi-start baseline (@njit, tries all N starting cities)
- **algorithms/ga.py** — GA: pop=100, tournament(k=5), Order Crossover (OX), inversion mutation, elitism=2. Hot paths JIT-compiled
- **algorithms/sa.py** — SA: 2-opt neighborhood, O(1) delta eval, auto-calibrated geometric cooling (binary search for ~80% initial acceptance). Inner loop JIT-compiled with pre-generated random values

## Key Design Decisions

- **FE counting, not wall-clock time** — enables fair cross-algorithm comparison regardless of hardware
- **Numba JIT on hot paths** — `tour_cost`, crossover, mutation, SA inner loop, `nearest_neighbor_cost` are all `@njit`-decorated for performance
- **SA auto-calibration** — temperature calibrated per instance via binary search; calibration samples not counted as FEs
- **Seeds 0-29** — deterministic reproducibility across trials
- **RECORD_INTERVAL = 1000** — convergence checkpoints every 1000 FEs
- **Bonferroni correction** — α/4 = 0.0125 for quality comparisons (4 instances), α/12 ≈ 0.0042 for convergence-efficiency comparisons (4 instances × 3 gap thresholds)
- **Median + IQR** on convergence plots — matches the non-parametric stats framework (Shapiro-Wilk non-normal → Mann-Whitney U → rank-biserial r). Mean±std would imply Gaussian.
