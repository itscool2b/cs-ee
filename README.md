# GA vs SA on TSP — IB Extended Essay Experiment

Comparing Genetic Algorithm (GA) and Simulated Annealing (SA) on the Traveling Salesman Problem using TSPLIB benchmark instances.

**Research Question:** How do GA and SA compare in efficiency (convergence speed) and solution quality across TSP instances of increasing size?

## Quick Start

```bash
pip install numpy scipy matplotlib pandas numba

# Quick test (~30 sec, 1 trial, 10% FE budget)
python main.py --quick

# Full experiment (~25-30 min, 30 trials, full FE budgets)
python main.py
```

## TSPLIB Instances

| Instance | Cities | Optimal | FE Budget | Description |
|----------|--------|---------|-----------|-------------|
| berlin52 | 52 | 7,542 | 500,000 | Locations in Berlin |
| pr76 | 76 | 108,159 | 750,000 | Padberg/Rinaldi |
| kroA100 | 100 | 21,282 | 1,000,000 | Krolak/Felts/Nelson |
| d198 | 198 | 15,780 | 2,000,000 | Drilling problem |

All instances use EUC_2D edge weight type with `nint(sqrt(dx^2 + dy^2))` distance computation per TSPLIB specification.

## Fair Comparison Methodology

Algorithms are compared using **equal function evaluations (FEs)** — a hardware-independent, reproducible metric. One FE = evaluating one complete candidate tour's cost. FE budgets scale with instance size to give both algorithms adequate search time.

- **30 independent trials** per (instance, algorithm) pair using seeds 0-29
- Convergence recorded every 1,000 FEs
- Statistical significance tested with Mann-Whitney U (nonparametric)

## Algorithm Parameters

### Genetic Algorithm (GA)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Population size | 100 | Balance between diversity and evaluation cost |
| Selection | Tournament (k=5) | Standard pressure; k=5 gives ~93% chance of selecting best from sample |
| Crossover | Order Crossover (OX) | Preserves relative city ordering; standard for permutation encoding |
| Crossover rate | 0.9 | High exploitation; most offspring created via recombination |
| Mutation | Inversion | Reverses a subsequence; preserves adjacency better than swap |
| Mutation rate | 0.3 | Moderate rate to maintain diversity without disrupting good solutions |
| Elitism | 2 individuals | Prevents loss of best solutions found so far |

**FE counting:** `pop_size` for initial evaluation + 1 per child evaluated in each generation.

### Simulated Annealing (SA)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Neighborhood | 2-opt | Standard local search operator for TSP; O(1) delta computation |
| Cooling schedule | Geometric: T = T₀ × α^k, α auto-calibrated | α computed so T reaches ~0.001 at FE budget exhaustion; ensures exploration spans the full run |
| Initial temperature | Auto-calibrated | Binary search for T₀ giving ~80% acceptance of worse moves |
| Acceptance | Metropolis: exp(-Δ/T) | Standard Boltzmann acceptance criterion |

**FE counting:** 1 FE for initial tour + 1 per neighbor evaluated (delta computation counts as evaluating the neighbor tour).

**Temperature calibration:** 1,000 random 2-opt moves are sampled to collect worsening deltas. Binary search finds T₀ such that mean acceptance probability of worse moves ≈ 0.8. These calibration moves are not counted as FEs.

## Project Structure

```
├── main.py          # Entry point (--quick for fast testing)
├── tsp.py           # TSPLIB parser, distance matrix, tour cost
├── ga.py            # Genetic Algorithm implementation
├── sa.py            # Simulated Annealing implementation
├── experiment.py    # Trial runner, progress output, CSV saving
├── analysis.py      # Statistical tests + 5 publication-quality figures
├── README.md        # This file
├── berlin52.tsp     # TSPLIB instance files
├── pr76.tsp
├── kroA100.tsp
├── d198.tsp
└── results/         # Generated output directory
    ├── summary.csv
    ├── convergence.npz
    ├── statistical_summary.csv
    ├── fig1_convergence.png
    ├── fig2_boxplots.png
    ├── fig3_best_tours.png
    ├── fig4_gap_barchart.png
    └── fig5_summary_table.png
```

## Output

### Figures

1. **Convergence curves** (2x2 grid): Mean ± std shading, GA vs SA per instance
2. **Box plots** (2x2 grid): Final tour cost distributions per instance
3. **Best tours** (4x2 grid): City coordinates with tour edges for best solution found
4. **Gap bar chart**: Mean % gap from optimal, grouped by instance, with error bars
5. **Summary table**: Rendered as image with all key statistics

### Statistical Analysis

Per instance:
- **Descriptive statistics:** mean, std, min, max, % gap from optimal
- **Shapiro-Wilk test:** Tests normality to justify nonparametric approach
- **Mann-Whitney U test:** Nonparametric comparison at α = 0.05
- **Effect size:** Rank-biserial r = 1 − 2U/(n₁·n₂)

## Dependencies

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pandas>=2.0
numba>=0.59
```
