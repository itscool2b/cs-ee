# GA vs SA on TSP

IB Computer Science Extended Essay experiment comparing Genetic Algorithm and Simulated Annealing on the Traveling Salesman Problem.

**Research Question:** To what extent do genetic algorithm and simulated annealing differ in efficiency (convergence iterations) and solution quality when solving the Traveling Salesman Problem on TSPLIB instances berlin52, pr76, kroA100, and d198?

## Setup

Requires Python 3.10+ with numpy, scipy, matplotlib, pandas, and numba.

**NixOS** (recommended — handles all dependencies):
```bash
nix-shell
python main.py
```

**pip/venv:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pandas numba
python main.py
```

## Usage

```bash
python main.py          # Full experiment: 30 trials, full FE budgets → results/
python main.py --quick  # Test run: 1 trial, 10% FE budgets → results_quick/
```

Full run takes ~30–75 minutes (hardware-dependent). Quick mode takes ~20 seconds.

## Instances

| Instance | Cities | Known Optimal | FE Budget |
|----------|--------|---------------|-----------|
| berlin52 | 52 | 7,542 | 500,000 |
| pr76 | 76 | 108,159 | 750,000 |
| kroA100 | 100 | 21,282 | 1,000,000 |
| d198 | 198 | 15,780 | 2,000,000 |

All use EUC_2D edge weights: `nint(sqrt(dx^2 + dy^2))` per TSPLIB spec. FE budgets scale with instance size.

## Methodology

**Fair comparison metric:** Equal function evaluations (FEs). One FE = evaluating one candidate tour's cost. FE count is hardware-independent and reproducible, unlike wall-clock time.

**Statistical design:** 30 independent trials per (instance, algorithm) pair using deterministic seeds 0-29. Convergence recorded every 1,000 FEs.

**Efficiency metric:** FEs required to reach within X% of optimal (X = 15, 10, 5), compared with Mann-Whitney U test.

**Quality metric:** Final tour cost after full FE budget, compared with Mann-Whitney U test, Shapiro-Wilk normality check, and rank-biserial effect size.

## Algorithm Parameters

### Genetic Algorithm

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Population | 100 | Balances diversity against evaluation cost per generation |
| Selection | Tournament, k=5 | ~93% probability of selecting best from sample |
| Crossover | Order Crossover (OX), rate=0.9 | Preserves relative city ordering; standard for permutations |
| Mutation | Inversion, rate=0.3 | Reverses a subsequence; preserves adjacency better than swap |
| Elitism | 2 | Prevents loss of best solutions between generations |

FE counting: `pop_size` for initial population + 1 per child evaluated.

### Simulated Annealing

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Neighborhood | 2-opt (O(1) delta) | Standard TSP local search; computes cost change from 4 edge lookups |
| Cooling | Geometric, rate auto-calibrated | Rate computed as `(T_final / T0)^(1/max_fe)` so temperature spans the full budget |
| Initial temp | Auto-calibrated | Binary search over 1,000 sample moves for ~80% acceptance of worse moves |
| T_final | 0.001 | Near-zero; SA converges to hill-climbing at budget exhaustion |
| Acceptance | Metropolis: `exp(-delta/T)` | Standard Boltzmann criterion |

FE counting: 1 for initial tour + 1 per neighbor evaluated.

Temperature calibration samples are not counted as FEs.

## Output

### Data files

| File | Contents |
|------|----------|
| `summary.csv` | 240 rows: best cost, gap%, wall time for every trial |
| `convergence.npz` | Compressed convergence curves and best tours for all trials |
| `statistical_summary.csv` | Descriptive stats, Shapiro-Wilk, Mann-Whitney U, effect sizes |
| `convergence_efficiency.csv` | FEs-to-threshold for each trial at 15%, 10%, 5% gap levels |

### Figures (300 DPI PNGs)

1. **Convergence curves** — Mean +/- std shading per instance, GA vs SA
2. **Box plots** — Final tour cost distributions per instance
3. **Best tours** — City coordinates with tour edges for the best solution found
4. **Gap bar chart** — Mean % gap from optimal with std error bars
5. **Summary table** — All key statistics rendered as an image
6. **Convergence efficiency** — Box plots of FEs needed to reach 15%, 10%, 5% gap thresholds

### Statistical tests

- **Shapiro-Wilk:** Tests normality to justify nonparametric approach
- **Mann-Whitney U:** Nonparametric two-sided test at alpha = 0.05
- **Effect size:** Rank-biserial r = 1 - 2U/(n1 * n2)

Applied to both solution quality (final costs) and convergence efficiency (FEs-to-threshold).

## Project Structure

```
main.py           Entry point (--quick for testing)
experiment.py     Trial runner, progress output, CSV/npz saving
analysis.py       Statistical tests + 6 figures
algorithms/       Algorithm implementations
  tsp.py          TSPLIB parser, distance matrix, tour cost
  ga.py           Genetic Algorithm
  sa.py           Simulated Annealing
datasets/         TSPLIB instance files
  berlin52.tsp
  pr76.tsp
  kroA100.tsp
  d198.tsp
results/          Output (CSV, npz, PNG figures)
shell.nix         Nix development environment
```

## Dependencies

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pandas>=2.0
numba>=0.59
```

## License

MIT
