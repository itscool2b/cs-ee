"""Statistical analysis and publication-quality figures for GA vs SA on TSP."""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from experiment import INSTANCES, ALGORITHMS, RECORD_INTERVAL


def load_results(results_dir: str = "results"):
    """Load summary CSV and convergence data."""
    df = pd.read_csv(os.path.join(results_dir, "summary.csv"))
    conv = np.load(os.path.join(results_dir, "convergence.npz"))
    return df, conv


def _get_convergence_arrays(conv, inst_name: str, alg_name: str, n_trials: int):
    """Extract convergence arrays for a given instance/algorithm pair."""
    fe_arrays = []
    cost_arrays = []
    for i in range(n_trials):
        fe_key = f"{inst_name}_{alg_name}_{i}_fe"
        cost_key = f"{inst_name}_{alg_name}_{i}_cost"
        if fe_key in conv:
            fe_arrays.append(conv[fe_key])
            cost_arrays.append(conv[cost_key])
    return fe_arrays, cost_arrays


def _interpolate_convergence(fe_arrays, cost_arrays, max_fe: int):
    """Interpolate convergence curves to common FE points for mean/std computation."""
    common_fe = np.arange(0, max_fe + 1, RECORD_INTERVAL)
    if common_fe[0] == 0:
        common_fe[0] = 1  # avoid 0

    interpolated = []
    for fe, cost in zip(fe_arrays, cost_arrays):
        interp_cost = np.interp(common_fe, fe, cost)
        interpolated.append(interp_cost)

    interpolated = np.array(interpolated)
    return common_fe, interpolated


def plot_convergence(df, conv, results_dir: str = "results"):
    """Figure 1: Convergence curves (2x2 grid) with mean +/- std shading."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Convergence Curves: GA vs SA", fontsize=14, fontweight="bold")

    n_trials = df.groupby(["instance", "algorithm"]).size().iloc[0]

    for idx, (inst_name, inst_config) in enumerate(INSTANCES.items()):
        ax = axes[idx // 2][idx % 2]
        max_fe = inst_config["fe_budget"]
        # Check if quick mode was used
        sample_fe = conv.get(f"{inst_name}_GA_0_fe")
        if sample_fe is not None:
            max_fe = int(sample_fe[-1])

        for alg_name, color in [("GA", "#2196F3"), ("SA", "#FF5722")]:
            fe_arrays, cost_arrays = _get_convergence_arrays(conv, inst_name, alg_name, n_trials)
            if not fe_arrays:
                continue

            common_fe, interpolated = _interpolate_convergence(fe_arrays, cost_arrays, max_fe)
            mean = np.mean(interpolated, axis=0)
            std = np.std(interpolated, axis=0)

            ax.plot(common_fe, mean, label=alg_name, color=color, linewidth=1.5)
            ax.fill_between(common_fe, mean - std, mean + std, alpha=0.2, color=color)

        ax.axhline(y=inst_config["optimal"], color="green", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"Optimal ({inst_config['optimal']:,})")
        ax.set_title(f"{inst_name} ({max_fe:,} FEs)")
        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel("Tour Cost")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig1_convergence.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig1_convergence.png")


def plot_boxplots(df, results_dir: str = "results"):
    """Figure 2: Box plots (2x2 grid) of final tour cost distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Final Tour Cost Distributions: GA vs SA", fontsize=14, fontweight="bold")

    for idx, inst_name in enumerate(INSTANCES):
        ax = axes[idx // 2][idx % 2]
        inst_df = df[df["instance"] == inst_name]

        ga_costs = inst_df[inst_df["algorithm"] == "GA"]["best_cost"].values
        sa_costs = inst_df[inst_df["algorithm"] == "SA"]["best_cost"].values

        bp = ax.boxplot(
            [ga_costs, sa_costs],
            labels=["GA", "SA"],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor("#BBDEFB")
        bp["boxes"][1].set_facecolor("#FFCCBC")

        ax.axhline(y=INSTANCES[inst_name]["optimal"], color="green",
                   linestyle="--", linewidth=1, alpha=0.7,
                   label=f"Optimal ({INSTANCES[inst_name]['optimal']:,})")
        ax.set_title(f"{inst_name}")
        ax.set_ylabel("Tour Cost")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig2_boxplots.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig2_boxplots.png")


def plot_best_tours(df, conv, results_dir: str = "results"):
    """Figure 3: Best tours (4x2 grid) showing city coordinates + tour edges."""
    from tsp import parse_tsplib

    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    fig.suptitle("Best Tours Found: GA vs SA", fontsize=14, fontweight="bold")

    n_trials = df.groupby(["instance", "algorithm"]).size().iloc[0]

    for row, (inst_name, inst_config) in enumerate(INSTANCES.items()):
        instance = parse_tsplib(inst_config["file"], inst_config["optimal"])

        for col, alg_name in enumerate(["GA", "SA"]):
            ax = axes[row][col]

            # Find best trial
            inst_df = df[(df["instance"] == inst_name) & (df["algorithm"] == alg_name)]
            best_seed = inst_df.loc[inst_df["best_cost"].idxmin(), "seed"]
            best_cost = inst_df["best_cost"].min()

            tour_key = f"{inst_name}_{alg_name}_{best_seed}_tour"
            tour = conv[tour_key]

            # Plot tour edges
            coords = instance.coords
            for i in range(len(tour)):
                c1 = tour[i]
                c2 = tour[(i + 1) % len(tour)]
                ax.plot(
                    [coords[c1, 0], coords[c2, 0]],
                    [coords[c1, 1], coords[c2, 1]],
                    color="#2196F3" if alg_name == "GA" else "#FF5722",
                    linewidth=0.5, alpha=0.7,
                )

            # Plot cities
            ax.scatter(coords[:, 0], coords[:, 1], s=8, c="black", zorder=5)

            ax.set_title(f"{inst_name} — {alg_name} (cost={best_cost:,})")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig3_best_tours.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig3_best_tours.png")


def plot_gap_barchart(df, results_dir: str = "results"):
    """Figure 4: Mean % gap from optimal, grouped by instance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    instances = list(INSTANCES.keys())
    x = np.arange(len(instances))
    width = 0.35

    ga_gaps = []
    sa_gaps = []
    ga_errs = []
    sa_errs = []

    for inst_name in instances:
        inst_df = df[df["instance"] == inst_name]
        ga_gap = inst_df[inst_df["algorithm"] == "GA"]["gap_pct"]
        sa_gap = inst_df[inst_df["algorithm"] == "SA"]["gap_pct"]
        ga_gaps.append(ga_gap.mean())
        sa_gaps.append(sa_gap.mean())
        ga_errs.append(ga_gap.std())
        sa_errs.append(sa_gap.std())

    bars1 = ax.bar(x - width / 2, ga_gaps, width, yerr=ga_errs,
                   label="GA", color="#2196F3", alpha=0.8, capsize=4)
    bars2 = ax.bar(x + width / 2, sa_gaps, width, yerr=sa_errs,
                   label="SA", color="#FF5722", alpha=0.8, capsize=4)

    # Add value labels on bars
    for bar, val in zip(bars1, ga_gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, sa_gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Instance")
    ax.set_ylabel("Mean % Gap from Optimal")
    ax.set_title("Solution Quality: Mean Percentage Gap from Known Optimal",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(instances)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig4_gap_barchart.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig4_gap_barchart.png")


def statistical_analysis(df, results_dir: str = "results"):
    """Perform statistical tests and create Figure 5 (summary table) + CSV."""
    rows = []

    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")

    for inst_name in INSTANCES:
        inst_df = df[df["instance"] == inst_name]
        ga_costs = inst_df[inst_df["algorithm"] == "GA"]["best_cost"].values
        sa_costs = inst_df[inst_df["algorithm"] == "SA"]["best_cost"].values
        optimal = INSTANCES[inst_name]["optimal"]

        ga_gaps = (ga_costs - optimal) / optimal * 100
        sa_gaps = (sa_costs - optimal) / optimal * 100

        # Shapiro-Wilk test for normality
        if len(ga_costs) >= 3:
            sw_ga_stat, sw_ga_p = stats.shapiro(ga_costs)
            sw_sa_stat, sw_sa_p = stats.shapiro(sa_costs)
        else:
            sw_ga_stat = sw_ga_p = sw_sa_stat = sw_sa_p = float("nan")

        # Mann-Whitney U test
        if len(ga_costs) >= 2 and len(sa_costs) >= 2:
            u_stat, u_p = stats.mannwhitneyu(ga_costs, sa_costs, alternative="two-sided")
            # Rank-biserial effect size: r = 1 - 2U/(n1*n2)
            n1, n2 = len(ga_costs), len(sa_costs)
            effect_size = 1 - 2 * u_stat / (n1 * n2)
        else:
            u_stat = u_p = effect_size = float("nan")

        print(f"\n--- {inst_name} (optimal={optimal:,}) ---")
        print(f"  GA: mean={np.mean(ga_costs):,.1f}  std={np.std(ga_costs):,.1f}  "
              f"min={np.min(ga_costs):,}  max={np.max(ga_costs):,}  "
              f"gap={np.mean(ga_gaps):.2f}%")
        print(f"  SA: mean={np.mean(sa_costs):,.1f}  std={np.std(sa_costs):,.1f}  "
              f"min={np.min(sa_costs):,}  max={np.max(sa_costs):,}  "
              f"gap={np.mean(sa_gaps):.2f}%")
        print(f"  Shapiro-Wilk GA: W={sw_ga_stat:.4f}, p={sw_ga_p:.4f} "
              f"({'normal' if sw_ga_p > 0.05 else 'non-normal'})")
        print(f"  Shapiro-Wilk SA: W={sw_sa_stat:.4f}, p={sw_sa_p:.4f} "
              f"({'normal' if sw_sa_p > 0.05 else 'non-normal'})")
        print(f"  Mann-Whitney U: U={u_stat:.1f}, p={u_p:.6f} "
              f"({'significant' if u_p < 0.05 else 'not significant'} at alpha=0.05)")
        print(f"  Effect size (rank-biserial r): {effect_size:.4f}")

        for alg, costs, gaps in [("GA", ga_costs, ga_gaps), ("SA", sa_costs, sa_gaps)]:
            sw_stat = sw_ga_stat if alg == "GA" else sw_sa_stat
            sw_p = sw_ga_p if alg == "GA" else sw_sa_p
            rows.append({
                "Instance": inst_name,
                "Algorithm": alg,
                "n": len(costs),
                "Mean Cost": f"{np.mean(costs):,.1f}",
                "Std": f"{np.std(costs):,.1f}",
                "Min": f"{np.min(costs):,}",
                "Max": f"{np.max(costs):,}",
                "Mean Gap (%)": f"{np.mean(gaps):.2f}",
                "Shapiro-Wilk W": f"{sw_stat:.4f}",
                "Shapiro-Wilk p": f"{sw_p:.4f}",
                "Mann-Whitney U": f"{u_stat:.1f}",
                "Mann-Whitney p": f"{u_p:.6f}",
                "Effect Size r": f"{effect_size:.4f}",
            })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(results_dir, "statistical_summary.csv"), index=False)
    print(f"\nStatistical summary saved to {results_dir}/statistical_summary.csv")

    # Figure 5: Summary table as image
    fig, ax = plt.subplots(figsize=(16, 4 + len(rows) * 0.4))
    ax.axis("off")
    ax.set_title("Statistical Summary: GA vs SA on TSP",
                 fontsize=14, fontweight="bold", pad=20)

    # Simplified table for the figure
    table_data = []
    for _, row in stats_df.iterrows():
        table_data.append([
            row["Instance"], row["Algorithm"], row["n"],
            row["Mean Cost"], row["Std"], row["Min"], row["Max"],
            row["Mean Gap (%)"], row["Mann-Whitney p"], row["Effect Size r"],
        ])

    col_labels = [
        "Instance", "Algorithm", "n", "Mean Cost", "Std",
        "Min", "Max", "Gap (%)", "M-W p-value", "Effect r",
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#E3F2FD")
        table[(0, j)].set_text_props(fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        color = "#FFFFFF" if i % 2 == 1 else "#F5F5F5"
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)

    plt.savefig(os.path.join(results_dir, "fig5_summary_table.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig5_summary_table.png")

    return stats_df


def run_analysis(results_dir: str = "results"):
    """Run all analysis: load results, generate all 5 figures + statistical summary."""
    print("Loading results...")
    df, conv = load_results(results_dir)

    print(f"Loaded {len(df)} trial results")
    print(f"Instances: {df['instance'].unique()}")
    print(f"Algorithms: {df['algorithm'].unique()}")

    plot_convergence(df, conv, results_dir)
    plot_boxplots(df, results_dir)
    plot_best_tours(df, conv, results_dir)
    plot_gap_barchart(df, results_dir)
    stats_df = statistical_analysis(df, results_dir)

    print(f"\nAll figures and analysis saved to {results_dir}/")
    return stats_df
