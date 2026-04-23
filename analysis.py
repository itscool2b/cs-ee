"""Statistical analysis and publication-quality figures for GA vs SA on TSP."""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import Voronoi, Delaunay
from matplotlib.collections import PolyCollection
from matplotlib.colors import PowerNorm

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
    """Figure 1: Convergence curves (2x2 grid) with median + IQR shading."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Convergence Curves (median, IQR band): GA vs SA", fontsize=14, fontweight="bold")

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
            median = np.median(interpolated, axis=0)
            q25, q75 = np.percentile(interpolated, [25, 75], axis=0)

            ax.plot(common_fe, median, label=alg_name, color=color, linewidth=1.5)
            ax.fill_between(common_fe, q25, q75, alpha=0.2, color=color)

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
            tick_labels=["GA", "SA"],
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


def plot_best_tours(df, conv, results_dir: str = "results", base_dir: str = "."):
    """Figure 3: Best tours (4x2 grid) showing city coordinates + tour edges."""
    from algorithms.tsp import parse_tsplib

    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    fig.suptitle("Best Tours Found: GA vs SA", fontsize=14, fontweight="bold")

    n_trials = df.groupby(["instance", "algorithm"]).size().iloc[0]

    for row, (inst_name, inst_config) in enumerate(INSTANCES.items()):
        instance = parse_tsplib(os.path.join(base_dir, inst_config["file"]), inst_config["optimal"])

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
                    linewidth=1.0, alpha=0.7,
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


def plot_gap_barchart(df, results_dir: str = "results", base_dir: str = "."):
    """Figure 4: Mean % gap from optimal, grouped by instance, with NN baseline."""
    from algorithms.tsp import parse_tsplib, nearest_neighbor_cost

    fig, ax = plt.subplots(figsize=(10, 6))

    instances = list(INSTANCES.keys())
    x = np.arange(len(instances))
    width = 0.35

    ga_gaps = []
    sa_gaps = []
    ga_errs = []
    sa_errs = []
    nn_gaps = []

    for inst_name in instances:
        inst_df = df[df["instance"] == inst_name]
        ga_gap = inst_df[inst_df["algorithm"] == "GA"]["gap_pct"]
        sa_gap = inst_df[inst_df["algorithm"] == "SA"]["gap_pct"]
        ga_gaps.append(ga_gap.mean())
        sa_gaps.append(sa_gap.mean())
        ga_errs.append(ga_gap.std())
        sa_errs.append(sa_gap.std())

        inst_config = INSTANCES[inst_name]
        instance = parse_tsplib(os.path.join(base_dir, inst_config["file"]),
                                inst_config["optimal"])
        nn_cost = nearest_neighbor_cost(instance.dist_matrix)
        nn_gaps.append((nn_cost - inst_config["optimal"]) / inst_config["optimal"] * 100)

    # Clip lower error bars at zero (gap can't be negative)
    ga_lo = [min(g, e) for g, e in zip(ga_gaps, ga_errs)]
    sa_lo = [min(g, e) for g, e in zip(sa_gaps, sa_errs)]
    bars1 = ax.bar(x - width / 2, ga_gaps, width,
                   yerr=[ga_lo, ga_errs],
                   label="GA", color="#2196F3", alpha=0.8, capsize=4)
    bars2 = ax.bar(x + width / 2, sa_gaps, width,
                   yerr=[sa_lo, sa_errs],
                   label="SA", color="#FF5722", alpha=0.8, capsize=4)

    # NN baseline markers
    for i, nn_gap in enumerate(nn_gaps):
        ax.plot(x[i], nn_gap, marker='v', color='#4CAF50', markersize=10,
                zorder=5, label="NN baseline" if i == 0 else None)
        ax.text(x[i], nn_gap + 0.5, f"{nn_gap:.1f}%",
                ha="center", va="bottom", fontsize=8, color="#4CAF50")

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
    """Perform statistical tests and save summary CSV."""
    rows = []
    n_comparisons = len(INSTANCES)  # 4 tests → Bonferroni α = 0.05/4
    alpha_corrected = 0.05 / n_comparisons

    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS (Bonferroni-corrected alpha={alpha_corrected})")
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
        p_corrected = min(u_p * n_comparisons, 1.0)
        print(f"  Mann-Whitney U: U={u_stat:.1f}, p={u_p:.6f}, "
              f"p_corrected={p_corrected:.6f} "
              f"({'significant' if p_corrected < 0.05 else 'not significant'} "
              f"at Bonferroni-corrected alpha={alpha_corrected})")
        print(f"  Effect size (rank-biserial r): {effect_size:.4f}")

        for idx_alg, (alg, costs, gaps) in enumerate([("GA", ga_costs, ga_gaps), ("SA", sa_costs, sa_gaps)]):
            sw_stat = sw_ga_stat if alg == "GA" else sw_sa_stat
            sw_p = sw_ga_p if alg == "GA" else sw_sa_p
            row = {
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
            }
            # Only show pairwise test stats on the first row per instance
            if idx_alg == 0:
                row["Mann-Whitney U"] = f"{u_stat:.1f}"
                row["Mann-Whitney p"] = f"{u_p:.6f}"
                row["Corrected p"] = f"{p_corrected:.6f}"
                row["Effect Size r"] = f"{effect_size:.4f}"
            else:
                row["Mann-Whitney U"] = ""
                row["Mann-Whitney p"] = ""
                row["Corrected p"] = ""
                row["Effect Size r"] = ""
            rows.append(row)

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(results_dir, "statistical_summary.csv"), index=False)
    print(f"\nStatistical summary saved to {results_dir}/statistical_summary.csv")

    return stats_df


def _fe_to_threshold(fe_array, cost_array, optimal, gap_pct):
    """Find the first FE count where cost is within gap_pct% of optimal.

    Returns (fe, reached): fe is the crossing point if reached, otherwise the
    final FE (budget cap) as a right-censored upper bound. reached is True
    only when the threshold was actually crossed.
    """
    target = optimal * (1 + gap_pct / 100.0)
    indices = np.where(cost_array <= target)[0]
    if len(indices) > 0:
        return int(fe_array[indices[0]]), True
    return int(fe_array[-1]), False


def convergence_efficiency(df, conv, results_dir: str = "results"):
    """Compute and plot FEs-to-threshold for convergence efficiency comparison.

    Measures how many FEs each algorithm needs to reach within X% of optimal.
    """
    n_trials = df.groupby(["instance", "algorithm"]).size().iloc[0]
    thresholds = [15, 10, 5]  # % gap from optimal

    # Compute FEs-to-threshold for every trial
    eff_rows = []
    for inst_name, inst_config in INSTANCES.items():
        optimal = inst_config["optimal"]
        for alg_name in ALGORITHMS:
            fe_arrays, cost_arrays = _get_convergence_arrays(conv, inst_name, alg_name, n_trials)
            for i, (fe, cost) in enumerate(zip(fe_arrays, cost_arrays)):
                for thresh in thresholds:
                    fe_val, reached = _fe_to_threshold(fe, cost, optimal, thresh)
                    eff_rows.append({
                        "instance": inst_name,
                        "algorithm": alg_name,
                        "seed": i,
                        "threshold_pct": thresh,
                        "fe_to_threshold": fe_val,
                        "reached": reached,
                    })

    eff_df = pd.DataFrame(eff_rows)
    eff_df.to_csv(os.path.join(results_dir, "convergence_efficiency.csv"), index=False)

    # Figure 6: Box plots of FEs-to-threshold (one subplot per threshold)
    fig, axes = plt.subplots(1, len(thresholds), figsize=(6 * len(thresholds), 7))
    fig.suptitle("Convergence Efficiency: FEs to Reach X% Gap from Optimal",
                 fontsize=14, fontweight="bold")

    from matplotlib.patches import Patch

    for t_idx, thresh in enumerate(thresholds):
        ax = axes[t_idx]
        thresh_df = eff_df[eff_df["threshold_pct"] == thresh]

        instances = list(INSTANCES.keys())
        x = np.arange(len(instances))
        width = 0.35

        ga_data = []
        sa_data = []
        ga_reach = []
        sa_reach = []
        for inst_name in instances:
            ga_rows = thresh_df[(thresh_df["instance"] == inst_name) &
                                (thresh_df["algorithm"] == "GA")]
            sa_rows = thresh_df[(thresh_df["instance"] == inst_name) &
                                (thresh_df["algorithm"] == "SA")]
            ga_data.append(ga_rows["fe_to_threshold"].values)
            sa_data.append(sa_rows["fe_to_threshold"].values)
            ga_reach.append(int(ga_rows["reached"].sum()))
            sa_reach.append(int(sa_rows["reached"].sum()))

        bp_ga = ax.boxplot([d for d in ga_data], positions=x - width / 2,
                           widths=width * 0.8, patch_artist=True)
        bp_sa = ax.boxplot([d for d in sa_data], positions=x + width / 2,
                           widths=width * 0.8, patch_artist=True)

        for box in bp_ga["boxes"]:
            box.set_facecolor("#BBDEFB")
        for box in bp_sa["boxes"]:
            box.set_facecolor("#FFCCBC")

        ax.set_xticks(x)
        ax.set_xticklabels(instances, fontsize=9)
        ax.set_ylabel("Function Evaluations")
        ax.set_title(f"FEs to reach {thresh}% gap")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(True, alpha=0.3, axis="y")

        # Reach counts below each box: trials that actually crossed the
        # threshold vs total. Values below n_trials mean the FE figure is
        # right-censored at the budget.
        for i in range(len(instances)):
            ax.text(x[i] - width / 2, -0.10, f"{ga_reach[i]}/{n_trials}",
                    ha="center", va="top", fontsize=7, color="#1565C0",
                    transform=ax.get_xaxis_transform())
            ax.text(x[i] + width / 2, -0.10, f"{sa_reach[i]}/{n_trials}",
                    ha="center", va="top", fontsize=7, color="#BF360C",
                    transform=ax.get_xaxis_transform())

        ax.legend(handles=[Patch(facecolor="#BBDEFB", label="GA"),
                           Patch(facecolor="#FFCCBC", label="SA")], fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig6_convergence_efficiency.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig6_convergence_efficiency.png")

    # Statistical tests on convergence efficiency
    # 4 instances × 3 thresholds = 12 comparisons
    n_eff_comparisons = len(INSTANCES) * len(thresholds)
    alpha_eff = 0.05 / n_eff_comparisons
    print(f"\n{'='*80}")
    print(f"CONVERGENCE EFFICIENCY ANALYSIS (Bonferroni-corrected alpha={alpha_eff:.4f})")
    print(f"{'='*80}")

    for thresh in thresholds:
        print(f"\n--- Threshold: {thresh}% gap from optimal ---")
        for inst_name in INSTANCES:
            thresh_df = eff_df[(eff_df["threshold_pct"] == thresh) &
                               (eff_df["instance"] == inst_name)]
            ga_rows = thresh_df[thresh_df["algorithm"] == "GA"]
            sa_rows = thresh_df[thresh_df["algorithm"] == "SA"]
            ga_fe = ga_rows["fe_to_threshold"].values
            sa_fe = sa_rows["fe_to_threshold"].values
            ga_r = int(ga_rows["reached"].sum())
            sa_r = int(sa_rows["reached"].sum())

            ga_mean = np.mean(ga_fe)
            sa_mean = np.mean(sa_fe)

            if len(ga_fe) >= 2 and len(sa_fe) >= 2:
                u_stat, u_p = stats.mannwhitneyu(ga_fe, sa_fe, alternative="two-sided")
                n1, n2 = len(ga_fe), len(sa_fe)
                r = 1 - 2 * u_stat / (n1 * n2)
                p_corr = min(u_p * n_eff_comparisons, 1.0)
            else:
                u_p = r = p_corr = float("nan")

            print(f"  {inst_name}: "
                  f"GA {ga_r}/{n_trials} reached (mean={ga_mean:,.0f})  "
                  f"SA {sa_r}/{n_trials} reached (mean={sa_mean:,.0f})  "
                  f"p={u_p:.6f}  p_corrected={p_corr:.6f}  r={r:.4f}")

    return eff_df


def _clip_polygon_to_bbox(polygon, bbox):
    """Clip a polygon to a rectangular bounding box (Sutherland-Hodgman)."""
    x_min, x_max, y_min, y_max = bbox

    def _clip_edge(poly, inside_fn, intersect_fn):
        if len(poly) == 0:
            return poly
        clipped = []
        prev = poly[-1]
        for curr in poly:
            if inside_fn(curr):
                if not inside_fn(prev):
                    clipped.append(intersect_fn(prev, curr))
                clipped.append(curr)
            elif inside_fn(prev):
                clipped.append(intersect_fn(prev, curr))
            prev = curr
        return np.array(clipped) if clipped else np.empty((0, 2))

    def _lerp(a, b, t):
        return a + t * (b - a)

    poly = polygon
    poly = _clip_edge(poly,
                      lambda p: p[0] >= x_min,
                      lambda a, b: _lerp(a, b, (x_min - a[0]) / (b[0] - a[0])))
    poly = _clip_edge(poly,
                      lambda p: p[0] <= x_max,
                      lambda a, b: _lerp(a, b, (x_max - a[0]) / (b[0] - a[0])))
    poly = _clip_edge(poly,
                      lambda p: p[1] >= y_min,
                      lambda a, b: _lerp(a, b, (y_min - a[1]) / (b[1] - a[1])))
    poly = _clip_edge(poly,
                      lambda p: p[1] <= y_max,
                      lambda a, b: _lerp(a, b, (y_max - a[1]) / (b[1] - a[1])))
    return poly


def _voronoi_finite_polygons(vor, bbox):
    """Convert scipy Voronoi to finite polygons clipped to bbox.

    Returns list of (M, 2) arrays, one polygon per input point.
    """
    center = vor.points.mean(axis=0)
    regions_out = []

    for point_idx in range(len(vor.points)):
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]

        if not region:
            regions_out.append(np.empty((0, 2)))
            continue

        if -1 not in region:
            polygon = vor.vertices[region]
            polygon = _clip_polygon_to_bbox(polygon, bbox)
            regions_out.append(polygon)
            continue

        # Infinite region: collect finite vertices + extend infinite ridges
        new_vertices = [vor.vertices[v] for v in region if v >= 0]

        for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
            if p1 != point_idx and p2 != point_idx:
                continue
            rv = vor.ridge_vertices[ridge_idx]
            if -1 not in rv:
                continue

            v_finite = rv[0] if rv[1] == -1 else rv[1]
            tangent = vor.points[p2] - vor.points[p1]
            normal = np.array([-tangent[1], tangent[0]])
            normal = normal / np.linalg.norm(normal)

            midpoint = 0.5 * (vor.points[p1] + vor.points[p2])
            if np.dot(normal, midpoint - center) < 0:
                normal = -normal

            extent = max(bbox[1] - bbox[0], bbox[3] - bbox[2]) * 2
            far_point = vor.vertices[v_finite] + normal * extent
            new_vertices.append(far_point)

        vs = np.array(new_vertices)
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        polygon = vs[np.argsort(angles)]
        polygon = _clip_polygon_to_bbox(polygon, bbox)
        regions_out.append(polygon)

    return regions_out


def plot_topology(results_dir: str = "results", base_dir: str = "."):
    """Figure 7: Topological structure of TSP instances (Voronoi + Delaunay)."""
    from algorithms.tsp import parse_tsplib

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Topological Structure of TSP Instances",
                 fontsize=14, fontweight="bold")

    for idx, (inst_name, inst_config) in enumerate(INSTANCES.items()):
        ax = axes[idx // 2][idx % 2]
        instance = parse_tsplib(os.path.join(base_dir, inst_config["file"]),
                                inst_config["optimal"])
        coords = instance.coords

        # Bounding box with 5% padding
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        pad = (maxs - mins) * 0.05
        bbox = (mins[0] - pad[0], maxs[0] + pad[0],
                mins[1] - pad[1], maxs[1] + pad[1])

        # Compute Voronoi and Delaunay
        vor = Voronoi(coords)
        tri = Delaunay(coords)

        # Get finite clipped polygons
        polygons = _voronoi_finite_polygons(vor, bbox)

        # Compute cell areas (shoelace formula) for coloring
        areas = []
        valid_polys = []
        for poly in polygons:
            if len(poly) >= 3:
                x, y = poly[:, 0], poly[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                areas.append(area)
                valid_polys.append(poly)
        areas = np.array(areas)

        # Colored Voronoi cells
        norm = PowerNorm(gamma=0.5, vmin=areas[areas > 0].min(), vmax=areas.max())
        pc = PolyCollection(valid_polys, array=areas, cmap="plasma", norm=norm,
                            edgecolors="white", linewidths=0.5, alpha=0.85)
        ax.add_collection(pc)

        # Delaunay edges (subtle)
        ax.triplot(coords[:, 0], coords[:, 1], tri.simplices,
                   color="#333333", linewidth=0.3, alpha=0.25)

        # City points
        ax.scatter(coords[:, 0], coords[:, 1], s=12, c="black", zorder=5)

        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
        ax.set_aspect("equal")
        ax.set_title(f"{inst_name} (n={instance.dimension}, optimal={inst_config['optimal']:,})")
        ax.set_xlabel("x coordinate")
        ax.set_ylabel("y coordinate")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig7_topology.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig7_topology.png")


def run_analysis(results_dir: str = "results"):
    """Run all analysis: load results, generate all 6 figures + statistical summary."""
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
    convergence_efficiency(df, conv, results_dir)
    plot_topology(results_dir)

    print(f"\nAll figures and analysis saved to {results_dir}/")
    return stats_df
