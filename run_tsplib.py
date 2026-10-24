"""
Extension 2: benchmark the heuristics on TSPLIB instances and measure
the true gap to each instance's published optimal tour length.

Usage:
    python run_tsplib.py             # all bundled instances
    python run_tsplib.py --sa-seeds 5
"""

import argparse
import csv
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark import COLOURS
from tsp_solvers import SOLVERS, EXACT, simulated_annealing
from tsplib import bundled_instances, load_instance

HEURISTICS = [name for name in SOLVERS if name not in EXACT]


def run(sa_seeds):
    rows = []
    for name in bundled_instances():
        _, dist, optimum = load_instance(name)
        n = len(dist)
        for solver_name in HEURISTICS:
            # Simulated annealing is randomised, so it is run with several seeds.
            seeds = range(sa_seeds) if solver_name == "Simulated annealing" else [None]
            for seed in seeds:
                start = time.perf_counter()
                if seed is None:
                    _, length = SOLVERS[solver_name](dist)
                else:
                    _, length = simulated_annealing(dist, seed=seed)
                secs = time.perf_counter() - start
                rows.append({"instance": name, "n": n, "solver": solver_name,
                             "seed": "" if seed is None else seed,
                             "seconds": secs, "length": length, "optimum": optimum,
                             "gap_pct": 100 * (length - optimum) / optimum})
        print(f"{name:<10} n={n:>3} done")
    return rows


def per_instance(rows, key):
    """Mean of `key` per (solver, instance), averaging over SA seeds."""
    out = {}
    for r in rows:
        out.setdefault(r["solver"], {}).setdefault(r["instance"], []).append(r[key])
    return {s: {i: statistics.mean(v) for i, v in d.items()} for s, d in out.items()}


def plot(rows, path="tsplib_gap.png"):
    order = list(dict.fromkeys(r["instance"] for r in rows))
    sizes = {r["instance"]: r["n"] for r in rows}
    gap = per_instance(rows, "gap_pct")

    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = range(len(order))
    for solver in HEURISTICS:
        ax.plot(x, [gap[solver][i] for i in order], marker="o", markersize=5,
                linewidth=1, label=solver, color=COLOURS[solver])
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{i}\n({sizes[i]})" for i in order], fontsize=8)
    ax.set_xlabel("TSPLIB instance (number of cities), smallest to largest")
    ax.set_ylabel("% above published optimum")
    ax.set_title("Heuristic tour quality on TSPLIB, measured against true optima")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary(rows):
    gap = per_instance(rows, "gap_pct")
    secs = per_instance(rows, "seconds")
    print("\n| Method | Mean gap | Worst gap | Best gap | Mean time (s) |")
    print("|---|---|---|---|---|")
    for solver in HEURISTICS:
        g = list(gap[solver].values())
        t = list(secs[solver].values())
        print(f"| {solver} | {statistics.mean(g):.1f}% | {max(g):.1f}% "
              f"| {min(g):.1f}% | {statistics.mean(t):.3f} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sa-seeds", type=int, default=3)
    args = p.parse_args()

    rows = run(args.sa_seeds)
    with open("tsplib_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    plot(rows)
    summary(rows)
    print("\nSaved tsplib_results.csv, tsplib_gap.png")
