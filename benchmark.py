"""
Benchmark: how runtime grows with problem size, and how far each
heuristic's tour is from the best available reference.

Reference tour length:
    n <= 15 : the exact optimum from Held-Karp
    n  > 15 : the best tour found by any solver on that instance
              (no optimum is available yet; TSPLIB instances come in stage 2)

Usage:
    python benchmark.py            # default sizes, 5 trials each
    python benchmark.py --trials 10
"""

import argparse
import csv
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsp_solvers import SOLVERS, EXACT, distance_matrix, random_euclidean_instance

# Largest n each solver is run on (brute force and Held-Karp blow up fast).
MAX_N = {name: 200 for name in SOLVERS}
MAX_N["Brute force"] = 10
MAX_N["Held-Karp"] = 15
OPTIMAL_UP_TO = MAX_N["Held-Karp"]

SIZES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 50, 75, 100, 150, 200]

COLOURS = {
    "Brute force": "#1f77b4",
    "Held-Karp": "#ff7f0e",
    "Nearest neighbour": "#2ca02c",
    "MST 2-approx": "#d62728",
    "Christofides": "#9467bd",
    "NN + 2-opt": "#8c564b",
    "NN + 2-opt + Or-opt": "#e377c2",
    "Simulated annealing": "#17becf",
}


def run(trials):
    rows = []
    for n in SIZES:
        for t in range(trials):
            seed = 1000 * n + t
            dist = distance_matrix(random_euclidean_instance(n, seed=seed))
            results = {}
            for name, solver in SOLVERS.items():
                if n > MAX_N[name]:
                    continue
                start = time.perf_counter()
                _, length = solver(dist)
                results[name] = (time.perf_counter() - start, length)

            if n <= OPTIMAL_UP_TO:
                ref, ref_type = results["Held-Karp"][1], "optimal"
            else:
                ref, ref_type = min(l for _, l in results.values()), "best_found"

            for name, (secs, length) in results.items():
                rows.append({"solver": name, "n": n, "trial": t, "seconds": secs,
                             "length": length,
                             "gap_pct": 100 * (length - ref) / ref,
                             "reference": ref_type})
        print(f"n={n:>3} done")
    return rows


def summarise(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r["solver"], {}).setdefault(r["n"], []).append(r[key])
    return {s: {n: statistics.mean(v) for n, v in sorted(d.items())} for s, d in out.items()}


def plot(rows):
    runtime = summarise(rows, "seconds")
    gap = summarise(rows, "gap_pct")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, d in runtime.items():
        ax.plot(list(d), list(d.values()), marker="o", markersize=4,
                label=name, color=COLOURS[name])
    ax.set_yscale("log")
    ax.set_xlabel("Number of cities (n)")
    ax.set_ylabel("Mean runtime (s, log scale)")
    ax.set_title("TSP runtime growth: exact vs approximate vs local search")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("runtime_vs_n.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, d in gap.items():
        if name in EXACT:
            continue
        ax.plot(list(d), list(d.values()), marker="o", markersize=4,
                label=name, color=COLOURS[name])
    ax.axvline(OPTIMAL_UP_TO + 0.5, color="grey", linestyle="--", linewidth=1)
    ax.text(OPTIMAL_UP_TO - 0.5, ax.get_ylim()[1] * 0.95, "vs optimum  ",
            ha="right", va="top", fontsize=9, color="grey")
    ax.text(OPTIMAL_UP_TO + 1.5, ax.get_ylim()[1] * 0.95, "vs best found",
            ha="left", va="top", fontsize=9, color="grey")
    ax.set_xscale("log")
    ax.set_xticks([4, 10, 15, 20, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of cities (n, log scale)")
    ax.set_ylabel("Mean % above reference tour")
    ax.set_title("Solution quality of heuristics")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("approx_gap.png", dpi=150)
    plt.close(fig)


def print_table(rows):
    runtime = summarise(rows, "seconds")
    gap = summarise(rows, "gap_pct")
    print(f"\n{'solver':<22}{'n':>5}{'mean time (s)':>16}{'mean gap %':>12}")
    for name, d in runtime.items():
        for n, s in d.items():
            print(f"{name:<22}{n:>5}{s:>16.6f}{gap[name][n]:>12.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=5)
    args = p.parse_args()

    rows = run(args.trials)
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    plot(rows)
    print_table(rows)
    print("\nSaved results.csv, runtime_vs_n.png, approx_gap.png")
