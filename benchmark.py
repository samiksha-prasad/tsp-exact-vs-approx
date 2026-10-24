"""
Benchmark: how runtime grows with problem size, and how far the
approximate tours are from optimal.

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

from tsp_solvers import SOLVERS, held_karp, distance_matrix, random_euclidean_instance

# Largest n each solver is run on (brute force and Held-Karp blow up fast).
MAX_N = {
    "Brute force": 10,
    "Held-Karp": 15,
    "Nearest neighbour": 200,
    "MST 2-approx": 200,
}
SIZES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 50, 75, 100, 150, 200]


def run(trials):
    rows = []
    for n in SIZES:
        for t in range(trials):
            seed = 1000 * n + t
            dist = distance_matrix(random_euclidean_instance(n, seed=seed))
            optimal = held_karp(dist)[1] if n <= MAX_N["Held-Karp"] else None
            for name, solver in SOLVERS.items():
                if n > MAX_N[name]:
                    continue
                start = time.perf_counter()
                _, length = solver(dist)
                elapsed = time.perf_counter() - start
                gap = 100 * (length - optimal) / optimal if optimal else None
                rows.append({"solver": name, "n": n, "trial": t,
                             "seconds": elapsed, "length": length, "gap_pct": gap})
        print(f"n={n:>3} done")
    return rows


def summarise(rows, key):
    out = {}
    for r in rows:
        if r[key] is None:
            continue
        out.setdefault(r["solver"], {}).setdefault(r["n"], []).append(r[key])
    return {s: {n: statistics.mean(v) for n, v in sorted(d.items())} for s, d in out.items()}


def plot(rows):
    runtime = summarise(rows, "seconds")
    gap = summarise(rows, "gap_pct")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d in runtime.items():
        ax.plot(list(d), list(d.values()), marker="o", label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Number of cities (n)")
    ax.set_ylabel("Mean runtime (s, log scale)")
    ax.set_title("TSP runtime growth: exact vs approximate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("runtime_vs_n.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("Nearest neighbour", "MST 2-approx"):
        d = gap.get(name, {})
        ax.plot(list(d), list(d.values()), marker="o", label=name)
    ax.axhline(100, color="grey", linestyle="--", linewidth=1, label="2-approx bound (100%)")
    ax.set_xlabel("Number of cities (n)")
    ax.set_ylabel("Mean % above optimal")
    ax.set_title("Approximation quality vs Held-Karp optimum")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("approx_gap.png", dpi=150)


def print_table(rows):
    runtime = summarise(rows, "seconds")
    gap = summarise(rows, "gap_pct")
    print(f"\n{'solver':<18}{'n':>5}{'mean time (s)':>16}{'mean gap %':>12}")
    for name, d in runtime.items():
        for n, s in d.items():
            g = gap.get(name, {}).get(n)
            g_str = f"{g:.2f}" if g is not None else "-"
            print(f"{name:<18}{n:>5}{s:>16.6f}{g_str:>12}")


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
