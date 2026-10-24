"""
Extension 3: empirical complexity analysis.

Measures each solver's runtime over a range of sizes and checks it
against the solver's theoretical growth rate.

  Polynomial methods: fit runtime = c * n^k on a log-log scale and
                      compare the fitted exponent k with theory.
  Exact methods:      fit runtime = c * f(n) for the theoretical f
                      (n! for brute force: (n-1)! tours, O(n) each; n^2 * 2^n for Held-Karp) and
                      check that runtime / f(n) stays roughly constant.

Usage:
    python complexity.py            # 5 trials per size (median is used)
    python complexity.py --trials 5
"""

import argparse
import csv
import gc
import math
import statistics
import time

import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import COLOURS
from tsp_solvers import (SOLVERS, held_karp, nearest_neighbour, two_opt_improve,
                         distance_matrix, random_euclidean_instance)

# Sizes are spaced evenly on a log scale so each part of the range
# has equal weight in the log-log fit.
POLY_SIZES = [50, 71, 100, 141, 200, 283, 400]
EXACT_SIZES = {
    "Brute force": list(range(5, 11)),
    "Held-Karp": list(range(6, 17)),
}

# Expected exponent k in runtime ~ n^k, with the reason.
THEORY = {
    "Nearest neighbour": (2, "n steps, each scanning up to n cities"),
    "MST 2-approx": (2, "Prim's algorithm on a dense graph"),
    "Christofides": (3, "minimum-weight perfect matching dominates"),
    "NN + 2-opt": (2, "O(n²) per pass, but the number of passes also grows with n"),
    "NN + 2-opt + Or-opt": (2, "as 2-opt, plus Or-opt passes of O(n²) each"),
    "Simulated annealing": (1, "2000·n moves; each costs O(1) to evaluate, "
                               "but an accepted move reverses a segment, up to O(n)"),
}
EXACT_MODEL = {
    "Brute force": (lambda n: math.factorial(n), "n!"),  # (n-1)! tours x O(n) each
    "Held-Karp": (lambda n: n * n * 2 ** n, "n² · 2ⁿ"),
}


def time_solver(solver, dist):
    gc.disable()
    start = time.perf_counter()
    solver(dist)
    elapsed = time.perf_counter() - start
    gc.enable()
    return elapsed


def measure(trials):
    rows = []
    plan = [(name, n) for name in THEORY for n in POLY_SIZES]
    plan += [(name, n) for name, sizes in EXACT_SIZES.items() for n in sizes]
    for name, n in plan:
        times = []
        for t in range(trials):
            dist = distance_matrix(random_euclidean_instance(n, seed=7000 + 100 * n + t))
            times.append(time_solver(SOLVERS[name], dist))
        rows.append({"solver": name, "n": n, "median_seconds": statistics.median(times),
                     "trials": trials})
        print(f"{name:<22} n={n:>3}  {statistics.median(times):.5f}s")
    return rows


def count_two_opt_work(trials):
    """Number of 2-opt passes and improving moves from a nearest-neighbour start."""
    rows = []
    for n in POLY_SIZES:
        passes, moves = [], []
        for t in range(trials):
            dist = distance_matrix(random_euclidean_instance(n, seed=7000 + 100 * n + t))
            stats = {}
            two_opt_improve(nearest_neighbour(dist)[0], dist, stats)
            passes.append(stats["passes"])
            moves.append(stats["moves"])
        rows.append({"n": n, "passes": statistics.median(passes),
                     "moves": statistics.median(moves)})
    return rows


def held_karp_memory(sizes=range(8, 16)):
    """Peak memory (bytes) of Held-Karp; theory says O(n * 2^n)."""
    rows = []
    for n in sizes:
        dist = distance_matrix(random_euclidean_instance(n, seed=n))
        tracemalloc.start()
        held_karp(dist)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        rows.append({"n": n, "peak_bytes": peak})
    return rows


def fit_power_law(ns, ts):
    """Least-squares fit of log t = k log n + log c. Returns (k, c, r^2)."""
    x, y = np.log(ns), np.log(ts)
    k, logc = np.polyfit(x, y, 1)
    pred = k * x + logc
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return k, math.exp(logc), r2


def fit_exact_model(ns, ts, f):
    """Fit t = c * f(n) in log space. Returns c, r^2 and the per-size ratios t / f(n)."""
    ratios = [t / f(n) for n, t in zip(ns, ts)]
    c = math.exp(statistics.mean(math.log(r) for r in ratios))
    y = np.log(ts)
    pred = np.log([c * f(n) for n in ns])
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return c, r2, ratios


def by_solver(rows):
    out = {}
    for r in rows:
        out.setdefault(r["solver"], ([], []))
        out[r["solver"]][0].append(r["n"])
        out[r["solver"]][1].append(r["median_seconds"])
    return out


def analyse_and_plot(rows, work, memory, path="complexity.png"):
    data = by_solver(rows)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.8))

    print("\nPolynomial methods: runtime ~ n^k")
    print("| Method | Fitted k | Expected k | R² | Why |")
    print("|---|---|---|---|---|")
    for name, (k_theory, why) in THEORY.items():
        ns, ts = data[name]
        k, c, r2 = fit_power_law(ns, ts)
        print(f"| {name} | {k:.2f} | {k_theory} | {r2:.3f} | {why} |")
        ax1.loglog(ns, ts, "o", color=COLOURS[name], markersize=5)
        xs = np.linspace(min(ns), max(ns), 50)
        ax1.loglog(xs, c * xs ** k, "-", color=COLOURS[name], linewidth=1.2,
                   label=f"{name}: k = {k:.2f} (theory {k_theory})")
    ax1.set_xlabel("Number of cities n (log scale)")
    ax1.set_ylabel("Median runtime in seconds (log scale)")
    ax1.set_title("Polynomial methods: fitted n^k")
    ax1.set_xticks(POLY_SIZES)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.minorticks_off()
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    print("\nExact methods: runtime ~ c · f(n)")
    print("| Method | Model | R² | Growth per added city (measured, last 3 sizes) | Predicted |")
    print("|---|---|---|---|---|")
    for name, (f, label) in EXACT_MODEL.items():
        ns, ts = data[name]
        c, r2, _ = fit_exact_model(ns, ts, f)
        measured = statistics.mean(ts[i + 1] / ts[i] for i in range(len(ts) - 3, len(ts) - 1))
        predicted = statistics.mean(f(ns[i + 1]) / f(ns[i]) for i in range(len(ns) - 3, len(ns) - 1))
        print(f"| {name} | {label} | {r2:.3f} | {measured:.2f}× | {predicted:.2f}× |")
        ax2.semilogy(ns, ts, "o", color=COLOURS[name], markersize=5)
        xs = list(range(min(ns), max(ns) + 1))
        ax2.semilogy(xs, [c * f(n) for n in xs], "-", color=COLOURS[name], linewidth=1.2,
                     label=f"{name}: c · {label}, R² = {r2:.3f}")
    ax2.set_xlabel("Number of cities n")
    ax2.set_ylabel("Median runtime in seconds (log scale)")
    ax2.set_title("Exact methods: measured vs theoretical model")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")

    print("\nWhy 2-opt grows faster than n^2: work done from a nearest-neighbour start")
    ns = [w["n"] for w in work]
    kp, cp, _ = fit_power_law(ns, [w["passes"] for w in work])
    km, cm, _ = fit_power_law(ns, [w["moves"] for w in work])
    print("| n | passes | improving moves |")
    print("|---|---|---|")
    for w in work:
        print(f"| {w['n']} | {w['passes']:.0f} | {w['moves']:.0f} |")
    print(f"passes ~ n^{kp:.2f}, improving moves ~ n^{km:.2f}; "
          f"predicted runtime exponent 2 + {kp:.2f} = {2 + kp:.2f}")
    k2 = fit_power_law(*data["NN + 2-opt"])[0]
    print(f"measured runtime exponent for NN + 2-opt: {k2:.2f}")

    ax3.loglog(ns, [w["passes"] for w in work], "o", color=COLOURS["NN + 2-opt"], markersize=5)
    ax3.loglog(ns, [w["moves"] for w in work], "s", color=COLOURS["Christofides"], markersize=5)
    xs = np.linspace(min(ns), max(ns), 50)
    ax3.loglog(xs, cp * xs ** kp, "-", color=COLOURS["NN + 2-opt"], linewidth=1.2,
               label=f"passes until no improvement: ~ n^{kp:.2f}")
    ax3.loglog(xs, cm * xs ** km, "-", color=COLOURS["Christofides"], linewidth=1.2,
               label=f"improving moves made: ~ n^{km:.2f}")
    ax3.set_xlabel("Number of cities n (log scale)")
    ax3.set_ylabel("Count (log scale)")
    ax3.set_title(f"2-opt: O(n²) per pass × n^{kp:.2f} passes ≈ n^{2 + kp:.2f}")
    ax3.set_xticks(POLY_SIZES)
    ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax3.minorticks_off()
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(fontsize=8, loc="upper left")

    print("\nHeld-Karp peak memory: theory O(n · 2^n)")
    print("| n | peak memory (MB) | growth vs previous | predicted 2(n)/(n-1) |")
    print("|---|---|---|---|")
    for prev, cur in zip([None] + memory[:-1], memory):
        n = cur["n"]
        growth = f"{cur['peak_bytes'] / prev['peak_bytes']:.2f}×" if prev else "-"
        predicted = f"{2 * n / (n - 1):.2f}×" if prev else "-"
        print(f"| {n} | {cur['peak_bytes'] / 1e6:.1f} | {growth} | {predicted} |")

    fig.suptitle("Measured runtime vs theoretical complexity (points: median of trials, lines: fits)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=5)
    args = p.parse_args()

    rows = measure(args.trials)
    with open("complexity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    work = count_two_opt_work(args.trials)
    memory = held_karp_memory()
    analyse_and_plot(rows, work, memory)
    print("\nSaved complexity.csv, complexity.png")
