"""
Extension 4: a pretrained neural solver (POMO) against the classical methods.

Questions:
  1. On real TSPLIB maps, how close does POMO get to the published optimum,
     compared with the best classical heuristic?
  2. POMO models are trained on random instances of one fixed size (20, 50
     or 100 cities). How well does each one generalise to other sizes?
  3. Does running classical local search on POMO's tour fix what it gets wrong?

Usage (from the repository root, after cloning POMO, see the notebook):
    python neural/run_neural.py --pomo-root /path/to/POMO
    python neural/run_neural.py --pomo-root /path/to/POMO --quick   # smaller run
"""

import argparse
import csv
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pomo_wrapper import PomoSolver, CHECKPOINTS
from tsp_solvers import (distance_matrix, nn_two_opt_or_opt, random_euclidean_instance,
                         tour_length, two_opt_or_opt_from)
from tsplib import bundled_instances, load_instance

TRAIN_SIZES = sorted(CHECKPOINTS)              # 20, 50, 100
RANDOM_SIZES = [20, 50, 100, 200, 300, 500]
COLOURS = {20: "#9467bd", 50: "#17becf", 100: "#d62728",
           "classical": "#e377c2", "hybrid": "#2ca02c"}


def timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def tsplib_experiment(solvers):
    """Every model on every TSPLIB instance, plus POMO-100 followed by local search."""
    rows = []
    for name in bundled_instances():
        points, dist, optimum = load_instance(name)
        n = len(points)

        def add(method, length, secs):
            rows.append({"instance": name, "n": n, "method": method, "length": length,
                         "optimum": optimum, "gap_pct": 100 * (length - optimum) / optimum,
                         "seconds": secs})

        (_, length), secs = timed(nn_two_opt_or_opt, dist)
        add("NN + 2-opt + Or-opt", length, secs)

        for size in TRAIN_SIZES:
            tour, secs = solvers[size].solve(points)
            add(f"POMO-{size}", tour_length(tour, dist), secs)
            if size == 100:
                no_aug, secs_na = solvers[size].solve(points, augment=False)
                add("POMO-100 (no augmentation)", tour_length(no_aug, dist), secs_na)
                (_, polished), secs_ls = timed(two_opt_or_opt_from, tour, dist)
                add("POMO-100 + 2-opt + Or-opt", polished, secs + secs_ls)
        print(f"{name:<10} n={n:>3} done")
    return rows


def random_experiment(solvers, sizes, per_size):
    """
    Generalisation across sizes on random uniform instances. No optimum is
    known here, so each POMO tour is compared with NN + 2-opt + Or-opt on the
    same instance (negative = POMO is shorter).
    """
    rows = []
    for n in sizes:
        for t in range(per_size):
            points = random_euclidean_instance(n, seed=90000 + 100 * n + t)
            dist = distance_matrix(points)
            (_, ref), ref_secs = timed(nn_two_opt_or_opt, dist)
            for size in TRAIN_SIZES:
                tour, secs = solvers[size].solve(points)
                length = tour_length(tour, dist)
                rows.append({"n": n, "trial": t, "train_size": size, "length": length,
                             "reference": ref, "diff_pct": 100 * (length - ref) / ref,
                             "seconds": secs, "reference_seconds": ref_secs})
        print(f"random n={n:>3} done")
    return rows


def mean_by(rows, keys, value):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r[value])
    return {k: statistics.mean(v) for k, v in out.items()}


def plot_tsplib(rows, path):
    order = list(dict.fromkeys(r["instance"] for r in rows))
    sizes = {r["instance"]: r["n"] for r in rows}
    gap = mean_by(rows, ["method", "instance"], "gap_pct")
    series = [("NN + 2-opt + Or-opt", COLOURS["classical"], "best classical heuristic"),
              ("POMO-20", COLOURS[20], "POMO trained on 20 cities"),
              ("POMO-50", COLOURS[50], "POMO trained on 50 cities"),
              ("POMO-100", COLOURS[100], "POMO trained on 100 cities"),
              ("POMO-100 + 2-opt + Or-opt", COLOURS["hybrid"], "POMO-100, then 2-opt + Or-opt")]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = list(range(len(order)))
    for method, colour, label in series:
        ax.plot(x, [gap[(method, i)] for i in order], marker="o", markersize=5,
                linewidth=1.2, color=colour, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n({sizes[i]})" for i in order], fontsize=8)
    ax.set_xlabel("TSPLIB instance (number of cities), smallest to largest")
    ax.set_ylabel("% above published optimum")
    ax.set_title("Neural (POMO) vs classical local search on TSPLIB")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_random(rows, path):
    diff = mean_by(rows, ["train_size", "n"], "diff_pct")
    sizes = sorted({r["n"] for r in rows})

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for size in TRAIN_SIZES:
        ax.plot(sizes, [diff[(size, n)] for n in sizes], marker="o", markersize=5,
                linewidth=1.5, color=COLOURS[size], label=f"POMO trained on {size} cities")
        ax.plot([size], [diff[(size, size)]] if (size, size) in diff else [], marker="*",
                markersize=14, color=COLOURS[size], linestyle="none")
    ax.axhline(0, color="grey", linewidth=1, linestyle="--")
    ax.text(sizes[-1], 0, " NN + 2-opt + Or-opt", va="bottom", ha="right", fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("Number of cities in the test instance (log scale)")
    ax.set_ylabel("% longer than classical local search (below 0 = POMO better)")
    ax.set_title("Generalisation across sizes (★ = the size each model was trained on)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarise(tsplib_rows, random_rows):
    gap = mean_by(tsplib_rows, ["method"], "gap_pct")
    secs = mean_by(tsplib_rows, ["method"], "seconds")
    worst = {}
    for r in tsplib_rows:
        worst[r["method"]] = max(worst.get(r["method"], 0), r["gap_pct"])
    small = mean_by([r for r in tsplib_rows if r["n"] <= 105], ["method"], "gap_pct")
    large = mean_by([r for r in tsplib_rows if r["n"] > 105], ["method"], "gap_pct")

    print("\nTSPLIB (15 instances)")
    print("| Method | Mean gap | n ≤ 105 | n > 105 | Worst | Mean time (s) |")
    print("|---|---|---|---|---|---|")
    for m in ["NN + 2-opt + Or-opt", "POMO-20", "POMO-50", "POMO-100",
              "POMO-100 (no augmentation)", "POMO-100 + 2-opt + Or-opt"]:
        print(f"| {m} | {gap[(m,)]:.2f}% | {small[(m,)]:.2f}% | {large[(m,)]:.2f}% "
              f"| {worst[m]:.1f}% | {secs[(m,)]:.2f} |")

    diff = mean_by(random_rows, ["train_size", "n"], "diff_pct")
    sizes = sorted({r["n"] for r in random_rows})
    print("\nRandom instances: % difference from NN + 2-opt + Or-opt (negative = POMO shorter)")
    print("| Trained on | " + " | ".join(f"n={n}" for n in sizes) + " |")
    print("|---|" + "---|" * len(sizes))
    for size in TRAIN_SIZES:
        print(f"| {size} | " + " | ".join(f"{diff[(size, n)]:+.1f}%" for n in sizes) + " |")


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pomo-root", required=True, help="path to a clone of github.com/yd-kwon/POMO")
    p.add_argument("--per-size", type=int, default=5, help="random instances per size")
    p.add_argument("--quick", action="store_true", help="fewer random sizes, 2 per size")
    p.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    args = p.parse_args()

    solvers = {s: PomoSolver(args.pomo_root, s, args.device) for s in TRAIN_SIZES}
    print(f"POMO running on: {solvers[100].device}")

    sizes = [20, 50, 100, 200] if args.quick else RANDOM_SIZES
    per_size = 2 if args.quick else args.per_size

    tsplib_rows = tsplib_experiment(solvers)
    random_rows = random_experiment(solvers, sizes, per_size)

    save_csv(tsplib_rows, os.path.join(ROOT, "neural_tsplib_results.csv"))
    save_csv(random_rows, os.path.join(ROOT, "neural_random_results.csv"))
    plot_tsplib(tsplib_rows, os.path.join(ROOT, "neural_tsplib_gap.png"))
    plot_random(random_rows, os.path.join(ROOT, "neural_generalisation.png"))
    summarise(tsplib_rows, random_rows)
    print("\nSaved neural_tsplib_results.csv, neural_random_results.csv, "
          "neural_tsplib_gap.png, neural_generalisation.png")
