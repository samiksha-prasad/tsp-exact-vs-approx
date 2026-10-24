# TSP: Exact vs Approximate Algorithms

Design and Analysis of Algorithms course project. It compares exact and approximate solvers for the Travelling Salesperson Problem and measures how their runtime grows with problem size.

The original course code was lost, so this repository is a rebuild of that project. Extensions added later will be listed separately below.

## Algorithms

| Method | Type | Time complexity | Guarantee |
|---|---|---|---|
| Brute force | Exact | O(n!) | Optimal |
| Held–Karp (DP over subsets) | Exact | O(n²·2ⁿ), memory O(n·2ⁿ) | Optimal |
| Nearest neighbour | Greedy | O(n²) | None (can be arbitrarily bad) |
| MST preorder walk | Approximation | O(n²) | ≤ 2 × optimal on metric instances |

## Run

```bash
pip install -r requirements.txt
python test_solvers.py         # correctness checks
python benchmark.py --trials 5 # writes results.csv and two plots
```

## Results (random Euclidean instances, mean of 3 trials)

![Runtime vs n](runtime_vs_n.png)

- **Brute force** grows factorially. It takes about 0.37 s at n = 10, and each extra city multiplies the time by roughly n.
- **Held–Karp** grows by about 2.5× per added city, close to the expected ~2× from the 2ⁿ term. It reaches n = 15 in about 0.23 s, a size brute force could not finish in reasonable time.
- **Both approximations** stay in the millisecond range at n = 200 and grow roughly quadratically.

![Approximation gap](approx_gap.png)

- On instances where Held–Karp gives the optimum (n ≤ 15), both heuristics land about 5–30% above optimal. That is well inside the MST method's proven factor-2 bound.
- The tests confirm that brute force and Held–Karp agree on every instance, and that the MST tour never exceeds twice the optimum.

## Files

- `tsp_solvers.py`: the four solvers and instance helpers
- `benchmark.py`: timing, optimality gap, CSV output and plots
- `test_solvers.py`: correctness tests

## Planned extensions

- TSPLIB benchmarks against published optimal tours
- Christofides (1.5-approximation), 2-opt / Or-opt local search, simulated annealing
- Empirical complexity fitting (log-log slopes)
- Comparison with a learned (neural) TSP solver, including generalisation to larger n
