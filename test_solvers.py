"""Sanity checks: exact solvers agree, approximations respect their bounds."""

from tsp_solvers import (brute_force, held_karp, nearest_neighbour, mst_approx,
                         christofides, nn_two_opt, nn_two_opt_or_opt,
                         simulated_annealing, distance_matrix,
                         random_euclidean_instance, tour_length)

HEURISTICS = (nearest_neighbour, mst_approx, christofides,
              nn_two_opt, nn_two_opt_or_opt, simulated_annealing)


def valid_tour(tour, n):
    return sorted(tour) == list(range(n))


def test_small_instances():
    """Compare everything against the exact optimum for n <= 9."""
    for n in range(2, 10):
        for seed in range(5):
            dist = distance_matrix(random_euclidean_instance(n, seed=seed))
            bt, bl = brute_force(dist)
            ht, hl = held_karp(dist)
            assert valid_tour(bt, n) and valid_tour(ht, n)
            assert abs(bl - hl) < 1e-9, (n, seed, bl, hl)
            assert abs(tour_length(ht, dist) - hl) < 1e-9
            for solver in HEURISTICS:
                t, l = solver(dist)
                assert valid_tour(t, n), solver.__name__
                assert abs(tour_length(t, dist) - l) < 1e-6, solver.__name__
                assert l >= hl - 1e-9, solver.__name__
            assert mst_approx(dist)[1] <= 2 * hl + 1e-9
            assert christofides(dist)[1] <= 1.5 * hl + 1e-9


def test_local_search_never_worse():
    """Local search starts from nearest neighbour, so it can only improve it."""
    for n in (20, 50):
        for seed in range(3):
            dist = distance_matrix(random_euclidean_instance(n, seed=seed))
            nn = nearest_neighbour(dist)[1]
            two = nn_two_opt(dist)
            both = nn_two_opt_or_opt(dist)
            sa = simulated_annealing(dist)
            for t, _ in (two, both, sa):
                assert valid_tour(t, n)
            assert two[1] <= nn + 1e-9
            assert both[1] <= two[1] + 1e-9
            assert sa[1] <= nn + 1e-9


if __name__ == "__main__":
    test_small_instances()
    test_local_search_never_worse()
    print("all tests passed")
