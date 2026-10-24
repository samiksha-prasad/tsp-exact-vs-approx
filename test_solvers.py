"""Sanity checks: exact solvers agree, approximations respect their bounds."""

from tsp_solvers import (brute_force, held_karp, nearest_neighbour, mst_approx,
                         distance_matrix, random_euclidean_instance, tour_length)


def valid_tour(tour, n):
    return sorted(tour) == list(range(n))


def test_all():
    for n in range(2, 10):
        for seed in range(5):
            dist = distance_matrix(random_euclidean_instance(n, seed=seed))
            bt, bl = brute_force(dist)
            ht, hl = held_karp(dist)
            assert valid_tour(bt, n) and valid_tour(ht, n)
            assert abs(bl - hl) < 1e-9, (n, seed, bl, hl)
            assert abs(tour_length(ht, dist) - hl) < 1e-9
            for solver in (nearest_neighbour, mst_approx):
                t, l = solver(dist)
                assert valid_tour(t, n)
                assert l >= hl - 1e-9
            assert mst_approx(dist)[1] <= 2 * hl + 1e-9
    print("all tests passed")


if __name__ == "__main__":
    test_all()
