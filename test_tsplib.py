"""Checks for the TSPLIB loader (Extension 2)."""

from tsp_solvers import nearest_neighbour, nn_two_opt, tour_length
from tsplib import bundled_instances, load_instance, load_optima


def test_parser_and_rounding():
    points, dist, optimum = load_instance("berlin52")
    assert len(points) == 52
    assert points[0] == (565.0, 575.0)
    # nodes 1 and 2: sqrt(540^2 + 390^2) = 666.1..., rounded to 666
    assert dist[0][1] == 666.0
    assert optimum == 7542


def test_every_instance_has_an_optimum():
    optima = load_optima()
    for name in bundled_instances():
        assert name in optima, name


def test_no_tour_beats_the_optimum():
    """
    If the distance function didn't match TSPLIB's rounding, a heuristic
    could report a tour shorter than the published optimum. It never should.
    """
    for name in bundled_instances():
        _, dist, optimum = load_instance(name)
        for solver in (nearest_neighbour, nn_two_opt):
            tour, length = solver(dist)
            assert sorted(tour) == list(range(len(dist)))
            assert abs(tour_length(tour, dist) - length) < 1e-6
            assert length >= optimum, (name, solver.__name__, length, optimum)


if __name__ == "__main__":
    test_parser_and_rounding()
    test_every_instance_has_an_optimum()
    test_no_tour_beats_the_optimum()
    print("all TSPLIB tests passed")
