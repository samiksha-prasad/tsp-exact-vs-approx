"""
Reading TSPLIB instances (Extension 2).

TSPLIB (Reinelt, 1991) is the standard benchmark library for the TSP.
Each instance has a published optimal tour length, so solvers can be
measured against the true optimum instead of each other.

Only EUC_2D instances are supported. TSPLIB defines their distance as the
Euclidean distance rounded to the nearest integer, nint(sqrt(dx^2 + dy^2)),
and the published optima use that rounding, so we must use it too.
"""

import math
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tsplib")


def parse_tsplib(path):
    """Return (name, points) from a .tsp file with a NODE_COORD_SECTION."""
    name, edge_type, dimension, points = None, None, None, []
    in_coords = False
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "EOF":
                break
            if in_coords:
                _, x, y = line.split()[:3]
                points.append((float(x), float(y)))
                continue
            if line.startswith("NODE_COORD_SECTION"):
                in_coords = True
                continue
            if ":" in line:
                key, value = (part.strip() for part in line.split(":", 1))
                if key == "NAME":
                    name = value
                elif key == "EDGE_WEIGHT_TYPE":
                    edge_type = value
                elif key == "DIMENSION":
                    dimension = int(value)
    if edge_type != "EUC_2D":
        raise ValueError(f"{path}: only EUC_2D is supported, got {edge_type}")
    if dimension is not None and len(points) != dimension:
        raise ValueError(f"{path}: expected {dimension} nodes, read {len(points)}")
    return name, points


def tsplib_distance_matrix(points):
    """EUC_2D distances as TSPLIB defines them: rounded to the nearest integer."""
    n = len(points)
    return [[float(int(math.dist(points[i], points[j]) + 0.5)) for j in range(n)]
            for i in range(n)]


def load_optima(path=None):
    """Published optimal tour lengths, from the `solutions` file."""
    path = path or os.path.join(DATA_DIR, "solutions")
    optima = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                name, value = line.split(":")
                optima[name.strip()] = int(value.split()[0])
    return optima


def load_instance(name):
    """Return (points, distance matrix, published optimum) for a bundled instance."""
    _, points = parse_tsplib(os.path.join(DATA_DIR, f"{name}.tsp"))
    return points, tsplib_distance_matrix(points), load_optima()[name]


def bundled_instances():
    """Names of the bundled instances, smallest first."""
    names = [f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".tsp")]
    return sorted(names, key=lambda n: len(parse_tsplib(os.path.join(DATA_DIR, n + ".tsp"))[1]))
