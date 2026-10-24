"""
Travelling Salesperson Problem solvers: exact vs approximate methods.
DAA course project (rebuilt).

Exact:
    - brute_force      : tries every permutation, O(n!)
    - held_karp        : dynamic programming over subsets, O(n^2 * 2^n)
Approximate:
    - nearest_neighbour: greedy construction, O(n^2), no constant-factor guarantee
    - mst_approx       : preorder walk of a minimum spanning tree, O(n^2),
                         2-approximation for metric (e.g. Euclidean) instances
"""

import itertools
import math
import random


# ---------- instance helpers ----------

def random_euclidean_instance(n, seed=None, size=100.0):
    """Return n random points in a size x size square."""
    rng = random.Random(seed)
    return [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(n)]


def distance_matrix(points):
    n = len(points)
    return [[math.dist(points[i], points[j]) for j in range(n)] for i in range(n)]


def tour_length(tour, dist):
    """Length of a closed tour (returns to its start)."""
    return sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


# ---------- exact methods ----------

def brute_force(dist):
    """Fix city 0 as the start and try all (n-1)! orderings of the rest."""
    n = len(dist)
    if n <= 1:
        return list(range(n)), 0.0
    best_tour, best_len = None, math.inf
    for perm in itertools.permutations(range(1, n)):
        tour = (0,) + perm
        length = tour_length(tour, dist)
        if length < best_len:
            best_tour, best_len = list(tour), length
    return best_tour, best_len


def held_karp(dist):
    """
    Held-Karp DP. dp[(mask, j)] = shortest path that starts at 0,
    visits exactly the cities in `mask`, and ends at j.
    Time O(n^2 * 2^n), memory O(n * 2^n).
    """
    n = len(dist)
    if n <= 1:
        return list(range(n)), 0.0

    INF = math.inf
    size = 1 << n
    dp = [[INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    dp[1][0] = 0.0  # only city 0 visited, standing at city 0

    for mask in range(1, size):
        if not (mask & 1):  # every path must include the start city
            continue
        for j in range(n):
            if not (mask & (1 << j)) or dp[mask][j] == INF:
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                new_mask = mask | (1 << k)
                cand = dp[mask][j] + dist[j][k]
                if cand < dp[new_mask][k]:
                    dp[new_mask][k] = cand
                    parent[new_mask][k] = j

    full = size - 1
    best_len, last = INF, -1
    for j in range(1, n):
        cand = dp[full][j] + dist[j][0]
        if cand < best_len:
            best_len, last = cand, j

    # reconstruct the tour by walking parents backwards
    tour, mask, cur = [], full, last
    while cur != -1:
        tour.append(cur)
        prev = parent[mask][cur]
        mask ^= (1 << cur)
        cur = prev
    tour.reverse()
    return tour, best_len


# ---------- approximate methods ----------

def nearest_neighbour(dist, start=0):
    """Always move to the closest unvisited city."""
    n = len(dist)
    unvisited = set(range(n)) - {start}
    tour = [start]
    while unvisited:
        cur = tour[-1]
        nxt = min(unvisited, key=lambda c: dist[cur][c])
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour, tour_length(tour, dist)


def _prim_mst(dist):
    """Prim's algorithm on a dense graph, O(n^2). Returns adjacency lists."""
    n = len(dist)
    in_tree = [False] * n
    key = [math.inf] * n
    par = [-1] * n
    key[0] = 0.0
    for _ in range(n):
        u = min((v for v in range(n) if not in_tree[v]), key=lambda v: key[v])
        in_tree[u] = True
        for v in range(n):
            if not in_tree[v] and dist[u][v] < key[v]:
                key[v], par[v] = dist[u][v], u
    adj = [[] for _ in range(n)]
    for v in range(1, n):
        adj[par[v]].append(v)
        adj[v].append(par[v])
    return adj


def mst_approx(dist):
    """
    2-approximation for metric TSP: build an MST, walk it in preorder,
    and skip repeated cities (shortcutting never increases length under
    the triangle inequality).
    """
    n = len(dist)
    if n <= 1:
        return list(range(n)), 0.0
    adj = _prim_mst(dist)
    tour, seen, stack = [], set(), [0]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        tour.append(u)
        stack.extend(reversed(adj[u]))
    return tour, tour_length(tour, dist)


SOLVERS = {
    "Brute force": brute_force,
    "Held-Karp": held_karp,
    "Nearest neighbour": nearest_neighbour,
    "MST 2-approx": mst_approx,
}
EXACT = {"Brute force", "Held-Karp"}
