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

Extension 1 (added after the course):
    - christofides        : 1.5-approximation for metric instances
    - nn_two_opt          : nearest neighbour improved by 2-opt local search
    - nn_two_opt_or_opt   : alternating 2-opt and Or-opt local search
    - simulated_annealing : randomised 2-opt moves with a cooling schedule
"""

import itertools
import math
import random

import networkx as nx


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


# ---------- Extension 1: stronger heuristics (added after the course) ----------

def christofides(dist):
    """
    Christofides' 1.5-approximation for metric TSP:
      1. minimum spanning tree T
      2. O = vertices with odd degree in T (always an even number of them)
      3. minimum-weight perfect matching M on O
      4. T + M has all even degrees -> Eulerian circuit
      5. shortcut repeated vertices
    Uses NetworkX for the matching (step 3) and Euler circuit (step 4).
    """
    n = len(dist)
    if n <= 3:
        return list(range(n)), tour_length(list(range(n)), dist)

    adj = _prim_mst(dist)
    odd = [v for v in range(n) if len(adj[v]) % 2 == 1]

    g_odd = nx.Graph()
    for i, u in enumerate(odd):
        for v in odd[i + 1:]:
            g_odd.add_edge(u, v, weight=dist[u][v])
    matching = nx.min_weight_matching(g_odd, weight="weight")

    multi = nx.MultiGraph()
    multi.add_nodes_from(range(n))
    for u in range(n):
        for v in adj[u]:
            if u < v:
                multi.add_edge(u, v)
    multi.add_edges_from(matching)

    tour, seen = [], set()
    for u, _ in nx.eulerian_circuit(multi, source=0):
        if u not in seen:
            seen.add(u)
            tour.append(u)
    return tour, tour_length(tour, dist)


def two_opt_improve(tour, dist, stats=None):
    """
    2-opt local search: remove two edges (a,b),(c,d) and reconnect as
    (a,c),(b,d) by reversing the segment between them, whenever that is
    shorter. Repeat until no improving move exists (a 2-opt local optimum).
    Each pass is O(n^2).

    If a dict is passed as `stats`, the number of passes and improving
    moves is recorded in it (used by the complexity analysis).
    """
    tour = list(tour)
    n = len(tour)
    if n < 4:
        return tour
    improved = True
    while improved:
        improved = False
        if stats is not None:
            stats["passes"] = stats.get("passes", 0) + 1
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                if d == a:
                    continue
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < -1e-10:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    b = tour[i]
                    improved = True
                    if stats is not None:
                        stats["moves"] = stats.get("moves", 0) + 1
    return tour


def or_opt_improve(tour, dist, max_seg=3):
    """
    Or-opt local search: cut out a chain of 1..max_seg consecutive cities
    and reinsert it (possibly reversed) between two other cities, whenever
    that is shorter. Complements 2-opt, which cannot relocate short chains.
    """
    tour = list(tour)
    n = len(tour)
    if n < 5:
        return tour
    improved = True
    while improved:
        improved = False
        for seg_len in range(1, max_seg + 1):
            i = 0
            while i <= n - seg_len:
                seg = tour[i:i + seg_len]
                prev, nxt = tour[i - 1], tour[(i + seg_len) % n]
                if nxt in seg or prev in seg:
                    i += 1
                    continue
                removal_gain = (dist[prev][seg[0]] + dist[seg[-1]][nxt]
                                - dist[prev][nxt])
                rest = tour[:i] + tour[i + seg_len:]
                best_delta, best_pos, best_rev = -1e-10, None, False
                for k in range(len(rest)):
                    u, v = rest[k], rest[(k + 1) % len(rest)]
                    if u == prev and v == nxt:
                        continue  # same place it came from
                    fwd = dist[u][seg[0]] + dist[seg[-1]][v] - dist[u][v]
                    rev = dist[u][seg[-1]] + dist[seg[0]][v] - dist[u][v]
                    for cost, is_rev in ((fwd, False), (rev, True)):
                        delta = cost - removal_gain
                        if delta < best_delta:
                            best_delta, best_pos, best_rev = delta, k, is_rev
                if best_pos is not None:
                    piece = seg[::-1] if best_rev else seg
                    tour = rest[:best_pos + 1] + piece + rest[best_pos + 1:]
                    improved = True
                i += 1
    return tour


def nn_two_opt(dist):
    """Nearest-neighbour start, then 2-opt to a local optimum."""
    tour = two_opt_improve(nearest_neighbour(dist)[0], dist)
    return tour, tour_length(tour, dist)


def nn_two_opt_or_opt(dist):
    """Nearest-neighbour start, then alternate 2-opt and Or-opt until neither improves."""
    tour = nearest_neighbour(dist)[0]
    best = tour_length(tour, dist)
    while True:
        tour = or_opt_improve(two_opt_improve(tour, dist), dist)
        length = tour_length(tour, dist)
        if length >= best - 1e-10:
            break
        best = length
    return tour, length


def simulated_annealing(dist, iters_per_city=2000, seed=0):
    """
    Simulated annealing with random 2-opt moves, starting from nearest
    neighbour. Worse moves are accepted with probability exp(-delta/T),
    which lets the search escape local optima early on; T cools
    geometrically so it behaves like pure local search at the end.
    """
    n = len(dist)
    tour = nearest_neighbour(dist)[0]
    if n < 4:
        return tour, tour_length(tour, dist)

    rng = random.Random(seed)
    cur_len = tour_length(tour, dist)
    best, best_len = list(tour), cur_len

    iters = iters_per_city * n
    t_start = cur_len / n          # roughly the average edge length
    t_end = t_start * 1e-3
    alpha = (t_end / t_start) ** (1.0 / iters)
    temp = t_start

    for _ in range(iters):
        i = rng.randint(1, n - 2)
        j = rng.randint(i + 1, n - 1)
        a, b = tour[i - 1], tour[i]
        c, d = tour[j], tour[(j + 1) % n]
        if d != a:
            delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                tour[i:j + 1] = reversed(tour[i:j + 1])
                cur_len += delta
                if cur_len < best_len - 1e-10:
                    best, best_len = list(tour), cur_len
        temp *= alpha
    return best, tour_length(best, dist)


SOLVERS = {
    "Brute force": brute_force,
    "Held-Karp": held_karp,
    "Nearest neighbour": nearest_neighbour,
    "MST 2-approx": mst_approx,
    "Christofides": christofides,
    "NN + 2-opt": nn_two_opt,
    "NN + 2-opt + Or-opt": nn_two_opt_or_opt,
    "Simulated annealing": simulated_annealing,
}
EXACT = {"Brute force", "Held-Karp"}
COURSE_SOLVERS = {"Brute force", "Held-Karp", "Nearest neighbour", "MST 2-approx"}
