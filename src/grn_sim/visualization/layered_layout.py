"""
Layered ("hierarchical") graph layout in pure Python, the way Graphviz `dot` arranges a directed graph:
regulators on top, their targets in the rows below, edges flowing downwards.

Used by the SVG fallback in ``graph_export`` when the Graphviz binaries are not installed.
Classic Sugiyama pipeline:
1. cycle removal   - edges closing a cycle are reversed for layout purposes only (DFS back edges)
2. layer assignment - longest path, then nodes with spare slack are moved next to their neighbours so
                      edges stay short (sources sit directly above their targets, not on the top row)
3. long edges are cut into one dummy node per layer they cross
4. crossing reduction - barycentre sweeps, keeping the ordering with the fewest crossings
5. x coordinates   - iterative neighbour averaging that respects the minimum node separation
"""

from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]
Edge = Tuple[int, int]

# Minimum horizontal distance between neighbours in a layer: real-real, real-dummy, dummy-dummy
_GAP_NODE, _GAP_MIXED, _GAP_DUMMY = 72.0, 44.0, 26.0


def _break_cycles(n: int, edges: Sequence[Edge]) -> Dict[Edge, Edge]:
    """
    Maps every edge to the (upper, lower) pair used for layout, reversing as few edges as practical.

    Eades-Lin-Smyth heuristic: build a node ordering by repeatedly moving sinks to the back and sources to the
    front, and when neither exists, the node with the largest (out-degree - in-degree) to the front. Edges that
    run against that ordering are reversed. (Each mutual pair a<->b necessarily has one reversed edge.)
    """
    out_nbrs: Dict[int, set] = {v: set() for v in range(n)}
    in_nbrs: Dict[int, set] = {v: set() for v in range(n)}
    for s, d in edges:
        out_nbrs[s].add(d)
        in_nbrs[d].add(s)

    remaining = set(range(n))
    front: List[int] = []
    back: List[int] = []

    def remove(v: int) -> None:
        remaining.discard(v)
        for w in out_nbrs[v]:
            in_nbrs[w].discard(v)
        for w in in_nbrs[v]:
            out_nbrs[w].discard(v)

    while remaining:
        peeled = True
        while peeled:
            peeled = False
            for v in sorted(remaining):
                if v not in remaining:
                    continue
                if not out_nbrs[v]:  # sink
                    back.append(v)
                    remove(v)
                    peeled = True
                elif not in_nbrs[v]:  # source
                    front.append(v)
                    remove(v)
                    peeled = True
        if remaining:
            v = max(sorted(remaining), key=lambda w: len(out_nbrs[w]) - len(in_nbrs[w]))
            front.append(v)
            remove(v)

    rank = {v: i for i, v in enumerate(front + back[::-1])}
    return {(s, d): ((d, s) if rank[s] > rank[d] else (s, d)) for s, d in edges}


def _assign_layers(n: int, dag_edges: Sequence[Edge]) -> List[int]:
    preds: Dict[int, List[int]] = {v: [] for v in range(n)}
    succs: Dict[int, List[int]] = {v: [] for v in range(n)}
    for u, v in dag_edges:
        succs[u].append(v)
        preds[v].append(u)

    # longest path from the sources (Kahn's algorithm gives the topological order)
    indegree = {v: len(preds[v]) for v in range(n)}
    order = [v for v in range(n) if indegree[v] == 0]
    layer = [0] * n
    for u in order:
        for v in succs[u]:
            layer[v] = max(layer[v], layer[u] + 1)
            indegree[v] -= 1
            if indegree[v] == 0:
                order.append(v)

    # tighten: a node with more outgoing than incoming edges moves down as far as its successors
    # allow (and one with more incoming than outgoing moves up), shortening the total edge length
    for _ in range(50):
        moved = False
        for v in order:
            surplus = len(preds[v]) - len(succs[v])
            if surplus == 0:
                continue
            lo = max((layer[u] + 1 for u in preds[v]), default=None)
            hi = min((layer[w] - 1 for w in succs[v]), default=None)
            target = hi if surplus < 0 else lo
            if target is not None and target != layer[v]:
                layer[v] = target
                moved = True
        if not moved:
            break

    base = min(layer, default=0)
    return [value - base for value in layer]


def _crossings(upper: List[int], lower: List[int], links: Sequence[Edge]) -> int:
    pos_u = {v: i for i, v in enumerate(upper)}
    pos_l = {v: i for i, v in enumerate(lower)}
    pairs = sorted((pos_u[u], pos_l[v]) for u, v in links)
    return sum(1 for a in range(len(pairs)) for b in range(a + 1, len(pairs)) if pairs[a][0] < pairs[b][0] and pairs[a][1] > pairs[b][1])


def layered_layout(
    n: int,
    edges: Sequence[Edge],
    rank_sep: float = 72.0,
) -> Tuple[List[Point], Dict[Edge, List[Point]]]:
    """
    Lays out the directed graph with ``n`` nodes and ``edges`` (source, target; no self loops).

    Returns the node centres and, for every edge, the polyline from its source to its target
    (centres of both end nodes plus one waypoint per layer the edge crosses).
    """
    edges = [(s, d) for s, d in dict.fromkeys(edges) if s != d]
    oriented = _break_cycles(n, edges)
    dag_edges = list(dict.fromkeys(oriented.values()))
    layer_of = _assign_layers(n, dag_edges)

    # ---- dummy nodes for edges that span several layers --------------------------------------
    layer_of = list(layer_of)
    chains: Dict[Edge, List[int]] = {}
    links: List[Edge] = []  # consecutive-layer edges of the expanded graph
    for u, v in dag_edges:
        chain = [u]
        for level in range(layer_of[u] + 1, layer_of[v]):
            layer_of.append(level)
            chain.append(len(layer_of) - 1)
        chain.append(v)
        chains[(u, v)] = chain
        links.extend(zip(chain, chain[1:]))

    total = len(layer_of)
    depth = (max(layer_of) + 1) if layer_of else 0
    layers: List[List[int]] = [[] for _ in range(depth)]
    for v in range(total):
        layers[layer_of[v]].append(v)

    up: Dict[int, List[int]] = {v: [] for v in range(total)}
    down: Dict[int, List[int]] = {v: [] for v in range(total)}
    for u, v in links:
        down[u].append(v)
        up[v].append(u)

    # ---- crossing reduction: barycentre sweeps, keep the best ordering -----------------------
    def total_crossings() -> int:
        return sum(
            _crossings(layers[k], layers[k + 1], [(u, v) for u in layers[k] for v in down[u]])
            for k in range(depth - 1)
        )

    best_layers, best_cost = [list(row) for row in layers], total_crossings()
    for sweep in range(24):
        indices = range(1, depth) if sweep % 2 == 0 else range(depth - 2, -1, -1)
        for k in indices:
            ref = layers[k - 1] if sweep % 2 == 0 else layers[k + 1]
            ref_pos = {v: i for i, v in enumerate(ref)}
            neighbours = up if sweep % 2 == 0 else down
            current = {v: i for i, v in enumerate(layers[k])}
            layers[k].sort(
                key=lambda v: (sum(ref_pos[w] for w in neighbours[v]) / len(neighbours[v])) if neighbours[v] else current[v]
            )
        cost = total_crossings()
        if cost < best_cost:
            best_layers, best_cost = [list(row) for row in layers], cost
            if cost == 0:
                break
    layers = best_layers

    # ---- x coordinates: neighbour averaging with minimum separation ---------------------------
    def gap(a: int, b: int) -> float:
        dummies = (a >= n) + (b >= n)
        return (_GAP_NODE, _GAP_MIXED, _GAP_DUMMY)[dummies]

    x = [0.0] * total
    for row in layers:
        cursor = 0.0
        for i, v in enumerate(row):
            if i:
                cursor += gap(row[i - 1], v)
            x[v] = cursor

    def settle(row: List[int], neighbours: Dict[int, List[int]]) -> None:
        desired = [(sum(x[w] for w in neighbours[v]) / len(neighbours[v])) if neighbours[v] else x[v] for v in row]
        forward, backward = list(desired), list(desired)
        for i in range(1, len(row)):
            forward[i] = max(desired[i], forward[i - 1] + gap(row[i - 1], row[i]))
        for i in range(len(row) - 2, -1, -1):
            backward[i] = min(desired[i], backward[i + 1] - gap(row[i], row[i + 1]))
        for i, v in enumerate(row):
            x[v] = (forward[i] + backward[i]) / 2.0  # both sequences keep the gaps, so does their mean

    for sweep in range(16):
        if sweep % 2 == 0:
            for k in range(1, depth):
                settle(layers[k], up)
        else:
            for k in range(depth - 2, -1, -1):
                settle(layers[k], down)

    centres = [(x[v], layer_of[v] * rank_sep) for v in range(total)]

    # ---- edge polylines in the original direction --------------------------------------------
    paths: Dict[Edge, List[Point]] = {}
    for (s, d), (u, v) in oriented.items():
        points = [centres[w] for w in chains[(u, v)]]
        paths[(s, d)] = points if (u, v) == (s, d) else points[::-1]
    return centres[:n], paths


def bounding_box(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points] or [0.0]
    ys = [p[1] for p in points] or [0.0]
    return min(xs), min(ys), max(xs), max(ys)
