"""Layered (Graphviz-`dot`-style) layout used by the SVG fallback."""

import math
import random

import numpy as np
import pytest

from grn_sim.visualization.layered_layout import _GAP_DUMMY, _GAP_MIXED, _GAP_NODE, layered_layout


def row_of(positions):
    """node -> layer index (rows are equally spaced, so y identifies the layer)."""
    ys = sorted({round(y, 6) for _, y in positions})
    return [ys.index(round(y, 6)) for _, y in positions]


def test_chain_is_layered_top_to_bottom():
    pos, paths = layered_layout(4, [(0, 1), (1, 2), (2, 3)])
    assert [y for _, y in pos] == sorted(y for _, y in pos) and len({y for _, y in pos}) == 4
    assert all(len(path) == 2 for path in paths.values())


def test_sources_sit_directly_above_their_targets_like_graphviz():
    # same shape as the 16-gene network: some regulators feed a deeper chain, others only the last node
    #   0 -> 1 -> 2 -> 3      4 -> 3      5 -> 3      6 -> 2
    pos, _ = layered_layout(7, [(0, 1), (1, 2), (2, 3), (4, 3), (5, 3), (6, 2)])
    layer = row_of(pos)
    assert layer[0] == 0 and layer[1] == 1 and layer[2] == 2 and layer[3] == 3
    assert layer[4] == layer[5] == 2, "sources of the last node belong right above it, not on the top row"
    assert layer[6] == 1, "a source of node 2 belongs one row above it"


def test_empty_isolated_and_single_node_graphs():
    assert layered_layout(0, []) == ([], {})
    pos, paths = layered_layout(3, [])
    assert len(pos) == 3 and not paths and len({y for _, y in pos}) == 1
    xs = sorted(x for x, _ in pos)
    assert all(b - a >= _GAP_NODE - 1e-6 for a, b in zip(xs, xs[1:])), "isolated nodes sit side by side"
    assert layered_layout(1, [(0, 0)])[1] == {}, "self loops are not part of the layout"


def test_crossing_is_removed():
    # 0 -> 3 and 1 -> 2 cross when the rows are ordered by node index
    pos, _ = layered_layout(4, [(0, 3), (1, 2)])
    left_top, right_top = sorted([0, 1], key=lambda v: pos[v][0])
    expected_left_bottom = 3 if left_top == 0 else 2
    bottom = sorted([2, 3], key=lambda v: pos[v][0])
    assert bottom[0] == expected_left_bottom


def test_long_edge_is_routed_around_the_nodes_in_between():
    pos, paths = layered_layout(3, [(0, 1), (1, 2), (0, 2)])
    detour = paths[(0, 2)]
    assert len(detour) == 3 and detour[0] == pos[0] and detour[-1] == pos[2]
    waypoint = detour[1]
    assert waypoint[1] == pos[1][1], "the waypoint lives in the middle row"
    assert abs(waypoint[0] - pos[1][0]) >= _GAP_MIXED - 1e-6, "and keeps clear of the real node in that row"


def test_layout_is_deterministic():
    edges = [(0, 1), (1, 2), (2, 0), (3, 1), (3, 4), (4, 3), (0, 4)]  # cycles and a reciprocal pair
    assert layered_layout(5, edges) == layered_layout(5, edges)


def test_reciprocal_edges_share_one_route():
    pos, paths = layered_layout(2, [(0, 1), (1, 0)])
    assert paths[(0, 1)] == paths[(1, 0)][::-1]
    assert pos[0][1] != pos[1][1]


def random_graph(rng, n, density, acyclic):
    edges = []
    for u in range(n):
        for v in range(n):
            if u != v and (not acyclic or u < v) and rng.random() < density:
                edges.append((u, v))
    return edges


@pytest.mark.parametrize("seed", range(40))
def test_random_dags_flow_downwards_without_overlaps(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 28)
    perm = list(range(n))
    rng.shuffle(perm)  # node indices must not already be a topological order
    edges = [(perm[u], perm[v]) for u, v in random_graph(rng, n, rng.choice([0.05, 0.12, 0.3]), acyclic=True)]
    pos, paths = layered_layout(n, edges)

    assert len(pos) == n and all(math.isfinite(c) for p in pos for c in p)
    for (s, d), path in paths.items():
        assert path[0] == pos[s] and path[-1] == pos[d]
        assert all(a[1] < b[1] for a, b in zip(path, path[1:])), f"edge {s}->{d} must flow downwards one layer at a time"
    by_row = {}
    for v, (x, y) in enumerate(pos):
        by_row.setdefault(y, []).append(x)
    for xs in by_row.values():
        xs.sort()
        assert all(b - a >= _GAP_NODE - 1e-6 for a, b in zip(xs, xs[1:])), "nodes in one row must not touch"


@pytest.mark.parametrize("seed", range(40))
def test_random_cyclic_graphs_are_laid_out_with_consistent_routes(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(2, 20)
    edges = random_graph(rng, n, rng.choice([0.08, 0.2, 0.4]), acyclic=False)
    pos, paths = layered_layout(n, edges)

    assert set(paths) == set(edges)
    assert all(math.isfinite(c) for p in pos for c in p)
    for (s, d), path in paths.items():
        assert path[0] == pos[s] and path[-1] == pos[d]
        steps = [b[1] - a[1] for a, b in zip(path, path[1:])]
        assert all(step > 0 for step in steps) or all(step < 0 for step in steps), "a route never zig-zags between rows"
    upward = [e for e, path in paths.items() if path[-1][1] < path[0][1]]
    assert len(upward) < len(edges) or not edges, "most edges still flow downwards"


def test_dummy_gaps_are_ordered():
    assert _GAP_DUMMY < _GAP_MIXED < _GAP_NODE


def test_wide_fan_in_stays_compact_and_centred():
    pos, _ = layered_layout(41, [(k, 40) for k in range(40)])
    target_x = pos[40][0]
    xs = [x for x, _ in pos[:40]]
    assert min(xs) < target_x < max(xs)
    assert abs(target_x - np.mean(xs)) < _GAP_NODE, "the common target sits roughly under the middle of its regulators"


def upward_edges(paths):
    return sum(1 for path in paths.values() if path[-1][1] < path[0][1])


def test_feedback_edges_are_kept_to_the_minimum_on_simple_cycles():
    _, three_cycle = layered_layout(3, [(0, 1), (1, 2), (2, 0)])
    assert upward_edges(three_cycle) == 1
    _, mutual = layered_layout(2, [(0, 1), (1, 0)])
    assert upward_edges(mutual) == 1
    _, two_mutual_pairs = layered_layout(4, [(0, 1), (1, 0), (2, 3), (3, 2), (1, 2)])
    assert upward_edges(two_mutual_pairs) == 2


def test_dense_regulatory_network_has_about_one_upward_edge_per_mutual_pair():
    # 9-gene network with 17 mutual pairs (the E. coli SOS circuit's shape): 17 upward edges is optimal
    rng = random.Random(7)
    n = 9
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    chosen = rng.sample(pairs, 17)
    edges = [e for a, b in chosen for e in ((a, b), (b, a))] + [(0, 8), (3, 5)]
    edges = list(dict.fromkeys(edges))
    mutual = sum(1 for (a, b) in edges if (b, a) in edges) // 2
    _, paths = layered_layout(n, edges)
    assert upward_edges(paths) <= mutual + 2


@pytest.mark.parametrize("seed", range(20))
def test_feedback_edges_stay_below_the_classic_bound(seed):
    rng = random.Random(500 + seed)
    n = rng.randint(4, 24)
    edges = random_graph(rng, n, rng.choice([0.1, 0.25, 0.4]), acyclic=False)
    _, paths = layered_layout(n, edges)
    assert upward_edges(paths) <= len(edges) / 2 + n / 2
