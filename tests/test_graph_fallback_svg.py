"""The pure-Python SVG fallback (used when the Graphviz `dot`/`neato` binaries are missing) must show edge direction."""

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from grn_sim.visualization import graph_export
from grn_sim.visualization.graph_export import NINE_STATE_STYLES, render_graphviz_svg

NS = "{http://www.w3.org/2000/svg}"
RX, RY = 26.0, 18.0


@pytest.fixture(autouse=True)
def no_graphviz(monkeypatch):
    monkeypatch.setattr(graph_export.shutil, "which", lambda *_: None)


def render(matrix, engine="dot", **kw):
    svg = render_graphviz_svg("digraph {}", engine=engine, matrix=np.array(matrix), **kw)
    root = ET.fromstring(svg)  # also proves the fallback is well-formed XML
    nodes = [(float(e.get("cx")), float(e.get("cy"))) for e in root.iter(NS + "ellipse")]
    heads = list(root.iter(NS + "polygon")) + [e for e in root.iter(NS + "line") if e.get("data-arrowhead")]
    shafts = [e for e in root.iter(NS + "line") if not e.get("data-arrowhead")] + list(root.iter(NS + "path"))
    return root, nodes, heads, shafts


def ellipse_distance(point, node):
    """1.0 means exactly on the node outline, <1 inside, >1 outside."""
    return math.sqrt(((point[0] - node[0]) / RX) ** 2 + ((point[1] - node[1]) / RY) ** 2)


def head_tip(head):
    if head.get("data-arrowhead") == "normal":
        x, y = head.get("points").split()[0].split(",")
        return float(x), float(y)
    # tee: the bar's centre, which sits ~2px short of the outline
    return ((float(head.get("x1")) + float(head.get("x2"))) / 2, (float(head.get("y1")) + float(head.get("y2"))) / 2)


@pytest.mark.parametrize("engine", ["dot", "neato"])
def test_every_edge_ends_in_an_arrow_head_on_its_target_node(engine):
    # matrix[i, j] != 0 : edge from regulator j to target i
    m = [[0, 0, 0, 0],
         [1, 0, 0, 0],   # 0 -> 1 activation
         [0, -1, 0, 0],  # 1 -> 2 inhibition
         [1, 0, 1, 0]]   # 0 -> 3 and 2 -> 3 activations
    _, nodes, heads, shafts = render(m, engine)
    assert len(nodes) == 4 and len(heads) == 4 and len(shafts) == 4

    found = {}  # target node -> kinds of the arrow heads landing on it
    for head in heads:
        tip = head_tip(head)
        owner = min(range(4), key=lambda n: abs(ellipse_distance(tip, nodes[n]) - 1.0))
        assert ellipse_distance(tip, nodes[owner]) == pytest.approx(1.0, abs=0.12)
        found.setdefault(owner, []).append(head.get("data-arrowhead"))
    assert sorted(found[1]) == ["normal"] and found[2] == ["tee"] and sorted(found[3]) == ["normal", "normal"]
    assert 0 not in found, "node 0 only regulates, so no arrow head may land on it"


def test_shafts_run_from_source_outline_to_the_arrow_head():
    _, nodes, heads, shafts = render([[0, 0], [1, 0]])  # edge 0 -> 1
    (shaft,), (head,) = shafts, heads
    start = (float(shaft.get("x1")), float(shaft.get("y1")))
    end = (float(shaft.get("x2")), float(shaft.get("y2")))
    assert ellipse_distance(start, nodes[0]) == pytest.approx(1.0, abs=0.02)
    tip = head_tip(head)
    assert math.dist(end, tip) == pytest.approx(11.0, abs=0.2)  # shaft stops at the triangle's base
    assert math.dist(start, tip) > math.dist(start, end)


def test_inhibition_uses_a_tee_bar_perpendicular_to_the_edge():
    _, nodes, heads, shafts = render([[0, 0], [-1, 0]])
    (head,) = heads
    assert head.tag == NS + "line" and head.get("data-arrowhead") == "tee"
    bar = (float(head.get("x2")) - float(head.get("x1")), float(head.get("y2")) - float(head.get("y1")))
    edge = (nodes[1][0] - nodes[0][0], nodes[1][1] - nodes[0][1])
    assert abs(bar[0] * edge[0] + bar[1] * edge[1]) < 1e-6 * math.hypot(*bar) * math.hypot(*edge) + 1e-6
    assert math.hypot(*bar) == pytest.approx(14.0, abs=0.2)


def test_reciprocal_edges_are_drawn_side_by_side_each_with_a_head():
    _, nodes, heads, shafts = render([[0, 1], [1, 0]])  # 0 -> 1 and 1 -> 0
    assert len(heads) == 2 and len(shafts) == 2
    assert math.dist(head_tip(heads[0]), head_tip(heads[1])) > 4.0

    # the two shafts are parallel to the node-to-node axis, 2 x 4.5 px apart
    ax, ay = nodes[1][0] - nodes[0][0], nodes[1][1] - nodes[0][1]
    norm = math.hypot(ax, ay)
    offsets = [((float(sh.get("x1")) - nodes[0][0]) * ay - (float(sh.get("y1")) - nodes[0][1]) * ax) / norm for sh in shafts]
    assert abs(offsets[0] - offsets[1]) == pytest.approx(9.0, abs=0.5)


@pytest.mark.parametrize("engine, side", [("dot", "right"), ("neato", "top")])
def test_self_regulation_loop_has_an_arrow_head_where_it_meets_the_node(engine, side):
    root, nodes, heads, shafts = render([[1, 0], [0, 0]], engine)
    (circle,) = list(root.iter(NS + "circle"))
    assert len(heads) == 1 and not shafts
    centre = (float(circle.get("cx")), float(circle.get("cy")))
    radius = float(circle.get("r"))
    node = nodes[0]

    # the loop sits on the free side: right of the node in layered layouts, above it in the circular one
    if side == "right":
        assert centre[0] > node[0] + RX - 1 and abs(centre[1] - node[1]) < 1
    else:
        assert centre[1] < node[1] - RY + 1 and abs(centre[0] - node[0]) < 1

    tip = head_tip(heads[0])
    assert math.dist(tip, centre) == pytest.approx(radius, abs=0.6), "head must sit on the loop"
    assert ellipse_distance(tip, node) == pytest.approx(1.0, abs=0.12), "head must sit on the node outline"


@pytest.mark.parametrize("state", sorted(NINE_STATE_STYLES))
def test_nine_state_edges_use_their_own_arrow_head_style(state):
    _, _, heads, _ = render([[0, 0], [state, 0]], is_9state=True)
    assert [h.get("data-arrowhead") for h in heads] == [NINE_STATE_STYLES[state]["arrowhead"]]


def test_event_handlers_are_attached_to_shaft_and_head():
    root, _, heads, shafts = render([[0, 0], [1, 0]])
    handler = "window.inspectEdge && window.inspectEdge(0,1)"
    assert shafts[0].get("onclick") == handler and heads[0].get("onclick") == handler


def test_layered_svg_declares_its_natural_size_while_the_circular_one_fits_its_container():
    matrix = np.zeros((6, 6), dtype=int)
    for k in range(5):
        matrix[k + 1, k] = 1  # a chain: tall layered layout
    layered = ET.fromstring(render_graphviz_svg("", engine="dot", matrix=matrix))
    vb = [float(v) for v in layered.get("viewBox").split()]
    assert (float(layered.get("width")), float(layered.get("height"))) == (round(vb[2]), round(vb[3]))
    assert vb[3] > vb[2], "a 6-level chain is taller than it is wide"
    assert "max-width:100%" in layered.get("style")

    circular = ET.fromstring(render_graphviz_svg("", engine="neato", matrix=matrix))
    assert (circular.get("width"), circular.get("height")) == ("100%", "100%")
