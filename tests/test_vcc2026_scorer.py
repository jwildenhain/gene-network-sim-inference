"""
The Virtual Cell Challenge 2026 scorer (grn_sim.evaluation.vcc2026) and the prediction adapter (vcc_prediction).

The scorer is a port of cell-eval2's ``vcc2026`` profile; cell-eval2 itself needs Python 3.11 and cannot run next to this
package, so the port is checked against (1) the two libraries it can be compared with here (scanpy's tie-corrected Wilcoxon
test, scipy's Benjamini-Hochberg), (2) the fixed points the metric specification states (a control paste, an exact paste,
the two anchors) and (3) hand-built differential-expression tables for the direction and set members.
"""

import numpy as np
import pytest

from grn_sim.evaluation import vcc2026 as v
from grn_sim.evaluation import vcc_prediction as vp


# -- fixtures --------------------------------------------------------------------------------------------------------------

def synthetic_panel(seed=1, G=40, P=12, n=60, n_ctrl=80):
    """A panel whose perturbation ``G<p>`` knocks down gene p and moves six other genes; ``(panel, folds)``."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(G)]
    base = np.exp(rng.normal(3, 1, G))
    blocks, labels, folds = [], [], {}
    blocks.append(rng.poisson(base[None, :] * np.exp(rng.normal(0, 0.2, (n_ctrl, 1))), (n_ctrl, G)))
    labels += [v.CONTROL] * n_ctrl
    for p in range(P):
        f = np.ones(G)
        f[p] = 0.2
        idx = rng.choice([g for g in range(G) if g != p], 6, replace=False)
        f[idx] = np.exp(rng.normal(0, 0.8, 6))
        folds[genes[p]] = f
        blocks.append(rng.poisson(base[None, :] * f[None, :] * np.exp(rng.normal(0, 0.2, (n, 1))), (n, G)))
        labels += [genes[p]] * n
    return v.Panel(np.vstack(blocks).astype(float), np.array(labels), genes), folds


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel()[0]


@pytest.fixture(scope="module")
def scorer(panel):
    return v.Scorer(panel)


def hand_tables(G, real_sig, real_lfc, pred_q, pred_p, pred_lfc, label="P"):
    """A ``(pred, real)`` pair of prepared sides from explicit DE arrays for one perturbation (no own-gene exclusion)."""
    genes = [f"g{i}" for i in range(G)]
    pan = v.Panel(np.ones((2, G)), np.array([v.CONTROL, label]), genes)

    def side(lfc, q, p):
        table = v.DETable(kept=np.ones(G, dtype=bool))
        table.lfc[label], table.p_adj[label], table.p_value[label] = np.asarray(lfc, float), np.asarray(q, float), np.asarray(p, float)
        return v.Prepared(pan, [label], {}, {}, {}, table, {}, np.ones((1, G)))

    real_q = np.where(real_sig, 0.001, 0.9)
    return side(pred_lfc, pred_q, pred_p), side(real_lfc, real_q, real_q)


# -- data preparation ------------------------------------------------------------------------------------------------------

def test_wilcoxon_matches_scanpy_tie_corrected_test():
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    import pandas as pd

    rng = np.random.default_rng(3)
    n, m, G = 50, 70, 120
    A = rng.poisson(rng.gamma(2, 3, G)[None, :] * np.exp(rng.normal(0, 0.3, (n, 1))), (n, G)).astype(float)
    B = rng.poisson(rng.gamma(2, 3, G)[None, :], (m, G)).astype(float)
    A[:, :5] = 0
    B[:, :5] = 0                                                       # constant genes: p = 1 on both sides
    ours = v.wilcoxon_pvalues(A, B)
    adata = ad.AnnData(X=np.vstack([A, B]), obs=pd.DataFrame({"g": ["a"] * n + ["ref"] * m}, index=[str(i) for i in range(n + m)]))
    adata.var_names = [f"g{i}" for i in range(G)]
    sc.tl.rank_genes_groups(adata, groupby="g", groups=["a"], reference="ref", method="wilcoxon", tie_correct=True, n_genes=G)
    res = adata.uns["rank_genes_groups"]
    order = np.array([int(x[1:]) for x in np.asarray(res["names"]["a"], dtype=str)])
    assert np.allclose(ours[order], np.asarray(res["pvals"]["a"], dtype=float), atol=1e-12)
    assert np.all(ours[:5] == 1.0)


def test_benjamini_hochberg_matches_scipy_and_keeps_nan():
    stats = pytest.importorskip("scipy.stats")
    fdc = getattr(stats, "false_discovery_control", None)
    p = np.random.default_rng(0).uniform(size=200) ** 3
    q = v.benjamini_hochberg(np.concatenate([p, [np.nan]]))
    assert np.isnan(q[-1]) and np.all((q[:-1] >= p) & (q[:-1] <= 1))
    if fdc is not None:
        assert np.allclose(q[:-1], fdc(p, method="bh"))


def test_jackknife_identity_half_split_distance_equals_sum_of_corrections():
    rng = np.random.default_rng(5)
    G, n = 50, 300
    mu = np.exp(rng.normal(3, 1, G))
    lib = np.exp(rng.normal(np.log(20000), 0.35, n))
    comp = mu * np.exp(rng.normal(0, 0.15, (n, G)))
    Y = rng.poisson(lib[:, None] * comp / comp.sum(axis=1, keepdims=True)).astype(float)
    d, c = [], []
    for _ in range(40):
        perm = rng.permutation(n)
        A, B = Y[perm[: n // 2]], Y[perm[n // 2:]]
        d.append(np.sum((v.bulk_profile(A) - v.bulk_profile(B)) ** 2))
        c.append(v.jackknife_correction(A) + v.jackknife_correction(B))
    assert abs(np.mean(d) / np.mean(c) - 1.0) < 0.1
    assert v.jackknife_correction(Y[:1]) == 0.0
    assert np.isclose(v.jackknife_per_gene(Y).sum(), v.jackknife_correction(Y))


def test_cpm_gate_reads_the_control_alone_and_bh_runs_per_perturbation(panel):
    prep = v.prepare(panel)
    ctrl = v.cpm(panel.counts[panel.rows(v.CONTROL)]).mean(axis=0)
    assert np.array_equal(prep.de.kept, ctrl > v.CPM_MIN)
    for p in prep.perturbations:
        q = prep.de.p_adj[p]
        assert np.all(np.isnan(q[~prep.de.kept])) and np.all(np.isfinite(q[prep.de.kept]))


# -- the metrics at their documented fixed points ------------------------------------------------------------------------

def test_control_paste_sits_at_every_no_skill_point(panel, scorer):
    ctrl = panel.counts[panel.rows(v.CONTROL)]
    blocks, labels = [], []
    for p in panel.perturbations:
        blocks.append(ctrl)                                            # the reference control cells, pasted whole
        labels += [p] * ctrl.shape[0]
    res = v.score_submission(scorer.prepared, v.Panel(np.vstack(blocks), np.array(labels), panel.genes))
    agg, per = res["aggregate"], res["per_perturbation"]
    assert all(x == 0.5 for x in per["pds"].values())                 # zero-norm effects tie the whole row: mid-rank
    assert abs(agg["mse"] - 1.0) < 0.15                                # the no-skill point, up to the jackknife's bias
    assert abs(agg["nmae"] - 1.0) < 0.05                               # a zero fold change makes numerator = denominator
    assert agg["fid"] == 0.0 or np.isnan(agg["fid"])                   # nothing called: silence scores 0
    assert agg["jac"] <= 0.05


def test_exact_paste_reaches_every_perfect_point(panel, scorer):
    res = v.score_submission(scorer.prepared, panel.subset(panel.perturbations, keep_control=False))
    agg, per = res["aggregate"], res["per_perturbation"]
    assert all(x == 1.0 for x in per["pds"].values())
    assert agg["mse"] <= 0.05
    assert agg["jac"] == 1.0 and agg["nmae"] == 0.0 and agg["fid"] == 1.0 and agg["reach"] == 1.0
    scaled = scorer.score(panel.subset(panel.perturbations, keep_control=False))
    assert scaled["members"]["mse"]["scaled"] == 1.0                   # clamped at the replicate
    assert scaled["members"]["reach"]["scaled"] >= 1.0                 # a perfect raw value scores at least 1 (spec section 4)
    assert scaled["overall"] > 1.0


def test_anchors_score_zero_and_one_by_construction(panel, scorer):
    base = scorer.score(v.baseline_prediction(scorer.prepared), allow_fractional=True)
    for m, d in base["members"].items():
        if d["scaled"] is not None:
            assert abs(d["scaled"]) < 1e-9, m
    a = scorer.anchors()
    assert a["n_perturbations"] == 12 and a["baseline"]["pds"] is not None
    assert set(a["replicate_splits"][0]) == set(v.METRICS) and len(a["replicate_splits"]) == v.N_SPLITS
    assert a["replicate"]["mse"] < a["baseline"]["mse"] and a["replicate"]["pds"] > a["baseline"]["pds"]


def test_true_fold_changes_score_between_baseline_and_replicate_on_most_members():
    panel, folds = synthetic_panel(seed=2)
    scorer = v.Scorer(panel)
    ctrl = panel.counts[panel.rows(v.CONTROL)]
    rng = np.random.default_rng(9)
    blocks, labels = [], []
    for p in panel.perturbations:
        blocks.append(v.emit_cells(ctrl, folds[p], 60, rng))
        labels += [p] * 60
    res = scorer.score(v.Panel(np.vstack(blocks), np.array(labels), panel.genes))
    assert res["overall"] > 0.5
    assert res["members"]["pds"]["scaled"] > 0.8


# -- the direction, set and fold-change members on hand-built tables -------------------------------------------------------

def test_reach_counts_the_deepest_pure_prefix_including_a_recovered_dip():
    G = 30
    real_sig = np.zeros(G, dtype=bool)
    real_sig[:20] = True
    real_lfc = np.ones(G)
    pred_lfc = np.ones(G)
    pred_lfc[0] = -1.0                                                 # a leading miss, then nine matches: purity 9/10 = 0.9
    pred_q = np.linspace(0.001, 0.02, G)                               # the prediction's confidence orders the genes
    pred, real = hand_tables(G, real_sig, real_lfc, pred_q, pred_q, pred_lfc)
    assert v.direction_reach(pred, real)["P"] == pytest.approx(20 / 20)   # purity stays >= 0.9 to the end: k* = 20
    pred_lfc[0] = 1.0
    pred_lfc[5:8] = -1.0                                               # three misses at ranks 6-8: purity falls below 0.9 ...
    pred, real = hand_tables(G, real_sig, real_lfc, pred_q, pred_q, pred_lfc)
    assert v.direction_reach(pred, real)["P"] == pytest.approx(5 / 20)    # ... and never recovers within the budget of 20
    real_sig[:] = False
    pred, real = hand_tables(G, real_sig, real_lfc, pred_q, pred_q, pred_lfc)
    assert np.isnan(v.direction_reach(pred, real)["P"])                   # an empty budget is dropped from the mean


def test_fidelity_is_matches_over_the_larger_call_set_and_drops_zero_over_zero():
    G = 20
    real_sig = np.zeros(G, dtype=bool)
    real_sig[:10] = True
    real_lfc = np.ones(G)
    pred_lfc = np.ones(G)
    pred_lfc[:2] = -1.0
    pred_q = np.full(G, 0.5)
    pred_q[:5] = 0.01                                                  # five calls, two wrong
    pred, real = hand_tables(G, real_sig, real_lfc, pred_q, pred_q, pred_lfc)
    assert v.direction_fidelity(pred, real)["P"] == pytest.approx(3 / 10)   # k = 3 over max(5, 10)
    assert v.significance_jaccard(pred, real)["P"] == pytest.approx(5 / 10)
    pred_q[:] = 0.5
    pred, real = hand_tables(G, ~real_sig & False, real_lfc, pred_q, pred_q, pred_lfc)
    assert np.isnan(v.direction_fidelity(pred, real)["P"])
    assert v.significance_jaccard(pred, real)["P"] == 1.0              # an empty union is agreement


def test_nmae_gate_needs_ten_genes_and_reads_a_missing_prediction_as_zero():
    G = 30
    real_sig = np.zeros(G, dtype=bool)
    real_sig[:12] = True
    real_lfc = np.linspace(0.5, 2.0, G)
    pred_lfc = np.full(G, np.nan)                                      # no usable prediction anywhere
    pred, real = hand_tables(G, real_sig, real_lfc, np.ones(G), np.ones(G), pred_lfc)
    assert v.fold_change_nmae(pred, real)["P"] == pytest.approx(1.0)      # zero fold change: numerator == denominator
    real_sig[:] = False
    real_sig[:9] = True
    pred, real = hand_tables(G, real_sig, real_lfc, np.ones(G), np.ones(G), pred_lfc)
    assert "P" not in v.fold_change_nmae(pred, real)                   # nine genes: omitted, for every submission alike


def test_own_gene_is_excluded_from_the_set_members(panel):
    prep = v.prepare(panel)
    p = prep.perturbations[0]
    col = prep.target_index[p]
    assert panel.genes[col] == p
    sig = prep.de.significant(p)
    assert sig[col], "the knocked-down gene itself is called"
    res = v.score_submission(prep, panel.subset(panel.perturbations, keep_control=False))
    assert res["per_perturbation"]["jac"][p] == 1.0                    # identical sets with the own gene removed on both sides


# -- scaling and the competition score ----------------------------------------------------------------------------------

def test_scale_member_clamps_and_floors():
    assert v.scale_member("pds", 0.75, 0.5, 1.0) == (0.5, "")
    assert v.scale_member("mse", -0.3, 0.9, 0.03)[0] == 1.0            # clamped at 1
    assert v.scale_member("mse", 5.0, 0.9, 0.03)[0] == 0.0             # clamped at 0
    assert v.scale_member("nmae", 40.0, 1.0, 0.4)[0] == -6.0           # floored
    assert v.scale_member("nmae", 2.0, 1.0, 0.4)[0] == pytest.approx(-1.0 / 0.6)
    assert v.scale_member("jac", float("nan"), 0.1, 0.5)[0] == pytest.approx(-0.25)   # a missing value takes the raw floor 0
    s, reason = v.scale_member("fid", 0.7, 0.6, 0.6)
    assert s is None and "degenerate" in reason


def test_resolved_members_flags_anchors_within_noise_and_an_unresolved_expression_error():
    b = {m: 0.5 for m in v.METRICS}
    r = {m: 0.9 for m in v.METRICS}
    sd = {m: 0.05 for m in v.METRICS}
    sd["reach"] = 0.3
    out = v.resolved_members(b, r, sd, {"sum_den": 1.0, "sum_noise": 0.1})
    assert out["pds"] == "" and "not resolved" in out["reach"]
    out = v.resolved_members(b, r, sd, {"sum_den": 0.1, "sum_noise": 0.2})
    assert "sampling noise" in out["mse"]
    b.update(mse=0.9, nmae=1.0)                                        # the two error members run the other way
    r.update(mse=0.03, nmae=0.4)
    scored = {"aggregate": {m: 0.7 for m in v.METRICS}, "cohort": {m: 5 for m in v.METRICS}}
    res = v.competition_score(scored, b, r, {"reach": "x"})
    assert res["n_scaled"] == 5 and res["members"]["reach"]["scaled"] is None and res["members"]["reach"]["raw"] == 0.7
    expected = [v.scale_member(m, 0.7, b[m], r[m])[0] for m in v.METRICS if m != "reach"]
    assert res["overall"] == pytest.approx(np.mean(expected))


def test_validate_submission_enforces_the_upload_contract(panel):
    pred = panel.subset(panel.perturbations, keep_control=False)
    v.validate_submission(pred, panel)
    with pytest.raises(ValueError, match="fractional"):
        v.validate_submission(v.Panel(pred.counts + 0.5, pred.labels, pred.genes), panel)
    with pytest.raises(ValueError, match="perturbation sets differ"):
        v.validate_submission(v.Panel(pred.counts[:10], pred.labels[:10], pred.genes), panel)
    big = pred.counts.copy()
    big[0, 0] = 2e6
    with pytest.raises(ValueError, match="more than"):
        v.validate_submission(v.Panel(big, pred.labels, pred.genes), panel)


def test_emit_cells_keeps_the_library_size_and_the_expected_fold(panel):
    ctrl = panel.counts[panel.rows(v.CONTROL)]
    fold = np.ones(panel.n_genes)
    fold[3] = 3.0
    X = v.emit_cells(ctrl, fold, 4000, np.random.default_rng(0))
    assert np.all(X == np.round(X)) and np.all(X >= 0)
    assert abs(X.sum(axis=1).mean() / ctrl.sum(axis=1).mean() - 1.0) < 0.05
    ratio = X.mean(axis=0) / ctrl.mean(axis=0)
    assert ratio[3] / np.median(np.delete(ratio, 3)) == pytest.approx(3.0, rel=0.1)
    with pytest.raises(ValueError):
        v.emit_cells(ctrl, fold[:-1], 5, np.random.default_rng(0))


# -- the prediction adapter ---------------------------------------------------------------------------------------------

def test_hold_out_takes_a_quarter_but_at_least_two_and_never_everything():
    assert len(vp.hold_out([f"g{i}" for i in range(40)], 1)) == 10
    assert len(vp.hold_out(["a", "b", "c"], 1)) == 2
    assert vp.hold_out(list("abcdefgh"), 4) == vp.hold_out(list("hgfedcba"), 4)
    with pytest.raises(ValueError):
        vp.hold_out(["a", "b"], 1)


def test_structure_response_recovers_a_chain_and_predicts_the_held_out_knockdown():
    # A chain 0 -> 1 -> 2 -> 3 -> 4 (activation): the knockdowns of 0-3 are seen, gene 4's own knockdown is held out.
    n = 5
    W = np.zeros((n, n))
    for j, w in enumerate((0.8, 0.6, 0.7, 0.5)):
        W[j, j + 1] = w
    delta = -1.5
    seen = [0, 1, 2, 3]
    X = np.column_stack([np.linalg.solve(np.eye(n) - W.T, delta * np.eye(n)[:, j]) for j in seen])
    U = np.zeros((n, len(seen)))
    for c, j in enumerate(seen):
        U[j, c] = delta
    scores = np.abs(W) + 0.01 * np.random.default_rng(0).uniform(size=(n, n))   # true edges plus weak noise everywhere
    model = vp.structure_response(scores, X, U, seen)
    assert model.orientation == "as given"
    # Two regulators are kept per target, so each target also carries one spurious parent whose ridge weight is small
    # but not zero on four training conditions: the predictions are close, not exact.
    pred = model.predict(4, delta)                                     # a leaf: its own shift dominates, little else moves
    assert pred[4] == pytest.approx(delta, rel=0.2) and np.all(np.abs(pred[:4]) < 0.5 * abs(delta))
    pred0 = model.predict(0, delta)                                    # the whole chain responds as in the data
    assert np.corrcoef(pred0, X[:, 0])[0, 1] > 0.95 and np.allclose(pred0, X[:, 0], rtol=0.3, atol=0.1)
    assert vp.structure_response(scores.T, X, U, seen).orientation == "transposed"


def test_virtual_cells_and_bystanders_have_the_requested_shape(panel):
    genes = ["a", "b", "c"]
    cells = vp.virtual_cells({"a": np.array([1.0, 2.0, 3.0]), "b": np.array([3.0, 2.0, 1.0])}, np.array([2.0, 2.0, 2.0]),
                             genes, cells_per_condition=50, seed=1, depth=1000)
    assert cells.counts.shape == (150, 3) and set(cells.labels) == {v.CONTROL, "a", "b"}
    assert abs(cells.counts.sum(axis=1).mean() / 1000 - 1.0) < 0.3
    with_by = vp.with_bystanders(panel, 200, share=0.8, seed=1)
    assert with_by.n_genes == panel.n_genes + 200 and with_by.genes[-1].startswith("BY")
    extra = with_by.counts[:, panel.n_genes:].sum(axis=1)
    assert abs(extra.sum() / with_by.counts.sum() - 0.8) < 0.03
    assert vp.with_bystanders(panel, 0) is panel


def test_screen_predictions_hold_out_a_quarter_and_every_algorithm_gets_a_row():
    from grn_sim.evaluation import screen_benchmark as sb
    from grn_sim.evaluation import vcc_benchmark as vb

    run = vb.simulate_screen(vb.Setting(generator="linear_sem", n_genes=24, cells=30), 3)
    ace = sb.screen_inputs(run, "cells", 3, normalisation=vb.NORMALISATION)
    algorithms = [a for a in vb.algorithm_library() if a.name in ("random", "mean_difference", "s2g", "gene_features")]
    panel, held, rows = vp.screen_predictions(run, ace, algorithms, 3)
    assert 2 <= len(held) <= 8 and set(held) <= set(panel.perturbations)
    keys = [r.key for r in rows]
    assert keys[:4] == ["random", "mean_difference", "s2g", "gene_features"]
    assert {"oracle_structure", "truth_means", "self_prior"} <= set(keys)
    assert all(r.folds is None or set(r.folds) == set(held) for r in rows)
    assert all(np.all(f >= 0) for r in rows if r.folds for f in r.folds.values())
    out = vp.score_prediction_rows(panel, held, rows, seed=3)
    by_key = {r["key"]: r for r in out["rows"]}
    assert by_key["baseline"]["overall"] == 0.0 and by_key["replicate"]["overall"] == 1.0
    assert all(r["available"] or r["reason"] for r in out["rows"])
    assert out["held_out"] == sorted(held) and set(out["anchors"]["baseline"]) == set(v.METRICS)
