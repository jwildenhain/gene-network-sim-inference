"""
Unit tests for AtlasShift empirical transfer algorithm, single-cell Perturb-seq simulator,
and Virtual Cell Challenge 2026 comparative benchmark.
"""

from pathlib import Path
import tempfile
import numpy as np

from grn_sim.inference.atlasshift import AtlasShift
from grn_sim.simulator.perturb_seq import (
    simulate_perturb_seq_counts,
    create_perturb_seq_anndata,
    export_perturb_seq_h5ad,
    HAS_ANNDATA,
)
from grn_sim.benchmarks.vcc_benchmark import run_vcc_comparative_benchmark, compute_profile_similarity


class TestAtlasShift:
    """Tests for AtlasShift empirical transfer algorithm."""

    def test_atlas_shift_basic_fit(self):
        rng = np.random.default_rng(42)
        n = 6
        X = rng.normal(0, 1, (n, n))
        U = np.eye(n) * 2.0

        model = AtlasShift(de_threshold=0.2)
        res = model.fit(X, U)

        assert res.algorithm_name == "atlasshift"
        assert res.A_est.shape == (n, n)
        assert res.adjacency_pred.shape == (n, n)
        # Check no self-loops
        assert np.all(np.diag(res.adjacency_pred) == 0)

    def test_fuse_source_effects(self):
        n = 4
        X1 = np.ones((n, n)) * 2.0
        X2 = np.ones((n, n)) * 4.0
        model = AtlasShift()
        fused = model.fuse_source_effects([X1, X2], weights=[1.0, 1.0])
        assert np.allclose(fused, 3.0)

    def test_desired_mean_log2fc(self):
        model = AtlasShift(amplitude=1.0, space="log2fc")
        ctrl_p = np.array([0.5, 0.5])
        effects = np.array([1.0, 0.0])  # Gene 0 doubled, Gene 1 unchanged
        desired = model.desired_mean(ctrl_p, effects)
        assert np.isclose(desired.sum(), 1.0)
        assert desired[0] > desired[1]

    def test_dual_moment_counts(self):
        model = AtlasShift()
        n_cells = 50
        n_genes = 5
        template = np.ones((n_cells, n_genes))
        p = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
        depths = np.full(n_cells, 1000)

        counts = model.dual_moment_counts(template, p, depths, seed=42, iterations=10)
        assert counts.shape == (n_cells, n_genes)
        assert np.all(counts >= 0)
        # Total counts per cell should match depths
        row_totals = counts.sum(axis=1)
        assert np.all(row_totals == 1000)


class TestPerturbSeqSimulator:
    """Tests for single-cell Perturb-seq simulation and AnnData export."""

    def test_simulate_perturb_seq_counts(self):
        gene_names = ["LexA", "RecA", "DinI"]
        # Expression: 3 genes, 4 conditions (control + 3 single knockdowns)
        mean_expr = np.array([
            [1.0, 0.2, 0.8, 1.1],
            [1.0, 0.9, 0.1, 1.0],
            [1.0, 1.2, 1.0, 0.1],
        ])
        counts, cpm, obs_df, var_df = simulate_perturb_seq_counts(
            gene_names=gene_names,
            mean_expression_matrix=mean_expr,
            cells_per_condition=50,
            mean_library_depth=5000,
            seed=42,
        )

        assert counts.shape == (200, 3)
        assert cpm.shape == (200, 3)
        assert len(obs_df) == 200
        assert len(var_df) == 3
        assert "condition" in obs_df.columns
        assert "total_counts" in obs_df.columns
        assert np.all(counts >= 0)

    def test_create_and_export_anndata(self):
        gene_names = ["G1", "G2"]
        mean_expr = np.array([
            [1.0, 0.5, 1.5],
            [1.0, 1.2, 0.2],
        ])
        adata = create_perturb_seq_anndata(
            gene_names=gene_names,
            mean_expression_matrix=mean_expr,
            cells_per_condition=20,
            mean_library_depth=2000,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_perturb_seq.h5ad"
            saved_path = export_perturb_seq_h5ad(adata, out_path)
            if HAS_ANNDATA:
                assert saved_path.exists()
                assert saved_path.suffix == ".h5ad"
            else:
                assert saved_path.with_suffix(".npz").exists()


class TestVCCComparativeBenchmark:
    """Tests for Virtual Cell Challenge comparative benchmark execution."""

    def test_profile_similarity(self):
        y1 = np.array([1.0, 2.0, 3.0])
        y2 = np.array([1.0, 2.0, 3.0])
        sim = compute_profile_similarity(y1, y2)
        assert sim["pearson_r"] == 1.0
        assert sim["cosine_sim"] == 1.0
        assert sim["rmse"] == 0.0

    def test_run_vcc_benchmark(self):
        res = run_vcc_comparative_benchmark(num_genes=6, k_av=1.5, seed=42, rewire_fractions=[0.0, 0.2])
        assert res["status"] == "success"
        assert "summary_table" in res
        assert "in_distribution" in res
        assert "out_of_distribution" in res
        assert "rewire_decay" in res

        # Verify AtlasShift is included in the summary table
        algorithms = [row["algorithm"] for row in res["summary_table"]]
        assert "ATLASSHIFT" in algorithms
        assert "S2G" in algorithms
        assert "G2S" in algorithms
        assert "VOTE" in algorithms
