"""Before/after benchmark of inference algorithms (run from repo root: python scripts/benchmark_before_after.py)."""
import sys, warnings, json; sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'src')); warnings.filterwarnings('ignore')
import numpy as np
from grn_sim.core.topology import TopologyType
from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.experiment import PerturbationSimulator
from grn_sim.inference import *
from grn_sim.evaluation.metrics import evaluate_inference, compute_aupr

def methods(k):
    nir_r = lambda **kw: NIRAlgorithm(k_interactions=k, perturbation_mode="regulator", **kw)
    return {
     # (before = repository defaults prior to this work, after = improved configuration)
     "nir":      (lambda: NIRAlgorithm(k_interactions=k),              lambda: nir_r(adaptive_k=True)),
     "fast":     (lambda: FastSVD(k_interactions=k),                   lambda: FastSVD(k_interactions=k, perturbation_mode="regulator", adaptive_k=True)),
     "ridge":    (lambda: KaggleMultiOutputRidge(k_interactions=k),    lambda: KaggleMultiOutputRidge(k_interactions=k, perturbation_mode="regulator", adaptive_k=True)),
     "s2g":      (lambda: SimpleToGeneral(),                           lambda: SimpleToGeneral()),
     "g2s":      (lambda: GeneralToSpecific(),                         lambda: GeneralToSpecific(max_interactions=2*k)),
     "vote":     (lambda: VotingEnsemble(),                            lambda: VotingEnsemble(extra_members=[nir_r(adaptive_k=True)])),
     "cellbox":  (lambda: CellBoxNeuralODE(epochs=70, l1_reg=0.01, threshold=0.05, adaptive_fallback=False),
                  lambda: CellBoxNeuralODE(epochs=70, perturbation_mode="regulator")),
     "netdeconv":(lambda: NetworkDeconvolution(k_interactions=k),      lambda: NetworkDeconvolution(k_interactions=k)),
    }
if __name__ == "__main__":
    out = {}
    for topo in (TopologyType.RANDOM, TopologyType.POWER_LAW):
      for noise in (0.0, 0.05):
        key = f"{topo.value} noise={noise}"; agg = {}
        for seed in range(100, 108):
            net = GeneNetwork.create(n=12, k_av=2, net_type=topo, seed=seed)
            e = PerturbationSimulator(2.0, measurement_noise=noise).run(net, seed=seed)
            for name, (bf, af) in methods(2).items():
                for tag, f in (("before", bf), ("after", af)):
                    r = f().fit(e.X, e.U); m = evaluate_inference(e.true_matrix, r.adjacency_pred)
                    agg.setdefault((name, tag), []).append([m.precision, m.recall, m.f1_score, m.signed_ppv, compute_aupr(e.true_matrix, r.A_est), r.total_predicted_edges])
        out[key] = {f"{n}|{t}": np.mean(v, 0).tolist() for (n, t), v in agg.items()}
        print(f"\n== {key}  (8 seeds)  F1 / signedPPV / AUPR / edges   before -> after")
        for n in methods(2):
            b, a = out[key][f"{n}|before"], out[key][f"{n}|after"]
            print(f"{n:10} F1 {b[2]:.2f}->{a[2]:.2f} | sPPV {b[3]:.2f}->{a[3]:.2f} | AUPR {b[4]:.2f}->{a[4]:.2f} | edges {b[5]:.0f}->{a[5]:.0f}")
    json.dump(out, open("docs/benchmark_before_after.json", "w"), indent=1)
