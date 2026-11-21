"""
Paper algorithms that are implemented and benchmarked but deliberately *not* registered (``grn_sim.inference.registry``): the
registry's contents define the archive leaderboard, the stored runs and the studio, so an algorithm joins it only after it has
beaten the existing baselines (see ``docs/results/paper_algorithms.md``).

* ``DCDFGLite`` - a linear-Gaussian, dense-acyclicity version of DCD-FG (Lopez et al., NeurIPS 2022): interventional
  likelihood on cells with a low-rank factor-graph weight matrix. It fits cells, not an (X, U) pair, so it cannot be
  registered (the registry contract is ``fit(X, U)``) and it did not beat the baselines.

LLC's mean-shift equations (``LLCMean``) passed the comparison and are registered as ``llc_mean``
(``grn_sim.inference.llc``).
"""
from grn_sim.inference.experimental.dcdfg import DCDFGLite

__all__ = ["DCDFGLite"]
