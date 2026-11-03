"""
Orientation of each inference algorithm's predicted matrix.

The simulator and the studio read a network as T[regulator, target] (``NetworkTopology``). The algorithms do not
share one output orientation: the Gardner-style algorithms (A X = -U) estimate A[target, regulator] on additive
(linear steady-state) data, while on the Hill simulator's data most of them come out as T[regulator, target], and a
few algorithms return A[target, regulator] whatever the data. Fits under the regulator-input perturbation model
(``regulator_input_fit``) are already transposed to T[regulator, target].

The sets below were measured on networks with known topology (cyclic and acyclic, 8-12 genes), using the
configurations of the studio, the benchmark studio and the parameter sweep: an algorithm is listed when its
predictions match T.T clearly better than T (at least 10 points more true positives and 1.3 times as many).
Algorithms that match T, or match both about equally, are not listed and are taken as T[regulator, target].
GENIE3, CausalGRN, DeepRIG and SFINN fill row i from the evidence for target i (W[i, j] = regulator j -> target i), so
they are A[target, regulator] by construction on every data model; GENIE3's importances are close to symmetric on
additive data and on large scale-free networks, so the measurement alone does not separate its orientations there.
``tests/test_inference_orientation.py`` re-checks the main entries.
"""

from typing import Dict, FrozenSet

import numpy as np

# Data the algorithms are fitted to.
HILL = "hill"  # Hill kinetics simulator (hill_nonlinear, perturb_seq_counts, biophysical fits)
ADDITIVE_GARDNER = "additive_gardner"  # linear steady state with B X = -U (DREAM4 / Norman / Replogle synthetic sets)
ADDITIVE_PHYSICAL = "additive_physical"  # linear steady state with B X = +U (linear / CellBox engines, measured data)
DATA_MODELS = (HILL, ADDITIVE_GARDNER, ADDITIVE_PHYSICAL)

_ALIASES = {"ridge": "ridge_cv"}

# A[target, regulator] by construction, whatever the data.
_TARGET_ROW_ALGORITHMS = frozenset({"genie3", "causalgrn", "deeprig", "sfinn"})

# Estimate A[target, regulator] on additive data whatever the sign convention (ridge_cv solves A X = -U row by row
# like NIR and FAST; its dense predictions match both orientations about equally, so it is listed by construction).
_ADDITIVE_TARGET_REGULATOR = frozenset({"s2g", "g2s", "vote", "score_vote", "nir", "fast", "cellbox", "ridge_cv"}) | _TARGET_ROW_ALGORITHMS

TARGET_REGULATOR_OUTPUT: Dict[str, FrozenSet[str]] = {
    HILL: _TARGET_ROW_ALGORITHMS,
    # NOTEARS follows the sign of the response: A[target, regulator] when X opposes U, T[regulator, target] otherwise
    ADDITIVE_GARDNER: _ADDITIVE_TARGET_REGULATOR | {"notears"},
    ADDITIVE_PHYSICAL: _ADDITIVE_TARGET_REGULATOR,
}

# Random baselines: either orientation scores the same in expectation, so they are left as returned.
ORIENTATION_FREE = frozenset({"by_chance"})

# Algorithms whose regulator-input fit (``perturbation_mode="regulator"``) returns T[regulator, target].
REGULATOR_INPUT_ALGORITHMS = frozenset({"nir", "fast", "ridge_cv", "cellbox"})


def declare_output_orientation(
    algorithm: str,
    *,
    target_row: bool = False,
    additive_target_regulator: bool = False,
    regulator_input: bool = False,
    orientation_free: bool = False,
) -> None:
    """
    Declares how a newly registered algorithm orients its output, so scoring needs no edit here (called by
    ``grn_sim.inference.registry.register``):

    * ``target_row``: A[target, regulator] by construction on every data model (GENIE3-style);
    * ``additive_target_regulator``: A[target, regulator] on additive data (the Gardner family), T[regulator, target]
      on Hill data;
    * ``regulator_input``: its ``perturbation_mode="regulator"`` fit returns T[regulator, target];
    * ``orientation_free``: random baselines, left as returned.
    """
    global _TARGET_ROW_ALGORITHMS, _ADDITIVE_TARGET_REGULATOR, ORIENTATION_FREE, REGULATOR_INPUT_ALGORITHMS
    if target_row:
        _TARGET_ROW_ALGORITHMS = _TARGET_ROW_ALGORITHMS | {algorithm}
    if target_row or additive_target_regulator:
        _ADDITIVE_TARGET_REGULATOR = _ADDITIVE_TARGET_REGULATOR | {algorithm}
    if regulator_input:
        REGULATOR_INPUT_ALGORITHMS = REGULATOR_INPUT_ALGORITHMS | {algorithm}
    if orientation_free:
        ORIENTATION_FREE = ORIENTATION_FREE | {algorithm}
    # the dict is shared with importers, so it is updated in place
    TARGET_REGULATOR_OUTPUT[HILL] = _TARGET_ROW_ALGORITHMS
    TARGET_REGULATOR_OUTPUT[ADDITIVE_GARDNER] = _ADDITIVE_TARGET_REGULATOR | {"notears"}
    TARGET_REGULATOR_OUTPUT[ADDITIVE_PHYSICAL] = _ADDITIVE_TARGET_REGULATOR


def output_is_target_regulator(algorithm: str, data_model: str, perturbation_mode: str = "additive") -> bool:
    """True when ``algorithm`` fitted to ``data_model`` data returns A[target, regulator]."""
    if data_model not in TARGET_REGULATOR_OUTPUT:
        raise ValueError(f"Unknown data model {data_model!r}; choose from {DATA_MODELS}")
    key = _ALIASES.get(algorithm, algorithm)
    if key in ORIENTATION_FREE:
        return False
    if perturbation_mode == "regulator" and key in REGULATOR_INPUT_ALGORITHMS:
        return False
    return key in TARGET_REGULATOR_OUTPUT[data_model]


def orient_prediction(
    algorithm: str,
    matrix: np.ndarray,
    data_model: str,
    target_regulator: bool = False,
    perturbation_mode: str = "additive",
) -> np.ndarray:
    """
    ``matrix`` (as returned by ``algorithm``) in the orientation of the matrix it is scored against:
    A[target, regulator] when ``target_regulator`` (the E. coli SOS and real Norman ground truths),
    T[regulator, target] otherwise.
    """
    m = np.asarray(matrix)
    if _ALIASES.get(algorithm, algorithm) in ORIENTATION_FREE:
        return m
    flipped = output_is_target_regulator(algorithm, data_model, perturbation_mode) != target_regulator
    return m.T.copy() if flipped else m
