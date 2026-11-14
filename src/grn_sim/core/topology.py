"""
Network topology generation module for gene regulatory networks.
Supports Random (Erdos-Renyi), Scale-Free (Barabasi-Albert Power Law), Hierarchical Modular and Modular
(module-structured preferential attachment, calibrated to genome-scale Perturb-seq) topologies with signed
regulatory edges (+1: activation, -1: inhibition).
"""

from enum import Enum
from typing import Callable, Optional
import numpy as np


class TopologyType(str, Enum):
    RANDOM = "random"
    POWER_LAW = "power_law"
    HIERARCHICAL = "hierarchical"
    MODULAR = "modular"
    # Pathway architectures derived from well-studied regulatory systems (generators at the end of this module).
    CASCADE = "cascade"
    FEEDFORWARD = "feedforward"
    LINEAGE = "lineage"
    REGULON = "regulon"


PATHWAY_TOPOLOGIES = (TopologyType.CASCADE, TopologyType.FEEDFORWARD, TopologyType.LINEAGE, TopologyType.REGULON)


class NetworkTopology:
    """Represents a generated network graph structure."""

    def __init__(
        self,
        adjacency: np.ndarray,
        topology_type: TopologyType,
        k_av: float,
        ratio: float,
    ):
        """
        Args:
            adjacency: (n, n) signed adjacency matrix. T[j, i] is effect of gene j on gene i.
                       +1: activation, -1: inhibition, 0: no interaction.
            topology_type: Topology classification.
            k_av: Targeted average degree per node.
            ratio: Ratio of activating edges vs inhibiting edges (e.g. 0.5 = equal).
        """
        self.adjacency = adjacency
        self.topology_type = topology_type
        self.k_av = k_av
        self.ratio = ratio
        self.n = adjacency.shape[0]

    @property
    def in_degrees(self) -> np.ndarray:
        """In-degree per gene (number of regulators influencing this gene)."""
        return np.sum(np.abs(self.adjacency) > 0, axis=0)

    @property
    def out_degrees(self) -> np.ndarray:
        """Out-degree per gene (number of targets influenced by this gene)."""
        return np.sum(np.abs(self.adjacency) > 0, axis=1)

    @property
    def total_degrees(self) -> np.ndarray:
        """Sum of in-degrees and out-degrees."""
        return self.in_degrees + self.out_degrees

    @property
    def total_edges(self) -> int:
        """Total number of non-zero interactions."""
        return int(np.sum(np.abs(self.adjacency) > 0))

    @property
    def positive_edges(self) -> int:
        """Number of activating interactions (+1)."""
        return int(np.sum(self.adjacency > 0))

    @property
    def negative_edges(self) -> int:
        """Number of inhibiting interactions (-1)."""
        return int(np.sum(self.adjacency < 0))

    @property
    def connectivity(self) -> float:
        """Connectivity zeta = k_av / n."""
        return float(self.total_edges) / float(self.n * self.n)


def generate_topology(
    n: int,
    k_av: float = 1.5,
    net_type: TopologyType = TopologyType.POWER_LAW,
    ratio: float = 0.5,
    subdiv: int = 2,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    p_within: float = 0.85,
    attachment_offset: float = 0.5,
) -> NetworkTopology:
    """
    Generates a gene regulatory network topology matching netsim.m from the thesis.

    Args:
        n: Number of genes in the network.
        k_av: Target average number of interactions per gene.
        net_type: Topology type ('random', 'power_law', 'hierarchical' or 'modular').
        ratio: Ratio threshold for positive (activating) vs negative (inhibiting) edges.
        subdiv: Number of subdivisions for hierarchical networks (must divide n); number of modules for 'modular'.
        rng: Optional NumPy random generator for deterministic seeds.
        p_within: 'modular' only: probability that a regulator is drawn from the target's own module.
        attachment_offset: 'modular' only: preferential-attachment offset (smaller = heavier-tailed out-degree).

    The netsim.m samplers for 'power_law' and 'hierarchical' accept an edge with probability ~(degree share)^2,
    which stalls once n exceeds ~60 genes (at n = 500 they end with n edges instead of k_av * n). When a sampler
    exhausts its attempts, the missing edges are drawn by the same degree-preferential rule directly
    (:func:`_top_up_edges`). Networks that never stalled are unchanged, bit for bit.

    Returns:
        NetworkTopology instance.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    T = np.zeros((n, n), dtype=int)
    dk = np.ones(n, dtype=float)  # degree distribution tracking
    target_edges = int(np.ceil(k_av * n))

    def _sample_sign() -> int:
        # Matches MATLAB: my_sign = sign(ratio - rand)
        # If rand < ratio -> +1 (activator), else -1 (inhibitor)
        return 1 if rng.uniform(0.0, 1.0) < ratio else -1

    if net_type == TopologyType.RANDOM:
        # Erdős-Rényi procedure:
        # Phase 1: assign outgoing edge from random node to each node i
        for i in range(n):
            rand_src = rng.integers(0, n)
            dk[rand_src] += 1
            T[rand_src, i] = _sample_sign()

        # Phase 2: add additional random links until target degree reached
        added = n
        while added < target_edges:
            src = rng.integers(0, n)
            dst = rng.integers(0, n)
            dk[src] += 1
            dk[dst] += 1
            T[src, dst] = _sample_sign()
            added += 1

    elif net_type == TopologyType.POWER_LAW:
        # Barabási-Albert preferential attachment procedure from netsim.m
        # Phase 1: each target node connects with probability proportional to degree
        # dk holds integer-valued floats, so the running sum equals np.sum(dk) exactly.
        sum_dk = float(np.sum(dk))
        for i in range(n):
            while True:
                candidate = rng.integers(0, n)
                prob = dk[candidate] / sum_dk
                if rng.uniform(0.0, 1.0) < prob:
                    dk[candidate] += 1
                    sum_dk += 1.0
                    T[candidate, i] = _sample_sign()
                    break

        # Phase 2: add remaining edges with preferential attachment on both source and target
        added = n
        max_attempts = 100000
        attempts = 0
        while added < target_edges and attempts < max_attempts:
            attempts += 1
            src = rng.integers(0, n)
            dst = rng.integers(0, n)
            if (dk[src] / sum_dk > rng.uniform(0.0, 1.0)) and (dk[dst] / sum_dk > rng.uniform(0.0, 1.0)):
                dk[src] += 1
                dk[dst] += 1
                sum_dk += 2.0
                T[src, dst] = _sample_sign()
                added += 1
        if added < target_edges:
            _top_up_edges(T, dk, target_edges, rng, _sample_sign)

    elif net_type == TopologyType.HIERARCHICAL:
        # Modular hierarchical structure:
        if n % subdiv != 0:
            raise ValueError(f"Network size {n} must be divisible by subdiv {subdiv}")
        sub_size = n // subdiv
        dk_sub = np.ones(sub_size, dtype=float)

        # Build base sub-module using power law
        sum_sub = float(np.sum(dk_sub))
        for i in range(sub_size):
            while True:
                cand = rng.integers(0, sub_size)
                prob = dk_sub[cand] / sum_sub
                if rng.uniform(0.0, 1.0) < prob:
                    dk_sub[cand] += 1
                    sum_sub += 1.0
                    T[cand, i] = _sample_sign()
                    break

        # Replicate sub-module across all partitions
        for part in range(1, subdiv):
            start_col = part * sub_size
            end_col = (part + 1) * sub_size
            T[:sub_size, start_col:end_col] = T[:sub_size, :sub_size]
            dk[start_col:end_col] = dk_sub

        dk[:sub_size] = dk_sub

        # Add cross-module edges between different sub-modules
        added = sub_size * subdiv
        attempts = 0
        sum_dk = float(np.sum(dk))
        while added < target_edges and attempts < 100000:
            attempts += 1
            mod1 = rng.integers(0, subdiv)
            mod2 = rng.integers(0, subdiv)
            if mod1 != mod2:
                idx1 = rng.integers(0, sub_size) + (mod1 * sub_size)
                idx2 = rng.integers(0, sub_size) + (mod2 * sub_size)
                if (dk[idx1] / sum_dk > rng.uniform(0.0, 1.0)) and (dk[idx2] / sum_dk > rng.uniform(0.0, 1.0)):
                    dk[idx1] += 1
                    dk[idx2] += 1
                    sum_dk += 2.0
                    T[idx1, idx2] = _sample_sign()
                    added += 1
        if added < target_edges and subdiv > 1:
            module = np.arange(n) // sub_size
            _top_up_edges(T, dk, target_edges, rng, _sample_sign, module=module)

    elif net_type == TopologyType.MODULAR:
        T = _modular_topology(n, k_av, max(1, subdiv), p_within, attachment_offset, rng, _sample_sign)

    elif net_type == TopologyType.CASCADE:
        T = _cascade_topology(n, k_av, rng, _sample_sign)

    elif net_type == TopologyType.FEEDFORWARD:
        T = _feedforward_topology(n, k_av, ratio, rng, _sample_sign)

    elif net_type == TopologyType.LINEAGE:
        T = _lineage_topology(n, k_av, max(2, subdiv), rng, _sample_sign)

    elif net_type == TopologyType.REGULON:
        T = _regulon_topology(n, k_av, rng, _sample_sign)

    else:
        raise ValueError(f"Unknown topology type: {net_type}")

    return NetworkTopology(adjacency=T, topology_type=net_type, k_av=k_av, ratio=ratio)


def _top_up_edges(
    T: np.ndarray,
    dk: np.ndarray,
    target_edges: int,
    rng: np.random.Generator,
    sample_sign: Callable[[], int],
    module: Optional[np.ndarray] = None,
) -> None:
    """
    Adds edges to ``T`` (in place) until it has ``target_edges`` distinct edges, drawing source and target
    independently with probability proportional to their degree ``dk`` - the distribution the netsim.m rejection
    sampler targets, without its (degree share)^2 acceptance rate. With ``module`` given (hierarchical networks),
    only edges between different modules are added, as in netsim.m's cross-module phase.
    """
    n = T.shape[0]
    max_edges = n * n if module is None else int(n * n - np.sum(np.bincount(module) ** 2))
    target = min(int(target_edges), max_edges)
    missing = target - int(np.count_nonzero(T))
    guard = 0
    while missing > 0 and guard < 1000:
        guard += 1
        p = dk / dk.sum()
        batch = max(2 * missing, 16)
        src = rng.choice(n, size=batch, p=p)
        dst = rng.choice(n, size=batch, p=p)
        for a, b in zip(src, dst):
            if missing <= 0:
                break
            if T[a, b] != 0 or (module is not None and module[a] == module[b]):
                continue
            T[a, b] = sample_sign()
            dk[a] += 1
            dk[b] += 1
            missing -= 1


def _modular_topology(
    n: int,
    k_av: float,
    n_modules: int,
    p_within: float,
    attachment_offset: float,
    rng: np.random.Generator,
    sample_sign: Callable[[], int],
) -> np.ndarray:
    """
    Module-structured preferential attachment (after Aguirre et al., PLoS Comput Biol 2025, whose simulated
    networks best matching genome-scale Perturb-seq have 2-4 regulators per gene, 5-10 modules and a few genes with
    very many targets).

    Every gene gets ``1 + Poisson(k_av - 1)`` regulators (mean in-degree ``k_av``, nobody unregulated). Each regulator
    comes from the gene's own module with probability ``p_within`` (otherwise from another module) and, within the
    chosen module, with probability proportional to ``out_degree + attachment_offset``: rich-get-richer out-degree,
    a narrow in-degree. No self-loops and no duplicate edges.
    """
    if not 0.0 <= p_within <= 1.0:
        raise ValueError(f"p_within must be in [0, 1], got {p_within}")
    if attachment_offset <= 0.0:
        raise ValueError(f"attachment_offset must be > 0, got {attachment_offset}")
    T = np.zeros((n, n), dtype=int)
    n_modules = min(n_modules, n)
    module = rng.permutation(np.arange(n) % n_modules)
    members = [np.flatnonzero(module == m) for m in range(n_modules)]
    out_deg = np.zeros(n, dtype=float)
    in_deg = 1 + rng.poisson(max(k_av - 1.0, 0.0), size=n)
    for k in rng.permutation(n):
        for _ in range(int(min(in_deg[k], n - 1))):
            own = module[k]
            if n_modules > 1 and rng.uniform() >= p_within:
                pool = members[rng.choice([m for m in range(n_modules) if m != own])]
            else:
                pool = members[own]
            cand = pool[(pool != k) & (T[pool, k] == 0)]
            if cand.size == 0:
                cand = np.flatnonzero((np.arange(n) != k) & (T[:, k] == 0))
                if cand.size == 0:
                    break
            w = out_deg[cand] + attachment_offset
            j = int(rng.choice(cand, p=w / w.sum()))
            T[j, k] = sample_sign()
            out_deg[j] += 1
    return T


# --------------------------------------------------------------------------------------------------------------------
# Pathway architectures. ``T[j, k]`` is the effect of gene j on gene k, as everywhere else. Each generator reproduces the
# wiring logic of a well-studied regulatory system, so that inference methods are tested on structures biology actually
# uses and not only on random or scale-free graphs. Degree and sign statistics follow ``k_av`` and the activating ratio.
# --------------------------------------------------------------------------------------------------------------------

def _add_edge(T: np.ndarray, src: int, dst: int, sign: int) -> bool:
    if src == dst or T[src, dst] != 0:
        return False
    T[src, dst] = sign
    return True


def _cascade_topology(n: int, k_av: float, rng: np.random.Generator, sample_sign: Callable[[], int]) -> np.ndarray:
    """
    Signalling cascade (receptor -> kinase tiers -> transcription factors -> effectors, as in MAPK / NF-kB / Wnt
    signalling): genes sit in tiers that widen towards the output; every gene is driven by 1-2 genes of the tier above
    (hub-biased: a few kinases pass the signal on to many), extra edges are skip connections from the tier above or the
    one before it, and about 8% of the edges are inhibitory feedback from a deeper tier onto a shallower one
    (phosphatases, negative feedback). Signals travel through several tiers, so most measured effects are indirect.
    """
    n_tiers = int(np.clip(round(np.log2(max(n, 4))), 3, 8))
    weights = np.linspace(1.0, 2.5 * n_tiers / 3.0, n_tiers) ** 1.5      # narrow at the top, wide at the output
    sizes = np.maximum(1, np.round(weights / weights.sum() * n).astype(int))
    while sizes.sum() > n:
        sizes[int(np.argmax(sizes))] -= 1
    while sizes.sum() < n:
        sizes[-1] += 1
    tier = np.repeat(np.arange(n_tiers), sizes)
    genes = [np.flatnonzero(tier == t) for t in range(n_tiers)]
    T = np.zeros((n, n), dtype=int)
    out_deg = np.zeros(n)

    def pick(pool: np.ndarray, k: int) -> int:
        cand = pool[(pool != k) & (T[pool, k] == 0)]
        if cand.size == 0:
            return -1
        w = out_deg[cand] + 0.5
        return int(rng.choice(cand, p=w / w.sum()))

    for t in range(1, n_tiers):
        for kg in genes[t]:
            for _ in range(1 + int(rng.uniform() < 0.4)):
                j = pick(genes[t - 1], int(kg))
                if j >= 0:
                    T[j, kg] = sample_sign()
                    out_deg[j] += 1
    target = int(np.ceil(k_av * n))
    n_feedback = int(round(0.08 * target))
    guard = 0
    while int(np.count_nonzero(T)) < target - n_feedback and guard < 50 * target:
        guard += 1
        k = int(rng.integers(0, n))
        if tier[k] == 0:
            continue
        up = genes[max(0, tier[k] - 1 - int(rng.integers(0, 2)))]            # the tier above or the one before it
        j = pick(up, k)
        if j >= 0:
            T[j, k] = sample_sign()
            out_deg[j] += 1
    guard = 0
    while int(np.count_nonzero(T)) < target and guard < 50 * target:
        guard += 1
        k = int(rng.integers(0, n))
        deeper = np.flatnonzero(tier > tier[k])
        if deeper.size:
            _add_edge(T, int(rng.choice(deeper)), k, -1)
    return T


def _feedforward_topology(n: int, k_av: float, ratio: float, rng: np.random.Generator,
                          sample_sign: Callable[[], int]) -> np.ndarray:
    """
    Feed-forward-loop-rich transcription network (Milo et al. 2002; Shen-Orr et al. 2002 found the FFL to be the
    over-represented three-gene motif in E. coli and yeast): genes are added in an order, and each new gene Z is wired
    to a regulator X *and* to a gene Y that X also regulates, making the triple X -> Y -> Z, X -> Z. Two thirds of the
    loops are coherent (the direct edge has the sign of the two-step path), the rest incoherent (opposite sign: a pulse
    generator). The network is acyclic, so every perturbation propagates one way and a direct and an indirect path
    coexist for the same pair, which is where total effects and direct edges differ most.
    """
    T = np.zeros((n, n), dtype=int)
    out_deg = np.ones(n)
    order = rng.permutation(n)                      # the topological order of the cascade
    target = int(np.ceil(k_av * n))

    def sign() -> int:
        return 1 if rng.uniform() < ratio else -1

    for pos in range(1, n):
        z = int(order[pos])
        earlier = order[:pos]
        x_pos = int(rng.choice(pos, p=out_deg[earlier] / out_deg[earlier].sum()))
        x = int(order[x_pos])
        later = order[x_pos + 1:pos]                 # genes between x and z in the order: the loop's middle gene, so no cycle
        if later.size == 0:
            _add_edge(T, x, z, sign())
            out_deg[x] += 1
            continue
        children = [int(y) for y in later if T[x, y] != 0]
        if children:
            y = int(rng.choice(children))
        else:
            y = int(rng.choice(later))
            _add_edge(T, x, y, sign())
            out_deg[x] += 1
        s_xy, s_yz = int(T[x, y]), sign()
        s_xz = s_xy * s_yz if rng.uniform() < 2.0 / 3.0 else -s_xy * s_yz
        _add_edge(T, y, z, s_yz)
        _add_edge(T, x, z, s_xz)
        out_deg[x] += 1
        out_deg[y] += 1
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    guard = 0
    while int(np.count_nonzero(T)) < target and guard < 100 * target:
        guard += 1
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if rank[a] < rank[b] and rng.uniform() < out_deg[a] / out_deg.max():
            if _add_edge(T, a, b, sample_sign()):
                out_deg[a] += 1
    return T


def _lineage_topology(n: int, k_av: float, n_lineages: int, rng: np.random.Generator,
                      sample_sign: Callable[[], int]) -> np.ndarray:
    """
    Lineage-decision network (master regulators of haematopoiesis, e.g. GATA1 / PU.1 / CEBPA): ``n_lineages`` master
    regulators, each dominating the next one in a one-way chain (a hierarchy of lineage priority; true mutual-inhibition
    toggle switches are bistable, so perturbations flip the whole network between states and no steady-state screen is
    defined for them); each master activates the transcription factors of its own lineage, which activate the lineage's
    effector genes; effectors are also repressed by the masters of the other lineages, and some lineage genes feed back
    on their own master. Perturbing a master changes a whole programme at once (large, cross-lineage effects), perturbing
    an effector changes almost nothing: very uneven total effects.
    """
    n_lineages = int(min(n_lineages, max(2, n // 6)))
    T = np.zeros((n, n), dtype=int)
    masters = np.arange(n_lineages)
    rest = np.arange(n_lineages, n)
    lineage_of = np.empty(n, dtype=int)
    lineage_of[masters] = masters
    lineage_of[rest] = rng.permutation(np.arange(rest.size) % n_lineages)
    for a in range(n_lineages - 1):
        T[a, a + 1] = -1                                                  # lineage priority chain
    n_tf = max(n_lineages, int(0.2 * rest.size))
    tf_pool = rest[:n_tf]
    effectors = rest[n_tf:]
    for g in tf_pool:
        T[int(lineage_of[g]), g] = 1
    for e in effectors:
        own = [int(g) for g in tf_pool if lineage_of[g] == lineage_of[e]] or [int(lineage_of[e])]
        T[int(rng.choice(own)), e] = 1
    target = int(np.ceil(k_av * n))
    guard = 0
    while int(np.count_nonzero(T)) < target and guard < 100 * target:
        guard += 1
        e = int(rng.choice(rest))
        r = rng.uniform()
        if r < 0.45:                                                      # a second TF of the same lineage
            pool = [int(g) for g in tf_pool if lineage_of[g] == lineage_of[e] and g != e]
            if pool:
                _add_edge(T, int(rng.choice(pool)), e, 1)
        elif r < 0.8:                                                     # another lineage's master represses it
            _add_edge(T, int(rng.choice([m for m in masters if m != lineage_of[e]])), e, -1)
        elif e in tf_pool:                                                # a lineage TF feeds back on its own master
            _add_edge(T, e, int(lineage_of[e]), -1)                       # (negative feedback keeps it stable)
        else:                                                             # an effector acts on a TF of a lower tier
            _add_edge(T, e, int(rng.choice(tf_pool)), -1)
    return T


def _regulon_topology(n: int, k_av: float, rng: np.random.Generator, sample_sign: Callable[[], int]) -> np.ndarray:
    """
    Dense overlapping regulons (the E. coli transcription network: Shen-Orr et al. 2002): about 15% of the genes are
    transcription factors with heavy-tailed target numbers; the rest are targets, each regulated by one to several TFs of
    overlapping regulons; the TFs themselves are regulated by one other TF. The network is two layers deep, so total and
    direct effects nearly coincide: the easy case for interventional methods.
    """
    n_tf = int(np.clip(round(0.15 * n), 2, n - 1))
    tfs = np.arange(n_tf)
    targets = np.arange(n_tf, n)
    T = np.zeros((n, n), dtype=int)
    popularity = rng.pareto(1.5, size=n_tf) + 0.5
    for t in range(1, n_tf):
        _add_edge(T, int(rng.choice(tfs[:t], p=popularity[:t] / popularity[:t].sum())), t, sample_sign())
    mean_in = max((int(np.ceil(k_av * n)) - int(np.count_nonzero(T))) / max(1, targets.size), 1.0)
    in_deg = 1 + rng.poisson(mean_in - 1.0, size=targets.size)
    for k, d in zip(targets, in_deg):
        regs = rng.choice(tfs, size=int(min(d, n_tf)), replace=False, p=popularity / popularity.sum())
        for r in regs:
            _add_edge(T, int(r), int(k), sample_sign())
    return T
