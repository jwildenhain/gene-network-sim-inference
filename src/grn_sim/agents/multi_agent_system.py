"""
Multi-Agent System for autonomous gene network simulation, inference, and benchmarking.
Provides specialized autonomous agents collaborating across the simulation and inference lifecycle:
- SimulationAgent: generates topologies, configures kinetic models, executes perturbation experiments
- InferenceAgent: executes multiple inference algorithms (S2G, G2S, NIR, FAST, Ensemble)
- EvaluationAgent: calculates comparative metrics, confusion matrices, and performance summaries
- CoordinatorAgent: orchestrates end-to-end multi-agent pipelines and synthesizes reports
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

from grn_sim.core.topology import TopologyType
from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.experiment import PerturbationSimulator, SimulationExperiment
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.fast_svd import FastSVD
from grn_sim.inference.ensemble import VotingEnsemble
from grn_sim.inference.by_chance import ByChance
from grn_sim.evaluation.metrics import evaluate_inference, BenchmarkMetrics


@dataclass
class AgentMessage:
    """Inter-agent communication message."""
    sender: str
    recipient: str
    topic: str
    payload: Dict[str, Any]


class BaseAgent(ABC):
    """Abstract base class for autonomous simulation and inference agents."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.inbox: List[AgentMessage] = []

    def receive(self, message: AgentMessage) -> None:
        self.inbox.append(message)

    @abstractmethod
    def step(self) -> Optional[AgentMessage]:
        """Processes pending messages and produces action or response message."""
        pass


class SimulationAgent(BaseAgent):
    """Specialist agent responsible for topology generation and in silico perturbation experiments."""

    def __init__(self):
        super().__init__(name="SimulationAgent", role="In Silico Experiment Designer & Simulator")
        self.last_experiment: Optional[SimulationExperiment] = None

    def step(self) -> Optional[AgentMessage]:
        if not self.inbox:
            return None
        msg = self.inbox.pop(0)

        if msg.topic == "RUN_SIMULATION":
            p = msg.payload
            n = p.get("n", 10)
            k_av = p.get("k_av", 1.5)
            net_type = TopologyType(p.get("net_type", "power_law"))
            noise = p.get("measurement_noise", 0.0)
            bio_var = p.get("biological_variability", 0.0)
            seed = p.get("seed", 42)

            net = GeneNetwork.create(n=n, k_av=k_av, net_type=net_type, seed=seed)
            sim = PerturbationSimulator(
                measurement_noise=noise,
                biological_variability=bio_var,
            )
            exp = sim.run(net, seed=seed)
            self.last_experiment = exp

            return AgentMessage(
                sender=self.name,
                recipient=msg.sender,
                topic="SIMULATION_COMPLETE",
                payload={"experiment": exp},
            )
        return None


class InferenceAgent(BaseAgent):
    """Specialist agent responsible for orchestrating reverse engineering algorithms."""

    def __init__(self):
        super().__init__(name="InferenceAgent", role="Network Reverse Engineering Specialist")
        self.algos = {
            "s2g": SimpleToGeneral(),
            "g2s": GeneralToSpecific(),
            "nir": NIRAlgorithm(k_interactions=2),
            "fast": FastSVD(k_interactions=2),
            "voting": VotingEnsemble(),
            "by_chance": ByChance(),
        }

    def step(self) -> Optional[AgentMessage]:
        if not self.inbox:
            return None
        msg = self.inbox.pop(0)

        if msg.topic == "RUN_INFERENCE":
            exp: SimulationExperiment = msg.payload["experiment"]
            k_val = msg.payload.get("k_interactions", 2)

            results = {}
            for name, algo in self.algos.items():
                res = algo.fit(exp.X, exp.U, k=k_val)
                results[name] = res

            return AgentMessage(
                sender=self.name,
                recipient=msg.sender,
                topic="INFERENCE_COMPLETE",
                payload={"inference_results": results, "true_matrix": exp.true_matrix},
            )
        return None


class EvaluationAgent(BaseAgent):
    """Specialist agent responsible for benchmarking, verification, and diagnostics."""

    def __init__(self):
        super().__init__(name="EvaluationAgent", role="Quality Assurance & Evaluation Specialist")

    def step(self) -> Optional[AgentMessage]:
        if not self.inbox:
            return None
        msg = self.inbox.pop(0)

        if msg.topic == "EVALUATE_RESULTS":
            inf_results = msg.payload["inference_results"]
            A_true = msg.payload["true_matrix"]

            metrics_table = {}
            for name, res in inf_results.items():
                m = evaluate_inference(A_true, res.adjacency_pred)
                metrics_table[name] = m

            return AgentMessage(
                sender=self.name,
                recipient=msg.sender,
                topic="EVALUATION_COMPLETE",
                payload={"metrics": metrics_table},
            )
        return None


class CoordinatorAgent(BaseAgent):
    """Lead Orchestrator Agent directing the autonomous multi-agent simulation and inference loop."""

    def __init__(self):
        super().__init__(name="CoordinatorAgent", role="Lead SDLC & Workflow Coordinator")
        self.sim_agent = SimulationAgent()
        self.inf_agent = InferenceAgent()
        self.eval_agent = EvaluationAgent()

    def run_study(
        self,
        n: int = 12,
        k_av: float = 1.5,
        net_type: str = "power_law",
        noise: float = 0.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Executes an autonomous multi-agent simulation, inference, and benchmarking study.
        """
        # Step 1: Dispatch simulation task
        sim_request = AgentMessage(
            sender=self.name,
            recipient=self.sim_agent.name,
            topic="RUN_SIMULATION",
            payload={"n": n, "k_av": k_av, "net_type": net_type, "measurement_noise": noise, "seed": seed},
        )
        self.sim_agent.receive(sim_request)
        sim_reply = self.sim_agent.step()
        exp = sim_reply.payload["experiment"]

        # Step 2: Dispatch inference task
        k_interactions = max(1, int(round(k_av)))
        inf_request = AgentMessage(
            sender=self.name,
            recipient=self.inf_agent.name,
            topic="RUN_INFERENCE",
            payload={"experiment": exp, "k_interactions": k_interactions},
        )
        self.inf_agent.receive(inf_request)
        inf_reply = self.inf_agent.step()
        inf_results = inf_reply.payload["inference_results"]

        # Step 3: Dispatch evaluation task
        eval_request = AgentMessage(
            sender=self.name,
            recipient=self.eval_agent.name,
            topic="EVALUATE_RESULTS",
            payload={"inference_results": inf_results, "true_matrix": exp.true_matrix},
        )
        self.eval_agent.receive(eval_request)
        eval_reply = self.eval_agent.step()
        metrics = eval_reply.payload["metrics"]

        # Synthesize final report
        summary = {
            "network_parameters": {
                "genes": n,
                "topology": net_type,
                "k_av": k_av,
                "noise": noise,
                "true_edges": int(np.sum(exp.true_matrix != 0)),
            },
            "metrics": {name: m.to_dict() for name, m in metrics.items()},
            "inferred_matrices": {name: res.adjacency_pred for name, res in inf_results.items()},
            "true_matrix": exp.true_matrix,
        }
        return summary

    def step(self) -> Optional[AgentMessage]:
        return None
