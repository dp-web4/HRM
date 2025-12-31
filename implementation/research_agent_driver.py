"""Research agent driver for Web4 4-Life integration (v0).

This module provides a minimal, local-first driver that can:

- Consume a Web4 world / life summary produced by
  `web4/game/one_life_home_society.py::run_single_life`.
- Propose a simple, rule-based action for the "research agent".
- Optionally log each life as an "episode" via a callback so it can be
  integrated with existing HRM memory tooling (e.g. `store_episode`).

Design goals (v0):
- No external network calls.
- No hard dependency on large LLMs; this can run on CPU.
- Keep the interface narrow so SAGE/LLM policy code can be plugged in
  later without changing Web4 / 4-Life wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Protocol


class EpisodeLogger(Protocol):
    """Protocol for episode logging.

    Any object that implements this can be passed into the driver to
    capture life summaries in a way that is compatible with existing
    HRM memory patterns (e.g. `memory.store_episode`).
    """

    def store_episode(self, inputs: Dict[str, Any], outputs: Dict[str, Any], success: bool) -> None:  # pragma: no cover - interface only
        ...


@dataclass
class AgentMetrics:
    """Per-life metrics tracked for the research agent."""

    life_state: str
    t3_history: list[float]
    atp_history: list[float]

    @property
    def final_t3(self) -> float:
        return self.t3_history[-1] if self.t3_history else 0.0

    @property
    def final_atp(self) -> float:
        return self.atp_history[-1] if self.atp_history else 0.0


@dataclass
class ProposedAction:
    """A minimal action proposal to feed back into the Web4 game loop."""

    action_type: str
    atp_cost: float
    description: str


class ResearchAgentDriver:
    """Heuristic research agent policy for the first 4-Life vertical slice.

    This v0 driver is intentionally simple:
    - It inspects T3 / ATP trajectories for the research agent.
    - It chooses between a few coarse actions:
      * "conservative_audit" when ATP is low or T3 is fragile.
      * "risky_spend" when ATP is high but T3 is stable.
      * "idle" when the life is already terminated.
    - It can log the life as an episode through an EpisodeLogger.

    Later versions can replace `propose_action` with a call into
    SAGE/LLM-based reasoning while preserving the same interface.
    """

    def __init__(self, episode_logger: Optional[EpisodeLogger] = None) -> None:
        self.episode_logger = episode_logger

    def _extract_metrics(self, life_summary: Dict[str, Any]) -> AgentMetrics:
        """Convert a Web4 life summary into AgentMetrics.

        Expected keys (per `run_single_life`):
        - life_state: "alive" | "terminated"
        - t3_history: list[float]
        - atp_history: list[float]
        """

        life_state = str(life_summary.get("life_state", "unknown"))
        t3_history = list(life_summary.get("t3_history", []) or [])
        atp_history = list(life_summary.get("atp_history", []) or [])
        return AgentMetrics(life_state=life_state, t3_history=t3_history, atp_history=atp_history)

    def propose_action(self, life_summary: Dict[str, Any]) -> ProposedAction:
        """Propose a simple action based on the current life summary.

        This is a **heuristic policy** designed for early experiments:

        - If the life is already terminated:
          -> return an "idle" action with zero cost.
        - Else, compute final T3 and ATP, then:
          - If ATP < 20 or T3 < 0.3:
            -> propose a conservative audit-style action.
          - If ATP > 80 and T3 >= 0.5:
            -> propose a higher-cost risky spend.
          - Otherwise:
            -> propose a small, exploratory spend.
        """

        metrics = self._extract_metrics(life_summary)

        if metrics.life_state == "terminated":
            return ProposedAction(
                action_type="idle",
                atp_cost=0.0,
                description="Life terminated; no further actions proposed.",
            )

        final_t3 = metrics.final_t3
        final_atp = metrics.final_atp

        if final_atp < 20.0 or final_t3 < 0.3:
            return ProposedAction(
                action_type="conservative_audit",
                atp_cost=5.0,
                description="Low ATP or fragile T3; propose audit/verification instead of spend.",
            )

        if final_atp > 80.0 and final_t3 >= 0.5:
            return ProposedAction(
                action_type="risky_spend",
                atp_cost=25.0,
                description="High ATP and solid T3; propose a higher-cost experimental spend.",
            )

        return ProposedAction(
            action_type="small_spend",
            atp_cost=10.0,
            description="Moderate ATP/T3; propose a small exploratory spend.",
        )

    def log_episode(self, life_summary: Dict[str, Any], action: ProposedAction) -> None:
        """Optionally log this life + action as an episode.

        This provides a bridge into HRM's existing `store_episode` style
        without taking a hard dependency on any particular memory class.
        """

        if self.episode_logger is None:
            return

        metrics = self._extract_metrics(life_summary)

        inputs = {
            "life_summary": life_summary,
        }
        outputs = {
            "agent_metrics": asdict(metrics),
            "proposed_action": asdict(action),
        }
        success = metrics.life_state == "alive" and metrics.final_t3 >= 0.3

        try:
            self.episode_logger.store_episode(inputs=inputs, outputs=outputs, success=success)
        except Exception:
            # Episode logging is non-critical for early experiments.
            pass


def run_policy_once(life_summary: Dict[str, Any], episode_logger: Optional[EpisodeLogger] = None) -> Dict[str, Any]:
    """Convenience helper to run the heuristic policy once.

    Returns a JSON-serializable dict with both the raw proposed action
    and the derived agent metrics, suitable for debugging or piping into
    higher-level orchestrators.
    """

    driver = ResearchAgentDriver(episode_logger=episode_logger)
    action = driver.propose_action(life_summary)
    metrics = driver._extract_metrics(life_summary)

    return {
        "agent_metrics": asdict(metrics),
        "proposed_action": asdict(action),
    }
