"""
Metacog core — five dysfunction detectors + signal framework.

Two detectors (budget_anxiety, exploration_starvation) are fully
functional now. Three (perseveration, intention_amnesia, stale_reference)
require WM v0.2 features (stable_key, get_by_type, slot timestamps)
and are implemented against the WM interface but will only activate
once those features ship.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class MetacogSignal:
    """A dysfunction signal emitted by the metacog monitor."""

    signal: str  # e.g. "perseveration", "budget_anxiety"
    severity: float  # 0.0–1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""
    tick: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "severity": self.severity,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
            "tick": self.tick,
            "timestamp": self.timestamp,
        }


@dataclass
class MetacogConfig:
    """Tunable parameters for each detector."""

    # Perseveration
    perseveration_window: int = 3  # actions to compare
    perseveration_severity: float = 0.8

    # Stale reference
    stale_reference_ticks: int = 5  # max age before flagging
    stale_reference_severity: float = 0.6

    # Intention amnesia
    intention_amnesia_severity: float = 0.7

    # Budget anxiety
    budget_anxiety_ratio: float = 1.5  # actions_remaining / estimated_to_goal
    budget_anxiety_severity: float = 0.5
    budget_critical_ratio: float = 1.0
    budget_critical_severity: float = 0.9

    # Exploration starvation
    exploration_novelty_floor: float = 0.1
    exploration_window: int = 10  # ticks to average novelty over
    exploration_severity: float = 0.6

    # General
    signal_cooldown_ticks: int = 5  # min ticks between re-firing same type


class Metacog:
    """Interoceptive monitor for the consciousness loop.

    Call observe_tick() once per loop iteration with whatever signals
    are available. Returns any dysfunction signals detected.

    Read-only WM consumer. Does not take actions.
    """

    def __init__(
        self,
        config: Optional[MetacogConfig] = None,
        wm: Any = None,  # WorkingMemory when CBP ships v0.2
        on_signal: Optional[Callable[[MetacogSignal], None]] = None,
    ) -> None:
        self.config = config or MetacogConfig()
        self._wm = wm
        self._on_signal = on_signal

        # Internal state
        self._tick = 0
        self._active_signals: List[MetacogSignal] = []
        self._acknowledged: set = set()  # (signal_type, tick) pairs
        self._last_fired: Dict[str, int] = {}  # signal_type → last tick

        # Detector state
        self._action_history: deque = deque(maxlen=50)
        self._novelty_history: deque = deque(maxlen=100)
        self._total_atp_spent: float = 0.0
        self._actions_taken: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def observe_tick(
        self,
        tick: int,
        *,
        action_taken: Optional[Dict[str, Any]] = None,
        state_delta: Optional[Dict[str, Any]] = None,
        snarc_novelty: Optional[float] = None,
        atp_balance: Optional[float] = None,
        atp_cost: Optional[float] = None,
        estimated_actions_to_goal: Optional[float] = None,
        goal_status: Optional[str] = None,
        goal_content: Optional[Dict[str, Any]] = None,
        plan_step_content: Optional[Dict[str, Any]] = None,
    ) -> List[MetacogSignal]:
        """Called once per consciousness-loop tick. Returns signals detected."""
        self._tick = tick
        signals: List[MetacogSignal] = []

        # Track action history
        if action_taken is not None:
            self._action_history.append({
                "tick": tick,
                "action": action_taken,
                "state_delta": state_delta,
            })
            self._actions_taken += 1
            if atp_cost is not None:
                self._total_atp_spent += atp_cost

        # Track novelty
        if snarc_novelty is not None:
            self._novelty_history.append(snarc_novelty)

        # Run detectors
        signals.extend(self._detect_perseveration())
        signals.extend(self._detect_intention_amnesia(goal_content, plan_step_content))
        signals.extend(self._detect_stale_reference())
        signals.extend(
            self._detect_budget_anxiety(atp_balance, estimated_actions_to_goal)
        )
        signals.extend(self._detect_exploration_starvation(goal_status))

        # Apply cooldown + emit
        emitted: List[MetacogSignal] = []
        for sig in signals:
            if self._should_fire(sig.signal):
                sig.tick = tick
                self._active_signals.append(sig)
                self._last_fired[sig.signal] = tick
                if self._on_signal:
                    self._on_signal(sig)
                emitted.append(sig)

        return emitted

    # ------------------------------------------------------------------
    # Signal management
    # ------------------------------------------------------------------

    def active_signals(self) -> List[MetacogSignal]:
        return [
            s for s in self._active_signals
            if (s.signal, s.tick) not in self._acknowledged
        ]

    def acknowledge(self, signal: MetacogSignal) -> None:
        self._acknowledged.add((signal.signal, signal.tick))

    def stats(self) -> Dict[str, Any]:
        return {
            "tick": self._tick,
            "actions_taken": self._actions_taken,
            "total_atp_spent": self._total_atp_spent,
            "active_signals": len(self.active_signals()),
            "total_signals_emitted": len(self._active_signals),
            "avg_novelty": (
                sum(self._novelty_history) / len(self._novelty_history)
                if self._novelty_history
                else None
            ),
        }

    def reset(self) -> None:
        self._active_signals.clear()
        self._acknowledged.clear()
        self._last_fired.clear()
        self._action_history.clear()
        self._novelty_history.clear()
        self._total_atp_spent = 0.0
        self._actions_taken = 0

    # ------------------------------------------------------------------
    # Router integration — feature_extraction.py calls get_block_list()
    # ------------------------------------------------------------------

    def get_block_list(self) -> List[str]:
        """Plugin names the router should NOT invoke this tick.

        Called by the router feature extractor (PRD §2.7). Only critical-
        severity signals (>= 0.9) produce blocks. The block list is a
        mapping from signal type to a heuristic "what plugin to avoid":

          perseveration → block the plugin that was perseverating
          budget_critical → block all non-essential plugins
          exploration_starvation → (no block — suggestion, not prohibition)

        For now, returns signal names as pseudo-plugin identifiers. The
        router treats unknown plugin names as noop (PRD §3.3 rule 5).
        """
        blocks: List[str] = []
        for sig in self.active_signals():
            if sig.severity >= 0.9:
                blocks.append(f"metacog:{sig.signal}")
        return blocks

    # ------------------------------------------------------------------
    # Detector: perseveration
    # ------------------------------------------------------------------

    def _detect_perseveration(self) -> List[MetacogSignal]:
        """Repeated similar actions with no state change."""
        cfg = self.config
        window = list(self._action_history)[-cfg.perseveration_window:]
        if len(window) < cfg.perseveration_window:
            return []

        # Check if all actions in the window have zero state delta
        all_zero_delta = all(
            not entry.get("state_delta") for entry in window
        )
        if not all_zero_delta:
            return []

        # Check if actions are similar (same keys in action dict)
        # With WM v0.2, this will use stable_key(). For now, use
        # a simple string comparison of sorted action keys.
        action_strs = [
            str(sorted(entry.get("action", {}).items())) for entry in window
        ]
        if len(set(action_strs)) > 1:
            return []  # different actions — not perseveration

        return [
            MetacogSignal(
                signal="perseveration",
                severity=cfg.perseveration_severity,
                evidence={
                    "action_count": len(window),
                    "repeated_action": window[-1].get("action"),
                    "state_deltas": [e.get("state_delta") for e in window],
                },
                suggestion="reframe or abandon current approach",
            )
        ]

    # ------------------------------------------------------------------
    # Detector: intention amnesia
    # ------------------------------------------------------------------

    def _detect_intention_amnesia(
        self,
        goal_content: Optional[Dict[str, Any]],
        plan_step_content: Optional[Dict[str, Any]],
    ) -> List[MetacogSignal]:
        """Goal doesn't match the action being taken."""
        if goal_content is None or plan_step_content is None:
            return []

        # Simple check: do goal and plan_step share any key terms?
        goal_terms = set(str(goal_content).lower().split())
        plan_terms = set(str(plan_step_content).lower().split())

        # If goal and plan share fewer than 20% of terms, flag
        if not goal_terms or not plan_terms:
            return []
        overlap = len(goal_terms & plan_terms) / max(len(goal_terms), len(plan_terms))
        if overlap >= 0.2:
            return []

        return [
            MetacogSignal(
                signal="intention_amnesia",
                severity=self.config.intention_amnesia_severity,
                evidence={
                    "goal_content": goal_content,
                    "plan_step_content": plan_step_content,
                    "term_overlap": round(overlap, 3),
                },
                suggestion="re-read goal before next action",
            )
        ]

    # ------------------------------------------------------------------
    # Detector: stale reference
    # ------------------------------------------------------------------

    def _detect_stale_reference(self) -> List[MetacogSignal]:
        """Acting on state information that's no longer current.

        Full implementation requires WM v0.2 binding slot timestamps.
        Stub: check if we have WM and if any binding is stale.
        """
        if self._wm is None:
            return []

        # WM v0.2 API: get_by_type("binding") returns slots with timestamps
        try:
            bindings = self._wm.get_by_type("binding")
        except (AttributeError, TypeError):
            return []  # WM doesn't support get_by_type yet

        stale: List[Dict[str, Any]] = []
        for slot in bindings:
            age = self._tick - getattr(slot, "last_access_tick", self._tick)
            if age > self.config.stale_reference_ticks:
                stale.append({
                    "slot_id": getattr(slot, "slot_id", "?"),
                    "age_ticks": age,
                })

        if not stale:
            return []

        return [
            MetacogSignal(
                signal="stale_reference",
                severity=self.config.stale_reference_severity,
                evidence={"stale_bindings": stale},
                suggestion="re-observe before acting",
            )
        ]

    # ------------------------------------------------------------------
    # Detector: budget anxiety
    # ------------------------------------------------------------------

    def _detect_budget_anxiety(
        self,
        atp_balance: Optional[float],
        estimated_actions_to_goal: Optional[float],
    ) -> List[MetacogSignal]:
        """Spending rate will exhaust resources before goal completion."""
        if atp_balance is None or estimated_actions_to_goal is None:
            return []
        if estimated_actions_to_goal <= 0:
            return []

        avg_cost = (
            self._total_atp_spent / self._actions_taken
            if self._actions_taken > 0
            else 1.0
        )
        actions_remaining = atp_balance / avg_cost if avg_cost > 0 else float("inf")
        ratio = actions_remaining / estimated_actions_to_goal

        cfg = self.config
        if ratio < cfg.budget_critical_ratio:
            return [
                MetacogSignal(
                    signal="budget_critical",
                    severity=cfg.budget_critical_severity,
                    evidence={
                        "atp_balance": atp_balance,
                        "avg_cost_per_action": round(avg_cost, 3),
                        "actions_remaining": round(actions_remaining, 1),
                        "estimated_to_goal": estimated_actions_to_goal,
                        "ratio": round(ratio, 3),
                    },
                    suggestion="abort current strategy — not enough energy to finish",
                )
            ]
        if ratio < cfg.budget_anxiety_ratio:
            return [
                MetacogSignal(
                    signal="budget_anxiety",
                    severity=cfg.budget_anxiety_severity,
                    evidence={
                        "atp_balance": atp_balance,
                        "avg_cost_per_action": round(avg_cost, 3),
                        "actions_remaining": round(actions_remaining, 1),
                        "estimated_to_goal": estimated_actions_to_goal,
                        "ratio": round(ratio, 3),
                    },
                    suggestion="reassess strategy cost",
                )
            ]
        return []

    # ------------------------------------------------------------------
    # Detector: exploration starvation
    # ------------------------------------------------------------------

    def _detect_exploration_starvation(
        self, goal_status: Optional[str]
    ) -> List[MetacogSignal]:
        """System stopped learning — novelty flatlined but goal not achieved."""
        cfg = self.config
        if len(self._novelty_history) < cfg.exploration_window:
            return []
        if goal_status == "completed":
            return []

        recent = list(self._novelty_history)[-cfg.exploration_window:]
        avg_novelty = sum(recent) / len(recent)

        if avg_novelty >= cfg.exploration_novelty_floor:
            return []

        return [
            MetacogSignal(
                signal="exploration_starvation",
                severity=cfg.exploration_severity,
                evidence={
                    "avg_novelty": round(avg_novelty, 4),
                    "window": cfg.exploration_window,
                    "goal_status": goal_status,
                },
                suggestion="try something different — current path is not generating new information",
            )
        ]

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------

    def _should_fire(self, signal_type: str) -> bool:
        last = self._last_fired.get(signal_type)
        if last is None:
            return True
        return self._tick - last >= self.config.signal_cooldown_ticks
