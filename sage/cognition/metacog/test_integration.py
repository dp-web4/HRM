"""
Integration test: metacog + WorkingMemory + simulated consciousness loop.

Proves all five detectors fire in a realistic tick sequence that mimics
an agent playing a game: perceive → plan → act → learn, with deliberate
dysfunction patterns injected to trigger each detector.

This is the integration fixture from the metacog spec's build checklist.
"""

import pytest
from sage.cognition.working_memory import WorkingMemory
from sage.cognition.metacog import Metacog, MetacogConfig, MetacogSignal


class SimulatedLoop:
    """Minimal consciousness loop stub that drives metacog through a
    realistic tick sequence. Not the real 12-step loop — just enough
    to exercise all five detectors in one integrated scenario."""

    def __init__(self):
        self.wm = WorkingMemory(capacity=7)
        self.metacog = Metacog(
            wm=self.wm,
            config=MetacogConfig(
                perseveration_window=3,
                stale_reference_ticks=4,
                budget_anxiety_ratio=1.5,
                budget_critical_ratio=1.0,
                exploration_window=5,
                exploration_novelty_floor=0.15,
                signal_cooldown_ticks=10,
            ),
        )
        self.tick = 0
        self.atp = 50.0
        self.signals_log: list[MetacogSignal] = []

    def step(
        self,
        *,
        action: dict | None = None,
        state_delta: dict | None = None,
        novelty: float = 0.5,
        atp_cost: float = 1.0,
        est_to_goal: float = 10.0,
        goal_status: str = "in_progress",
    ) -> list[MetacogSignal]:
        """One loop tick. Returns metacog signals."""
        self.atp -= atp_cost if action else 0
        self.wm.tick()

        signals = self.metacog.observe_tick(
            self.tick,
            action_taken=action,
            state_delta=state_delta,
            snarc_novelty=novelty,
            atp_balance=self.atp,
            atp_cost=atp_cost if action else None,
            estimated_actions_to_goal=est_to_goal,
            goal_status=goal_status,
        )
        self.signals_log.extend(signals)
        self.tick += 1
        return signals

    def signal_names(self) -> list[str]:
        return [s.signal for s in self.signals_log]


class TestIntegrationScenario:
    """Full scenario: agent starts a game level, plays, gets stuck,
    runs low on budget, stops learning, recovers."""

    def test_healthy_start_no_signals(self):
        """Phase 1: agent perceives, plans, acts successfully. No dysfunction."""
        loop = SimulatedLoop()

        # Set up goal + plan in WM
        loop.wm.add_item("goal", {"objective": "reach target at (8,3)"}, priority=0.9)
        loop.wm.add_item("plan_step", {"action": "move toward target (8,3)"}, priority=0.8)
        loop.wm.add_item("binding", {"sensor": "vision", "data": "target visible"}, priority=0.5)

        # 5 ticks of healthy play — different actions, state changes, good novelty
        for i in range(5):
            s = loop.step(
                action={"type": "move", "dir": ["N", "E", "N", "E", "N"][i]},
                state_delta={"player_pos": (i, i)},
                novelty=0.6,
                atp_cost=1.0,
                est_to_goal=10.0 - i,
            )
            assert not s, f"unexpected signal at healthy tick {i}: {[x.signal for x in s]}"

    def test_perseveration_fires_on_stuck_loop(self):
        """Phase 2: agent gets stuck, repeats same action 3 times."""
        loop = SimulatedLoop()
        loop.wm.add_item("goal", {"objective": "reach target"}, priority=0.9)

        # 3 identical clicks with no state change
        stuck_action = {"type": "click", "target": (5, 5)}
        for i in range(3):
            s = loop.step(action=stuck_action, state_delta=None, novelty=0.3)
        assert "perseveration" in loop.signal_names()

    def test_intention_amnesia_fires_on_drift(self):
        """Phase 3: agent's plan drifts from its goal."""
        loop = SimulatedLoop()
        loop.wm.add_item("goal",
                         {"objective": "deliver crystal to northern vault"},
                         priority=0.9)
        loop.wm.add_item("plan_step",
                         {"action": "rotate lever in southern chamber"},
                         priority=0.8)

        s = loop.step(novelty=0.5)
        assert "intention_amnesia" in [x.signal for x in s]

    def test_stale_reference_fires_on_old_binding(self):
        """Phase 4: agent acts on a binding that hasn't been refreshed."""
        loop = SimulatedLoop()
        loop.wm.add_item("goal", {"objective": "test"}, priority=0.9)
        slot_id = loop.wm.add_item("binding",
                                    {"sensor": "vision", "frame": 1},
                                    priority=0.5)
        # Mark this slot as touched at tick 0
        loop.metacog._slot_last_touched[slot_id] = 0

        # Advance 6 ticks without refreshing (threshold=4, need age > 4)
        for _ in range(6):
            loop.step(novelty=0.5)

        assert "stale_reference" in loop.signal_names()

    def test_budget_anxiety_fires_on_overspend(self):
        """Phase 5: agent is burning ATP too fast for the remaining goal."""
        loop = SimulatedLoop()
        loop.atp = 20.0  # low starting budget

        # Spend 3 per action, estimate 10 more needed
        for i in range(4):
            loop.step(
                action={"type": "expensive_action", "id": i},
                state_delta={"progress": i},
                atp_cost=3.0,
                est_to_goal=10.0,
                novelty=0.5,
            )
        # After 4 actions: spent 12, remaining=8, avg_cost=3,
        # actions_remaining=8/3=2.67, ratio=2.67/10=0.27 < 1.0 → critical
        assert "budget_critical" in loop.signal_names()

    def test_exploration_starvation_fires_on_flat_novelty(self):
        """Phase 6: agent stops learning — novelty flatlines."""
        loop = SimulatedLoop()

        # 6 ticks of zero novelty (window=5, floor=0.15)
        for i in range(6):
            loop.step(novelty=0.02, goal_status="in_progress")

        assert "exploration_starvation" in loop.signal_names()

    def test_no_exploration_starvation_when_completed(self):
        """Goal completed → no starvation signal even with low novelty."""
        loop = SimulatedLoop()
        for i in range(6):
            loop.step(novelty=0.02, goal_status="completed")
        assert "exploration_starvation" not in loop.signal_names()


class TestFullGameSession:
    """End-to-end: a simulated game session that goes through
    healthy play → stuck → recovery → completion."""

    def test_full_session_arc(self):
        loop = SimulatedLoop()

        # --- Setup ---
        loop.wm.add_item("goal", {"objective": "solve puzzle level 3 push block to target"}, priority=0.9)
        loop.wm.add_item("plan_step", {"action": "push block right toward puzzle target"}, priority=0.8)
        binding_id = loop.wm.add_item("binding",
                                       {"sensor": "vision", "frame": "level3_start"},
                                       priority=0.5)
        loop.metacog._slot_last_touched[binding_id] = 0

        # --- Phase A: Healthy play (ticks 0-4) ---
        for i in range(5):
            s = loop.step(
                action={"type": "push", "dir": "right", "step": i},
                state_delta={"block_pos": (i + 1, 3)},
                novelty=0.6 - i * 0.05,
                atp_cost=1.0,
                est_to_goal=8.0 - i,
            )
        healthy_signals = [s for s in loop.signals_log if s.tick < 5]
        # Should be clean — diverse actions, state changing, good novelty
        assert len(healthy_signals) == 0, \
            f"unexpected signals in healthy phase: {[s.signal for s in healthy_signals]}"

        # --- Phase B: Get stuck (ticks 5-7) ---
        # Refresh the binding to avoid stale_reference here
        loop.metacog._slot_last_touched[binding_id] = 5
        stuck_action = {"type": "push", "dir": "right", "step": "stuck"}
        for i in range(3):
            loop.step(
                action=stuck_action,
                state_delta=None,  # no change — stuck!
                novelty=0.05,
                atp_cost=1.0,
                est_to_goal=4.0,
            )
        # Perseveration should have fired
        assert "perseveration" in loop.signal_names()

        # --- Phase C: Acknowledge + recover ---
        for sig in loop.metacog.active_signals():
            loop.metacog.acknowledge(sig)
        assert len(loop.metacog.active_signals()) == 0

        # Agent reframes — different action, state starts changing again
        # (ticks 8-10, cooldown prevents re-fire of perseveration)
        loop.wm.add_item("plan_step", {"action": "try pulling block instead"}, priority=0.8)
        for i in range(3):
            loop.step(
                action={"type": "pull", "dir": "left", "step": i},
                state_delta={"block_pos": (5 - i, 3)},
                novelty=0.7,
                atp_cost=1.0,
                est_to_goal=3.0 - i,
            )

        # --- Phase D: Complete ---
        loop.step(
            action={"type": "place", "target": (2, 3)},
            state_delta={"puzzle": "solved"},
            novelty=0.9,
            atp_cost=1.0,
            est_to_goal=0.0,
            goal_status="completed",
        )

        # Final stats
        stats = loop.metacog.stats()
        assert stats["actions_taken"] >= 10
        assert stats["total_atp_spent"] > 0

        # The full arc: healthy → stuck (perseveration) → recover → complete
        all_signals = loop.signal_names()
        assert "perseveration" in all_signals
        # No signals during healthy or recovery phases beyond the stuck phase
        recovery_signals = [
            s for s in loop.signals_log
            if s.tick >= 8 and s.signal == "perseveration"
        ]
        assert len(recovery_signals) == 0, "perseveration should not re-fire during recovery"
