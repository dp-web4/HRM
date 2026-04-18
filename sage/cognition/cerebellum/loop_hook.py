"""
Consciousness loop integration for the cerebellum.

Provides CerebellumLoopHook — a stateless adapter that wires the
cerebellum into SAGEConsciousness at the correct loop steps:

  Step 5 (Select):  lookup() checks for matching habits
  Step 7 (Execute): execute() replays cached habits
  Step 8 (Learn):   observe() records (state, actions, outcome)

The hook reads WM state via stable_key() and feeds it to the
cerebellum's lookup/observe methods. The router's feature extraction
(sage.cognition.router.feature_extraction) already consumes
cerebellum.lookup() directly — this hook handles the other two steps.

Usage:
    from sage.cognition.cerebellum.loop_hook import CerebellumLoopHook

    hook = CerebellumLoopHook(cerebellum, domain="arc-game:cd82")

    # At step 5 — check for habits (router also does this via feature extraction)
    matches = hook.on_select(wm, goal_id="g1")

    # At step 7 — if router chose a habit, execute it
    success, result = hook.on_execute(habit, env)

    # At step 8 — after action completes, record the observation
    hook.on_learn(wm, actions_taken, outcome, goal_id="g1")
"""

from typing import Any, Optional

from sage.cognition.cerebellum.core import (
    Cerebellum,
    Habit,
    HabitMatch,
    StateSignature,
)


class CerebellumLoopHook:
    """Adapter that plugs the cerebellum into the consciousness loop."""

    def __init__(self, cerebellum: Cerebellum, domain: str = "sage"):
        self.cerebellum = cerebellum
        self.domain = domain
        self._last_state: Optional[StateSignature] = None

    def on_select(
        self,
        wm: Any,
        goal_id: Optional[str] = None,
    ) -> list[HabitMatch]:
        """Step 5 (Select): check for matching habits.

        Builds a StateSignature from WM and queries the cerebellum.
        Stores the state signature for use in on_learn().
        """
        try:
            state = StateSignature.from_wm(wm, self.domain, goal_id)
        except Exception:
            state = StateSignature(domain=self.domain, features={})

        self._last_state = state
        return self.cerebellum.lookup(state)

    def on_execute(
        self,
        habit: Habit,
        env: Any,
    ) -> tuple[bool, Any]:
        """Step 7 (Execute): replay a cached habit.

        Returns (success, result). The cerebellum updates the
        habit's trust metrics internally.
        """
        return self.cerebellum.execute(habit, env)

    def on_learn(
        self,
        wm: Any,
        actions_taken: list,
        outcome: dict,
        goal_id: Optional[str] = None,
    ) -> Optional[Habit]:
        """Step 8 (Learn): record a (state, actions, outcome) observation.

        Uses the state signature captured during on_select() if available,
        otherwise builds a fresh one from WM.
        """
        state = self._last_state
        if state is None:
            try:
                state = StateSignature.from_wm(wm, self.domain, goal_id)
            except Exception:
                state = StateSignature(domain=self.domain, features={})

        habit = self.cerebellum.observe(state, actions_taken, outcome)
        self._last_state = None  # reset for next cycle
        return habit

    def on_consolidate(
        self,
        episodes: list,
        domain: Optional[str] = None,
    ) -> list[Habit]:
        """Dream/rest state: batch-compile habits from episodic memory.

        Delegates to cerebellum.compile_from_episodes().
        For trajectory-based compilation, use the episodic_bridge
        module directly.
        """
        return self.cerebellum.compile_from_episodes(episodes)

    @property
    def stats(self) -> dict:
        return self.cerebellum.stats()
