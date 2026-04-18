"""
Consciousness loop integration for RPE (Reward Prediction Error).

Provides RPELoopHook — a stateless adapter that wires RPE into
SAGEConsciousness at the correct loop steps:

  Step 4.5 (Select):  priors() returns action priors for router weighting
  Step 6 (Execute):   predict() registers expected outcome BEFORE action
  Step 8 (Learn):     observe() computes RPE AFTER action, updates priors

The hook reads WM state via stable_key() for state identification.
The router's feature extraction already reads rpe.get_action_priors()
directly — this hook handles the predict/observe cycle.

Usage:
    from sage.cognition.rpe.loop_hook import RPELoopHook

    hook = RPELoopHook(rpe)

    # At step 4.5 — router queries priors
    priors = hook.on_select(wm, available_actions, goal_id="g1")

    # At step 6 — BEFORE executing the chosen action
    prediction = hook.on_pre_execute(wm, action="RIGHT", goal_id="g1",
                                     expected_outcome="player moves right")

    # At step 8 — AFTER action completes
    signal = hook.on_learn(prediction, wm, pixel_delta=50,
                           level_up=False, died=False)
"""

from typing import Any, Dict, List, Optional

from sage.cognition.rpe.core import (
    RewardPredictionError,
    Prediction,
    RPESignal,
    compute_outcome_value,
)


class RPELoopHook:
    """Stateless adapter wiring RPE into the consciousness loop.

    The hook doesn't own the RPE — it adapts between the loop's calling
    convention and RPE's API. Multiple hooks can share one RPE instance
    (e.g., one per goal domain).
    """

    def __init__(self, rpe: RewardPredictionError, domain: str = "default"):
        self.rpe = rpe
        self.domain = domain
        self._pending_prediction: Optional[Prediction] = None

    def on_select(self, wm, available_actions: List[str],
                  goal_id: Optional[str] = None) -> Dict[str, float]:
        """Step 4.5 (Select): Return action priors for router weighting.

        The router uses these as soft weights, not hard constraints.
        Unknown actions get default_prior (0.0 = "no expectation").

        Args:
            wm: WorkingMemory instance (for stable_key)
            available_actions: list of action names
            goal_id: optional goal context

        Returns:
            Dict mapping action name to predicted value [0, 1]
        """
        state_key = self._state_key(wm, goal_id)
        return self.rpe.get_action_priors(state_key, available_actions)

    def on_pre_execute(self, wm, action: str,
                       goal_id: Optional[str] = None,
                       expected_outcome: str = "",
                       confidence: float = 0.5) -> Prediction:
        """Step 6 (pre-Execute): Register prediction BEFORE acting.

        Call this BEFORE the action is executed. The returned Prediction
        object must be passed to on_learn() after the action completes.

        Args:
            wm: WorkingMemory instance
            action: the action about to be taken
            goal_id: optional goal context
            expected_outcome: textual description of expected result
            confidence: how confident in the prediction (from T3 Training)

        Returns:
            Prediction object to pass to on_learn()
        """
        state_key = self._state_key(wm, goal_id)
        prediction = self.rpe.predict(
            state_key=state_key,
            action=action,
            expected_outcome=expected_outcome,
            confidence=confidence,
        )
        self._pending_prediction = prediction
        return prediction

    def on_learn(self, prediction: Optional[Prediction] = None,
                 wm=None,
                 actual_outcome: str = "",
                 pixel_delta: int = 0,
                 level_up: bool = False,
                 died: bool = False,
                 budget_remaining: float = 1.0,
                 actual_value: Optional[float] = None) -> Optional[RPESignal]:
        """Step 8 (Learn): Compute RPE AFTER action, update priors.

        Uses the prediction from on_pre_execute(). If no prediction was
        registered (e.g., action was a habit bypass), uses the pending
        prediction or creates a default one.

        Args:
            prediction: from on_pre_execute() (or None to use pending)
            wm: WorkingMemory instance (for updated state)
            actual_outcome: textual description of what happened
            pixel_delta: raw pixel change count
            level_up: did the level advance?
            died: did the player die?
            budget_remaining: fraction of action budget remaining
            actual_value: override computed value (for custom valuation)

        Returns:
            RPESignal with the prediction error and prior update
        """
        if prediction is None:
            prediction = self._pending_prediction
        if prediction is None:
            # No prediction registered — can't compute RPE
            return None

        signal = self.rpe.observe(
            prediction=prediction,
            actual_outcome=actual_outcome,
            pixel_delta=pixel_delta,
            level_up=level_up,
            died=died,
            budget_remaining=budget_remaining,
            actual_value=actual_value,
        )
        self._pending_prediction = None
        return signal

    def get_stats(self) -> Dict[str, Any]:
        """Get RPE statistics for metacog / dashboard."""
        stats = self.rpe.get_stats()
        stats["domain"] = self.domain
        return stats

    def _state_key(self, wm, goal_id: Optional[str] = None) -> str:
        """Extract state key from WM for prior table lookup."""
        if wm is not None and hasattr(wm, 'stable_key'):
            try:
                return f"{self.domain}:{wm.stable_key(goal_id)}"
            except Exception:
                pass
        # Fallback: use domain + goal_id
        return f"{self.domain}:{goal_id or 'none'}"
