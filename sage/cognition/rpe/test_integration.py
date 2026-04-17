"""
Integration Test: RPE + Working Memory in a Game-Playing Loop

Simulates a simplified consciousness loop where:
1. WM holds the current game state (goal, plan_step, bindings)
2. RPE predicts outcomes before actions
3. Actions execute (simulated)
4. RPE observes outcomes, updates priors
5. Router queries RPE priors for next action selection

This is the minimal fixture proving RPE integrates with WM.
"""

import json
import time
from typing import Dict, List, Optional

from sage.cognition.working_memory import WorkingMemory, PlanStep
from sage.cognition.rpe.core import (
    RewardPredictionError,
    compute_outcome_value,
)


class SimulatedGame:
    """A trivial game: 3x3 grid, player at (0,0), gem at (2,2).
    RIGHT and DOWN are good. UP and LEFT hit walls (waste actions).
    """

    def __init__(self):
        self.player = [0, 0]
        self.gem = [2, 2]
        self.steps = 0
        self.max_steps = 20
        self.won = False
        self.dead = False

    def step(self, action: str) -> Dict:
        self.steps += 1
        dx, dy = {"UP": (0, -1), "DOWN": (0, 1),
                  "LEFT": (-1, 0), "RIGHT": (1, 0)}.get(action, (0, 0))

        new_x = self.player[0] + dx
        new_y = self.player[1] + dy

        # Bounds check
        if 0 <= new_x <= 2 and 0 <= new_y <= 2:
            self.player = [new_x, new_y]
            moved = True
        else:
            moved = False

        # Win check
        if self.player == self.gem:
            self.won = True

        # Budget check
        if self.steps >= self.max_steps:
            self.dead = True

        pixel_delta = 50 if moved else 0
        return {
            "moved": moved,
            "pixel_delta": pixel_delta,
            "level_up": self.won,
            "died": self.dead and not self.won,
            "position": list(self.player),
        }


def test_rpe_wm_game_loop():
    """Full integration: RPE + WM in a game-playing loop."""

    print("\n" + "=" * 60)
    print("INTEGRATION TEST: RPE + WM Game Loop")
    print("=" * 60)

    # Initialize components
    # Default prior 0.0 = "I expect nothing" — any positive outcome is a surprise
    # This encourages exploration: unknown actions get tried because RPE is positive
    signals_received = []
    rpe = RewardPredictionError(
        default_prior=0.0,
        on_signal=lambda s: signals_received.append(s)
    )
    wm = WorkingMemory(capacity=7)
    game = SimulatedGame()
    available_actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    # Set up WM with game goal
    wm.add_item("goal", {"game": "test", "target": "reach gem at (2,2)"},
                 priority=1.0, goal_id="game1")

    total_actions = 0
    action_log = []

    # Play the game
    for turn in range(15):
        if game.won or game.dead:
            break

        # Step 5: Router queries RPE priors
        state_key = f"pos_{game.player[0]}_{game.player[1]}"
        priors = rpe.get_action_priors(state_key, available_actions)

        # Policy: exploit best known action, with diminishing exploration
        # Early turns: explore (try untried actions). Later: exploit best priors.
        explore_bonus = max(0, 0.3 - turn * 0.05)  # decays to 0 by turn 6
        scored = {}
        for a, p in priors.items():
            entry = rpe.priors.get_entry(state_key, a)
            if entry is None or entry.observation_count == 0:
                scored[a] = p + explore_bonus
            else:
                scored[a] = p
        best_action = max(scored, key=scored.get)

        # Write plan_step to WM
        wm.add_item("plan_step",
                     {"action": best_action, "expected": f"move {best_action}"},
                     priority=0.8, goal_id="game1")

        # Step 7: RPE predicts before action
        prediction = rpe.predict(state_key, best_action,
                                 expected_outcome=f"move {best_action}")

        # Execute action
        result = game.step(best_action)
        total_actions += 1

        # Step 8: RPE observes outcome
        signal = rpe.observe(
            prediction,
            actual_outcome=f"pos={result['position']}, moved={result['moved']}",
            pixel_delta=result["pixel_delta"],
            level_up=result["level_up"],
            died=result["died"],
        )

        # Update WM with observation
        wm.add_item("binding",
                     {"position": result["position"], "turn": turn},
                     priority=0.5, goal_id="game1")

        action_log.append({
            "turn": turn,
            "action": best_action,
            "result": result["moved"],
            "rpe": round(signal.rpe, 3),
            "prior_after": round(signal.prior_after, 3),
        })

        # WM tick (decay ttl)
        wm.tick()

    print(f"\nGame result: {'WON' if game.won else 'LOST'}")
    print(f"Total actions: {total_actions}")
    print(f"Signals received: {len(signals_received)}")

    # Print action log
    print(f"\nAction log:")
    for entry in action_log:
        indicator = "✓" if entry["result"] else "✗"
        print(f"  Turn {entry['turn']}: {entry['action']} {indicator} "
              f"RPE={entry['rpe']:+.3f} prior→{entry['prior_after']:.3f}")

    # Verify RPE learned
    stats = rpe.get_stats()
    print(f"\nRPE stats: {json.dumps(stats, indent=2)}")

    # Check that priors at (0,0) reflect learned values
    start_priors = rpe.get_action_priors("pos_0_0", available_actions)
    print(f"\nLearned priors at start position:")
    for action, prior in sorted(start_priors.items()):
        print(f"  {action}: {prior:.3f}")

    # Assertions
    # RPE alone doesn't solve multi-step games — it INFORMS the router.
    # What we verify: the learning signal is correct, not that it wins.
    assert len(signals_received) == total_actions, "Every action should emit a signal"
    assert stats["total_signals"] > 0

    # Wall-hitting actions should have lower priors than movement actions
    # At (0,0): UP hits top wall, LEFT hits left wall
    # After learning, wall-hitting priors should be ≤ movement priors
    if "pos_0_0" in [s.prediction.state_key for s in signals_received]:
        up_prior = start_priors.get("UP", 0.5)
        down_prior = start_priors.get("DOWN", 0.5)
        assert up_prior <= down_prior, \
            f"UP (wall) prior ({up_prior:.3f}) should be ≤ DOWN (movement) ({down_prior:.3f})"

    # WM should have items
    assert len(wm.get_by_type("binding")) > 0

    # Dump roundtrip
    rpe_dump = rpe.dump()
    wm_dump = wm.dump()
    assert json.dumps(rpe_dump)  # serializable
    assert json.dumps(wm_dump)  # serializable

    print("\n✓ All assertions passed")
    print("=" * 60)


def test_rpe_learns_from_failure():
    """RPE should learn to avoid actions that lead to wall hits."""

    rpe = RewardPredictionError()

    # Simulate: action "UP" at state "corner" always hits wall (0px change)
    for _ in range(5):
        pred = rpe.predict("corner", "UP")
        rpe.observe(pred, pixel_delta=0)  # wall hit

    # Simulate: action "RIGHT" at state "corner" always moves (50px)
    for _ in range(5):
        pred = rpe.predict("corner", "RIGHT")
        rpe.observe(pred, pixel_delta=50)

    priors = rpe.get_action_priors("corner", ["UP", "RIGHT"])
    assert priors["RIGHT"] > priors["UP"], \
        f"RIGHT ({priors['RIGHT']:.3f}) should be preferred over UP ({priors['UP']:.3f}) after learning"
    print(f"✓ Failure learning: UP={priors['UP']:.3f}, RIGHT={priors['RIGHT']:.3f}")


def test_rpe_death_produces_strong_signal():
    """Death should produce a large negative RPE and dramatically lower the prior."""

    rpe = RewardPredictionError()

    # First action: die
    pred = rpe.predict("danger_zone", "CLICK")
    signal = rpe.observe(pred, died=True)

    assert signal.rpe < -0.5, f"Death RPE should be strongly negative, got {signal.rpe}"
    assert signal.prior_after < 0.1, f"Prior after death should be very low, got {signal.prior_after}"
    print(f"✓ Death signal: RPE={signal.rpe:.3f}, prior→{signal.prior_after:.3f}")


if __name__ == "__main__":
    test_rpe_wm_game_loop()
    test_rpe_learns_from_failure()
    test_rpe_death_produces_strong_signal()
    print("\n✓ All integration tests passed!")
