#!/usr/bin/env python3
"""
Lookahead-driven prompt — reframes the LLM invocation around predictions.

Instead of: "here's the state, pick an action"
This does: "here are the consequences of each action, which one advances the goal?"

The key insight: the LLM ignores predictions buried in a long prompt.
This makes predictions THE CENTRAL QUESTION, not supplementary context.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import copy

ACTION_NAMES = ["A0", "UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"]


def build_lookahead_prompt(
    env: Any,
    fd: Any,
    game: str,
    level: int,
    step_index: int,
    world_model_summary: str = "",
    recent_actions: List[int] = None,
    nn_hint: int = 0,
    nn_confidence: float = 0.0,
) -> Tuple[str, Dict]:
    """Build a prompt centered on action predictions.

    Returns (prompt_text, lookahead_data) where lookahead_data contains
    the raw predictions for logging/analysis.
    """
    from arcengine import GameAction
    from sage.cognition.thalamic_router.lookahead import lookahead, _get_frame

    # Get predictions
    predictions_text, elapsed = lookahead(env, fd, actions=[1, 2, 3, 4, 5])

    # Also try a few click coordinates if relevant
    click_predictions = ""
    curr_frame = _get_frame(fd)
    # Simple heuristic: try clicking near center and in quadrants
    click_coords = [(32, 32), (16, 16), (48, 16), (16, 48), (48, 48)]
    click_text, click_elapsed = lookahead(env, fd, actions=[], click_coords=click_coords)

    # Filter to only non-trivial click results
    active_clicks = [
        line for line in click_text.split('\n')
        if 'CLICK' in line and 'no change' not in line and 'trivial' not in line
    ]

    recent_str = " → ".join(
        ACTION_NAMES[a] if 0 <= a < len(ACTION_NAMES) else str(a)
        for a in (recent_actions or [])[-5:]
    ) or "(none)"

    nn_hint_name = ACTION_NAMES[nn_hint] if 0 <= nn_hint < len(ACTION_NAMES) else str(nn_hint)

    # Build the prediction-centered prompt
    prompt = f"""You are playing {game} (level {level}, step {step_index}).

{f"Game mechanics: {world_model_summary[:500]}" if world_model_summary else ""}

Recent actions: {recent_str}
NN suggestion: {nn_hint_name} (confidence {nn_confidence:.2f})

I tested what each action does from the current state:

{predictions_text}
{chr(10).join(active_clicks) if active_clicks else "  No effective CLICK targets found at standard coordinates."}

IMPORTANT: Use the predictions above to make your decision. Which action makes the most PROGRESS toward winning?

- Actions showing "no change" are BLOCKED — don't choose them.
- Actions showing large pixel changes are SIGNIFICANT — they affect the game state.
- SEL typically activates/launches/commits. Choose it when you're in position.
- If all directional actions are blocked, you may need to CLICK or try a different approach.

Actions: A0=0 UP=1 DOWN=2 LEFT=3 RIGHT=4 SEL=5 CLICK=6

First, briefly state which actions are blocked and which are available based on the predictions.
Then choose the best action.

ACTION=UP or DOWN or LEFT or RIGHT or SEL or CLICK[ X=<0-63> Y=<0-63>]
<one-sentence rationale referencing the prediction data>"""

    lookahead_data = {
        "predictions_text": predictions_text,
        "active_clicks": active_clicks,
        "elapsed_ms": (elapsed + click_elapsed) * 1000,
    }

    return prompt, lookahead_data


# ─── Quick test ───────────────────────────────────────────────────

if __name__ == "__main__":
    from arc_agi import Arcade

    arcade = Arcade(operation_mode='offline')
    env = arcade.make('cd82-fb555c5d')
    fd = env.reset()

    prompt, data = build_lookahead_prompt(
        env, fd, game="cd82", level=0, step_index=1,
        world_model_summary="Octagonal painting: basket orbits 8 positions. SEL launches paint.",
        nn_hint=3, nn_confidence=0.72,
    )
    print(prompt)
    print(f"\n--- Lookahead data ---")
    print(f"Elapsed: {data['elapsed_ms']:.0f}ms")
