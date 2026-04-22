"""
Lean prompt builder for small-model dispatch.

Builds a ~100-300 token prompt focused on the decision question,
not the architecture manual. The model's entire attention is on
"which action advances the goal?" — not on parsing dispatch pipeline
documentation.

Fleet convergence (2026-04-22):
  - Thor: lean prompt (260 tok) → 2 levels in 45 min (17× faster)
  - Legion: "prompt shape, not model size" — 7-organ regression is
    a prompt problem (context overload) not a capability problem
  - McNugget: 280-token proposal (world model 100, lookahead 100,
    frame state 50, hint 20, format 10)
  - Nomad: 35% parse failure on E2B with full prompt; directional-
    only games (simpler output) parse at 79%

Design principle: the prompt is a QUESTION, not a MANUAL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_lean_prompt(
    *,
    game: str,
    level: int,
    step: int,
    world_model_summary: str = "",
    predictions: str = "",
    frame_state_summary: str = "",
    nn_hint: str = "",
    stuck_info: str = "",
    goal_hint: str = "",
) -> str:
    """Build a lean dispatch prompt. Target: 100-300 tokens.

    Args:
        game: game family name
        level: current level
        step: current step number
        world_model_summary: 1-2 sentences of game mechanics (100 tok max)
        predictions: lookahead results per action (100 tok max)
        frame_state_summary: computed facts from frame diff (50 tok max)
        nn_hint: adapter's top suggestion + confidence (20 tok max)
        stuck_info: metacog stuck signal if active (20 tok max)
        goal_hint: what winning looks like if known (20 tok max)
    """
    parts: List[str] = []

    # Identity: 1 line (10 tok)
    parts.append(f"You are playing {game} (level {level}, step {step}).")

    # World model: brief (≤100 tok)
    if world_model_summary:
        parts.append(f"Mechanics: {world_model_summary}")

    # Goal if known (≤20 tok)
    if goal_hint:
        parts.append(f"Goal: {goal_hint}")

    # Predictions: the CENTRAL question (≤100 tok)
    if predictions:
        parts.append(f"I tested what each action does:")
        parts.append(predictions)
        parts.append("Use the predictions above to choose.")
    else:
        parts.append("Available actions: UP, DOWN, LEFT, RIGHT, SEL, CLICK(x,y)")

    # Frame state: computed facts (≤50 tok)
    if frame_state_summary:
        parts.append(f"Frame: {frame_state_summary}")

    # NN hint (≤20 tok)
    if nn_hint:
        parts.append(f"Hint: {nn_hint}")

    # Stuck warning (≤20 tok)
    if stuck_info:
        parts.append(f"WARNING: {stuck_info}")

    # Format instruction: tight (10 tok)
    parts.append("Reply with ONLY: ACTION=NAME")
    parts.append("Example: ACTION=LEFT")
    parts.append("For clicks: ACTION=CLICK X=32 Y=48")

    return "\n".join(parts)


def build_lean_retry_prompt() -> str:
    """Ultra-short retry prompt when first response didn't parse."""
    return (
        "Previous response could not be parsed.\n"
        "Reply with ONLY: ACTION=NAME\n"
        "Options: UP, DOWN, LEFT, RIGHT, SEL, CLICK"
    )


# --------------------------------------------------------------------------
# World model summaries per game (≤100 tokens each)
# Derived from world-models/{game}.md, compressed to essentials.
# --------------------------------------------------------------------------

GAME_SUMMARIES: Dict[str, str] = {
    "cd82": "Octagonal ring with 8 positions. Basket navigates ring. SEL/LAUNCH paints wedge regions. Match target color pattern.",
    "sb26": "Pick-and-place sequence puzzle. Select objects, place to match target. L6+ uses swap instead of pick-place.",
    "ft09": "Click-only constraint puzzle. Toggle cells to satisfy neighbor rules. Each click affects adjacent cells.",
    "tr87": "Two-row symbol cycling. UP/DOWN cycle symbols in columns. Match target pattern on both rows.",
    "sc25": "Spell-casting movement with teleportation. Cast spells to move, collect resources, reach exit.",
    "tu93": "Maze with chain reactions. Navigate to trigger activation zones in sequence. Timing matters.",
    "r11l": "Creature assembly via compass walking. Move endpoints to position ball at target.",
    "cn04": "Jigsaw connector alignment. Rotate and position pieces so connectors match neighbors.",
    "sp80": "Gravity pipes. Place pipe segments to route water into receptacles. Gravity pulls down.",
    "vc33": "Seesaw balance. Slide beams and rotate triggers to align indicators with targets.",
    "ar25": "Mirror reflection. Position movable shapes to match reflection across axis.",
    "lp85": "Cyclic permutation on grid tracks. Click to cycle positions along track loops.",
    "ka59": "Sokoban push puzzle with explosions. Push blocks to targets. Some blocks explode on contact.",
    "wa30": "Grid pathfinding with paired agents. Move player while followers auto-path to targets.",
    "su15": "Suika merge game. Vacuum fruits, dodge enemies. Same-type fruits merge on contact.",
    "sk48": "Rail routing. Build track segments to route train to match target pattern.",
    "g50t": "Clone replay maze. Record movement sequence, then clone replays it while you do something else.",
    "tn36": "Programmable block movement. Compose action sequences, debug programs.",
    "s5i5": "Slider-card rotation. Rotate cards on sliders to route paths. Parent-child hierarchy.",
    "m0r0": "Mirrored maze with symmetric paired tokens. Both tokens move simultaneously, mirrored.",
    "bp35": "Side-scrolling platformer. Gravity, timing, viewport awareness. Jump and navigate obstacles.",
    "dc22": "Crane + bridges + pressure plates + color cycling. Multi-mechanic puzzle.",
    "lf52": "Peg solitaire with pushable blocks. Jump pegs over each other to reduce count.",
    "re86": "Shape deformation + pattern matching. Deform shapes to match multi-attribute targets.",
    "ls20": "Maze + shape/color/rotation matching + fog of war. Navigate to matching exits.",
}
