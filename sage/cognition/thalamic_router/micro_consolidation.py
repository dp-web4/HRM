#!/usr/bin/env python3
"""Micro-consolidation — extract discovered physics from a game attempt.

After a game attempt (GAME_OVER, level clear, or stuck timeout), review
the trajectory and extract causal rules the LLM discovered during play.
These rules become the MechanicsBlock for the next attempt.

This is the dream cycle's narrowest form: one LLM call that turns
experience into persistent knowledge within a game session.

The consolidation prompt asks: "Given what happened, what are the
causal rules of this game?" Not "what should I do next" — that's
strategy. This is physics: "action X at state Y produces effect Z."

Usage:
    from sage.cognition.thalamic_router.micro_consolidation import consolidate_attempt

    # After a game attempt
    physics = consolidate_attempt(
        game="cd82", level=0,
        trajectory=trajectory,  # list of (action, frame_delta_pct, rationale)
        world_model_before=current_wm_text,
        llm=ollama_client,
    )
    # physics is a string of discovered causal rules
    # Feed it back as the MechanicsBlock for the next attempt
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ───────────────────────────────────────────────────────────────────
# Data types
# ───────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    """One step from a game attempt."""
    step: int
    action: str          # "UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"
    coords: Optional[Dict[str, int]]  # {"x": 38, "y": 46} for CLICK
    frame_delta_pct: float  # percentage of pixels that changed
    level_before: int
    level_after: int
    rationale: str       # what the LLM said (if invoked), else ""


# ───────────────────────────────────────────────────────────────────
# Consolidation prompt
# ───────────────────────────────────────────────────────────────────

CONSOLIDATION_PROMPT = """You just played {game} (level {level}). The attempt ended: {outcome}.

Here is what happened — each line is one action and its effect:
{trajectory_summary}

{world_model_section}

Your job: extract the CAUSAL RULES of this game. Not strategy — physics.
"When I do X at state Y, effect Z happens."

Rules format (one per line):
- ACTION at CONDITION → EFFECT
- ACTION at CONDITION → EFFECT

Only include rules you have EVIDENCE for from the trajectory above.
Do not guess. Do not include rules from the world model unless the
trajectory confirms them. If you discovered nothing new, say "No new
rules discovered."

Keep it short — max 10 rules."""


# ───────────────────────────────────────────────────────────────────
# Trajectory summarization (deterministic, no LLM)
# ───────────────────────────────────────────────────────────────────

def _summarize_trajectory(steps: List[TrajectoryStep], max_lines: int = 40) -> str:
    """Compress trajectory into a concise text summary.

    Focuses on steps where something HAPPENED (frame change > 2%,
    level change, or first/last of a stuck sequence). Collapses
    repeated no-change actions into "... repeated N times, no effect."
    """
    lines = []
    i = 0
    while i < len(steps) and len(lines) < max_lines:
        s = steps[i]

        # Check for runs of same action with no effect
        run_start = i
        while (i < len(steps) - 1
               and steps[i + 1].action == s.action
               and steps[i + 1].frame_delta_pct < 2.0
               and steps[i + 1].level_after == steps[i + 1].level_before):
            i += 1

        run_len = i - run_start + 1

        if run_len >= 3:
            lines.append(f"  Steps {s.step}-{steps[i].step}: {s.action} × {run_len} → no effect")
        elif s.level_after > s.level_before:
            coord_str = f" at ({s.coords['x']},{s.coords['y']})" if s.coords else ""
            lines.append(f"  Step {s.step}: {s.action}{coord_str} → LEVEL {s.level_before}→{s.level_after} ★")
        elif s.frame_delta_pct >= 2.0:
            coord_str = f" at ({s.coords['x']},{s.coords['y']})" if s.coords else ""
            lines.append(f"  Step {s.step}: {s.action}{coord_str} → {s.frame_delta_pct:.1f}% frame change")
        elif i == 0 or i == len(steps) - 1:
            # Always include first and last
            coord_str = f" at ({s.coords['x']},{s.coords['y']})" if s.coords else ""
            lines.append(f"  Step {s.step}: {s.action}{coord_str} → {s.frame_delta_pct:.1f}% change")

        i += 1

    return "\n".join(lines) if lines else "  (no meaningful actions recorded)"


# ───────────────────────────────────────────────────────────────────
# The consolidation call
# ───────────────────────────────────────────────────────────────────

def consolidate_attempt(
    game: str,
    level: int,
    trajectory: List[TrajectoryStep],
    world_model_before: str,
    outcome: str,  # "GAME_OVER", "LEVEL_CLEAR", "MAX_STEPS"
    llm: Any,      # OllamaClient or similar with .chat()
) -> str:
    """One LLM call: trajectory → discovered causal rules.

    Returns a string of discovered rules suitable for appending to
    the MechanicsBlock on the next attempt. Returns empty string
    if consolidation fails or produces nothing new.
    """
    traj_summary = _summarize_trajectory(trajectory)

    wm_section = ""
    if world_model_before:
        wm_section = f"The world model said before this attempt:\n{world_model_before[:500]}"

    prompt = CONSOLIDATION_PROMPT.format(
        game=game,
        level=level,
        outcome=outcome,
        trajectory_summary=traj_summary,
        world_model_section=wm_section,
    )

    try:
        response = llm.chat(prompt, max_tokens=500)
    except Exception:
        return ""

    # Clean the response — extract just the rules
    response = response.strip()
    if not response or "no new rules" in response.lower():
        return ""

    # Keep only lines that look like rules (start with - or contain →)
    rule_lines = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("-") or "→" in line or "->" in line:
            rule_lines.append(line)

    return "\n".join(rule_lines) if rule_lines else response[:300]


# ───────────────────────────────────────────────────────────────────
# Multi-attempt wrapper
# ───────────────────────────────────────────────────────────────────

def play_with_consolidation(
    play_fn,           # function that plays one attempt, returns result dict
    game: str,
    max_attempts: int = 3,
    llm: Any = None,
    **play_kwargs,
) -> Dict[str, Any]:
    """Play a game with micro-consolidation between attempts.

    After each attempt, consolidates discovered physics and feeds
    it into the next attempt's world model. Compounds understanding
    across retries.

    play_fn signature: play_fn(game, system_prompt=..., **play_kwargs) -> dict
    The result dict must have: final_levels, final_state, llm_responses (with
    action, coords, rationale, frame_delta_pct per step).
    """
    accumulated_physics = []
    all_results = []

    for attempt in range(max_attempts):
        # Build system prompt with accumulated physics
        physics_text = ""
        if accumulated_physics:
            physics_text = (
                "\n\nDISCOVERED PHYSICS (from previous attempts on this game):\n"
                + "\n".join(accumulated_physics)
            )

        result = play_fn(game=game, system_prompt=physics_text or None, **play_kwargs)
        all_results.append(result)

        # Check if we won
        levels = result.get("final_levels", 0)
        state = result.get("final_state", "UNKNOWN")

        print(f"\n[Attempt {attempt + 1}/{max_attempts}] L{levels}, {state}")
        if accumulated_physics:
            print(f"  Physics carried: {len(accumulated_physics)} rules")

        if state == "WON":
            break

        # Extract trajectory for consolidation
        responses = result.get("llm_responses", [])
        if not responses:
            continue

        trajectory = []
        for r in responses:
            action_name = r.get("action_name", "")
            if not action_name:
                action_idx = r.get("action", 0)
                ACTION_NAMES = ["A0", "UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"]
                action_name = ACTION_NAMES[action_idx] if 0 <= action_idx < len(ACTION_NAMES) else "?"

            trajectory.append(TrajectoryStep(
                step=r.get("step", 0),
                action=action_name,
                coords=r.get("coords"),
                frame_delta_pct=r.get("frame_delta_pct", 0.0),
                level_before=r.get("level", 0),
                level_after=r.get("level_after", r.get("level", 0)),
                rationale=r.get("rationale", ""),
            ))

        if not trajectory or llm is None:
            continue

        # Consolidate
        new_physics = consolidate_attempt(
            game=game,
            level=levels,
            trajectory=trajectory,
            world_model_before=play_kwargs.get("world_model", ""),
            outcome=state,
            llm=llm,
        )

        if new_physics:
            accumulated_physics.append(f"[Attempt {attempt + 1}]\n{new_physics}")
            print(f"  Consolidated {len(new_physics.splitlines())} new rules")
        else:
            print(f"  No new physics discovered")

    # Return the best result + accumulated physics
    best = max(all_results, key=lambda r: r.get("final_levels", 0))
    best["attempts"] = len(all_results)
    best["accumulated_physics"] = accumulated_physics
    return best
