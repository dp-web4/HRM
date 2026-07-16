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
        game="toy_a", level=0,
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
    visual_change: str = ""  # what the change LOOKED LIKE (from frame_state)


# ───────────────────────────────────────────────────────────────────
# Consolidation prompt
# ───────────────────────────────────────────────────────────────────

CONSOLIDATION_PROMPT = """You just played {game} (level {level}). The attempt ended: {outcome}.

Here is what happened — each line is one action and its effect:
{trajectory_summary}

{world_model_section}

Your job: extract what you LEARNED about this game's physics.
Not strategy ("I should do X") — mechanics ("X causes Y").

Answer these slots. Leave blank if no evidence.

OBJECTS: [comma-separated list of things you can see/interact with]
ACTIONS:
  UP: [what UP does in this game]
  DOWN: [what DOWN does]
  LEFT: [what LEFT does]
  RIGHT: [what RIGHT does]
  SEL: [what SEL/launch/activate does]
  CLICK: [what CLICK does — include coordinates if they matter]
WIN: [how to win, if you can tell]
RULES:
- [action] when [condition] → [observed effect]
- [action] when [condition] → [observed effect]
FAILED: [what you tried that didn't work and why]

Only include what you have EVIDENCE for from the trajectory above.
If an action always produced the same frame change %, describe what
that change LOOKED LIKE, not just the percentage."""


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
            visual = f" ({s.visual_change})" if s.visual_change else ""
            lines.append(f"  Step {s.step}: {s.action}{coord_str} → LEVEL {s.level_before}→{s.level_after}{visual} ★")
        elif s.frame_delta_pct >= 2.0:
            coord_str = f" at ({s.coords['x']},{s.coords['y']})" if s.coords else ""
            visual = f" — {s.visual_change}" if s.visual_change else f" — {s.frame_delta_pct:.1f}% pixels changed"
            lines.append(f"  Step {s.step}: {s.action}{coord_str}{visual}")
        elif i == 0 or i == len(steps) - 1:
            coord_str = f" at ({s.coords['x']},{s.coords['y']})" if s.coords else ""
            visual = f" — {s.visual_change}" if s.visual_change else ""
            lines.append(f"  Step {s.step}: {s.action}{coord_str}{visual}")

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
    existing_wm: Optional[Any] = None,  # GameWorldModel to update in place
) -> "GameWorldModel":
    """One LLM call: trajectory → populated GameWorldModel.

    Returns a GameWorldModel with slots populated from the trajectory.
    If existing_wm is provided, merges new discoveries into it.
    """
    from sage.cognition.thalamic_router.wm_schema import GameWorldModel, CausalRule

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
        response = llm.chat(prompt, max_tokens=600)
    except Exception:
        return existing_wm or GameWorldModel(game=game, level=level)

    # Parse the structured response into a GameWorldModel
    wm = existing_wm or GameWorldModel(game=game, level=level)
    wm.revision_count += 1
    wm.discovery_source = "consolidated"

    current_section = ""
    for line in response.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("OBJECTS:"):
            objs = stripped[8:].strip().strip("[]")
            for obj in objs.split(","):
                obj = obj.strip()
                if obj and obj not in wm.objects:
                    wm.objects.append(obj)
            current_section = "objects"

        elif upper.startswith("ACTIONS:"):
            current_section = "actions"

        elif upper.startswith("WIN:"):
            win = stripped[4:].strip()
            if win and win.lower() not in ("", "unknown", "unclear"):
                wm.win_condition = win
            current_section = "win"

        elif upper.startswith("RULES:"):
            current_section = "rules"

        elif upper.startswith("FAILED:"):
            fail = stripped[7:].strip()
            if fail:
                wm.add_failure(fail)
            current_section = "failed"

        elif current_section == "actions" and ":" in stripped:
            parts = stripped.lstrip("- ").split(":", 1)
            if len(parts) == 2:
                action_name = parts[0].strip().upper()
                desc = parts[1].strip()
                if desc and desc.lower() not in ("", "unknown", "?"):
                    wm.actions[action_name] = desc

        elif current_section == "rules" and stripped.startswith("-"):
            rule_text = stripped[1:].strip()
            # Parse "action when condition → effect"
            arrow = "→" if "→" in rule_text else "->" if "->" in rule_text else None
            if arrow:
                before, effect = rule_text.split(arrow, 1)
                effect = effect.strip()
                if " when " in before:
                    action_part, condition = before.split(" when ", 1)
                else:
                    action_part = before.strip()
                    condition = "any"

                # Check if this rule already exists
                already = False
                for existing in wm.causal_rules:
                    if (existing.action.lower() == action_part.strip().lower()
                            and existing.condition.lower() == condition.strip().lower()):
                        existing.predicted_effect = effect
                        existing.evidence_count += 1
                        existing.confidence = min(1.0, existing.confidence + 0.1)
                        already = True
                        break

                if not already:
                    wm.causal_rules.append(CausalRule(
                        action=action_part.strip(),
                        condition=condition.strip(),
                        predicted_effect=effect,
                        confidence=0.5,
                        evidence_count=1,
                        source="consolidated",
                    ))

    return wm


# ───────────────────────────────────────────────────────────────────
# Multi-attempt wrapper
# ───────────────────────────────────────────────────────────────────

def play_with_consolidation(
    play_fn,           # function that plays one attempt, returns result dict
    game: str,
    max_attempts: int = 3,
    llm: Any = None,
    initial_wm: Optional[Any] = None,  # GameWorldModel to start from
    **play_kwargs,
) -> Dict[str, Any]:
    """Play a game with micro-consolidation between attempts.

    After each attempt, consolidates discovered physics into a
    GameWorldModel and feeds wm.render() as context for the next
    attempt. Compounds understanding across retries.

    play_fn signature: play_fn(game, system_prompt=..., **play_kwargs) -> dict
    The result dict must have: final_levels, final_state, llm_responses (with
    action, action_name, coords, rationale, frame_delta_pct per step).
    """
    from sage.cognition.thalamic_router.wm_schema import GameWorldModel

    wm = initial_wm or GameWorldModel(game=game, level=0)
    all_results = []

    for attempt in range(max_attempts):
        # Feed the world model as system prompt
        wm_text = wm.render() if wm.causal_rules or wm.objects else None

        if wm_text:
            print(f"\n[Attempt {attempt + 1}] World model ({len(wm.causal_rules)} rules, "
                  f"{len(wm.objects)} objects, rev {wm.revision_count}):")
            print(wm_text[:300])

        result = play_fn(game=game, system_prompt=wm_text, **play_kwargs)
        all_results.append(result)

        levels = result.get("final_levels", 0)
        state = result.get("final_state", "UNKNOWN")

        print(f"\n[Attempt {attempt + 1}/{max_attempts}] L{levels}, {state}")

        if state == "WON":
            break

        # Extract trajectory for consolidation
        responses = result.get("llm_responses", [])
        if not responses:
            continue

        trajectory = []
        ACTION_NAMES = ["A0", "UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"]
        for r in responses:
            action_name = r.get("action_name", "")
            if not action_name:
                action_idx = r.get("action", 0)
                action_name = ACTION_NAMES[action_idx] if 0 <= action_idx < len(ACTION_NAMES) else "?"

            trajectory.append(TrajectoryStep(
                step=r.get("step", 0),
                action=action_name,
                coords=r.get("coords"),
                frame_delta_pct=r.get("frame_delta_pct", 0.0),
                level_before=r.get("level", 0),
                level_after=r.get("level_after", r.get("level", 0)),
                rationale=r.get("rationale", ""),
                visual_change=r.get("visual_change", ""),
            ))

        if not trajectory or llm is None:
            continue

        # Consolidate into the world model
        wm = consolidate_attempt(
            game=game,
            level=levels,
            trajectory=trajectory,
            world_model_before=wm.render() if wm.causal_rules else "",
            outcome=state,
            llm=llm,
            existing_wm=wm,
        )

        print(f"  WM after consolidation: {len(wm.objects)} objects, "
              f"{len(wm.causal_rules)} rules, {len(wm.failed_attempts)} failures")

    # Return the best result + final world model
    best = max(all_results, key=lambda r: r.get("final_levels", 0))
    best["attempts"] = len(all_results)
    best["final_world_model"] = wm.to_dict()
    best["final_world_model_text"] = wm.render()
    return best
