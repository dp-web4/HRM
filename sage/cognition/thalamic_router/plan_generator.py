#!/usr/bin/env python3
"""
Plan generator — asks the LLM to write an executable plan from WM + probes.

This is the bridge between Layer 1 (WM schema) and Layer 2 (plan executor).
The LLM's job: translate causal understanding into executable steps.
The executor's job: run those steps and verify predictions.

The plan generator:
1. Renders the WM schema as structured context
2. Runs lookahead probes to ground the plan in current reality
3. Asks the LLM to write a plan in the executor's format
4. Parses the plan (JSON or fallback text)
5. Returns a list of steps ready for plan_executor.execute_plan()

On revision (re-plan after failure):
1. Includes the previous plan + execution result
2. Shows exactly which step failed and why
3. Asks for a targeted fix, not a full rewrite
"""

import copy
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sage.cognition.thalamic_router.llm_dispatch import (
    OllamaClient, render_frame_png,
)
from sage.cognition.thalamic_router.lookahead import lookahead
from sage.cognition.thalamic_router.wm_schema import GameWorldModel
from sage.cognition.thalamic_router.plan_executor import (
    execute_plan, parse_plan_from_text, normalize_plan,
)


# ── Probe sequences for grounding ───────────────────────────────────

def _describe_change(frame0: np.ndarray, frame1: np.ndarray) -> str:
    """Describe WHERE and WHAT changed between two frames."""
    diff_mask = frame0 != frame1
    if not diff_mask.any():
        return "no change"
    ys, xs = np.where(diff_mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    n_px = int(diff_mask.sum())

    # Quadrant description
    cy, cx = (y_min + y_max) / 2, (x_min + x_max) / 2
    v = "top" if cy < 21 else ("bottom" if cy > 42 else "center")
    h = "left" if cx < 21 else ("right" if cx > 42 else "center")
    region = f"{v}-{h}" if v != "center" or h != "center" else "center"

    # Color analysis — what colors appeared/disappeared
    old_colors = set(frame0[ys, xs].flatten())
    new_colors = set(frame1[ys, xs].flatten())
    appeared = new_colors - old_colors
    disappeared = old_colors - new_colors

    parts = [f"{n_px}px in {region} (y:{y_min}-{y_max} x:{x_min}-{x_max})"]
    if appeared:
        parts.append(f"colors appeared: {sorted(appeared)[:3]}")
    if disappeared:
        parts.append(f"colors gone: {sorted(disappeared)[:3]}")
    return "; ".join(parts)


def _frame_match_pct(frame: np.ndarray, reference_frame: np.ndarray) -> float:
    """Percentage of pixels matching between current frame and reference."""
    return float(np.mean(frame == reference_frame) * 100)


def run_grounding_probes(env, fd, reference_frame: Optional[np.ndarray] = None) -> str:
    """Run enriched probes: what exists, what each element does, where it acts.

    Three layers of grounding:
    1. Directional actions — what movement does
    2. Click discovery — find all interactive elements with semantic descriptions
    3. Goal comparison — how close are we to winning (if reference available)
    """
    from arcengine import GameAction
    GA = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
          4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6}
    names = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'SEL', 6: 'CLICK'}
    frame0 = np.array(fd.frame)[-1].copy()

    lines = []

    # ── 1. Directional actions ──────────────────────────────────
    la_text, _ = lookahead(env, fd, actions=[1, 2, 3, 4, 5])
    lines.append("Single actions from current state:")
    lines.append(la_text)

    # Direction → SEL sequences
    lines.append("\nTwo-step sequences (move then SEL):")
    for d in [1, 2, 3, 4]:
        try:
            env_t = copy.deepcopy(env)
            env_t.step(GA[d])
            fd_t = env_t.step(GA[5])
            frame_t = np.array(fd_t.frame)[-1]
            diff = int(np.sum(frame_t != frame0))
            desc = _describe_change(frame0, frame_t) if diff > 10 else f"{diff}px"
            adv = fd_t.levels_completed > fd.levels_completed
            lines.append(f"  {names[d]}→SEL: {desc}{'  ★ LEVEL ADVANCE!' if adv else ''}")
        except Exception:
            pass

    # ── 2. Exhaustive click discovery with semantic descriptions ─
    click_results = []
    for cy in range(4, 60, 8):
        for cx in range(4, 60, 8):
            try:
                env_t = copy.deepcopy(env)
                fd_t = env_t.step(GA[6], data={'x': cx, 'y': cy})
                if fd_t is None:
                    continue
                frame_t = np.array(fd_t.frame)[-1]
                diff = int(np.sum(frame_t != frame0))
                if diff > 3:
                    desc = _describe_change(frame0, frame_t)
                    click_results.append((diff, cx, cy, desc, frame_t))
            except Exception:
                pass

    if click_results:
        click_results.sort(key=lambda x: -x[0])
        lines.append(f"\nActive click targets ({len(click_results)} found):")
        for diff, cx, cy, desc, _ in click_results[:10]:
            color = int(frame0[cy, cx]) if 0 <= cy < 64 and 0 <= cx < 64 else -1
            lines.append(f"  CLICK({cx},{cy}): {desc} [on color {color}]")

        # Group by effect similarity — which clicks do the SAME thing?
        # (helps model understand: "these 5 targets all cycle grid cells")
        effect_groups = {}
        for diff, cx, cy, desc, _ in click_results:
            # Group by magnitude bucket + region
            key = f"{diff // 20 * 20}px"
            effect_groups.setdefault(key, []).append((cx, cy))
        if len(effect_groups) < len(click_results):
            lines.append("\n  Effect groups (targets that do similar things):")
            for effect, coords in sorted(effect_groups.items(), key=lambda x: -int(x[0].rstrip('px'))):
                if len(coords) > 1:
                    coord_str = ", ".join(f"({x},{y})" for x, y in coords[:6])
                    lines.append(f"    ~{effect} change: {coord_str}")

        # CLICK→SEL and CLICK→CLICK on best targets
        lines.append("\nSequences from best click targets:")
        for diff, cx, cy, _, _ in click_results[:4]:
            # CLICK → SEL
            try:
                env_t = copy.deepcopy(env)
                env_t.step(GA[6], data={'x': cx, 'y': cy})
                fd_t = env_t.step(GA[5])
                frame_t = np.array(fd_t.frame)[-1]
                total_diff = int(np.sum(frame_t != frame0))
                adv = fd_t.levels_completed > fd.levels_completed
                desc = _describe_change(frame0, frame_t) if total_diff > 20 else f"{total_diff}px"
                lines.append(f"  CLICK({cx},{cy})→SEL: {desc}{'  ★ LEVEL ADVANCE!' if adv else ''}")
            except Exception:
                pass

            # CLICK same target twice (for toggle/cycle mechanics)
            try:
                env_t = copy.deepcopy(env)
                env_t.step(GA[6], data={'x': cx, 'y': cy})
                fd_t = env_t.step(GA[6], data={'x': cx, 'y': cy})
                frame_t = np.array(fd_t.frame)[-1]
                double_diff = int(np.sum(frame_t != frame0))
                if double_diff != diff:  # double-click differs from single
                    lines.append(f"  CLICK({cx},{cy})×2: {double_diff}px {'(reverts!)' if double_diff < diff // 2 else '(compounds)'}")
            except Exception:
                pass
    else:
        lines.append("\nNo active click targets found (game is directional-only).")

    # ── 3. Goal comparison ──────────────────────────────────────
    if reference_frame is not None:
        match_pct = _frame_match_pct(frame0, reference_frame)
        lines.append(f"\nGoal proximity: {match_pct:.1f}% of pixels match target")
        diff_mask = frame0 != reference_frame
        if diff_mask.any():
            ys, xs = np.where(diff_mask)
            lines.append(f"  Mismatched region: y:{ys.min()}-{ys.max()} x:{xs.min()}-{xs.max()}")

    return "\n".join(lines)


# ── Plan generation prompt ──────────────────────────────────────────

PLAN_PROMPT = """You are writing a game-playing plan. Execute this plan to clear the level.

## Game mechanics
{wm_text}

## Current state (from probes)
{probes}

## Your task
Write a JSON plan — a list of steps to clear this level.
Each step is a dict. Simple steps: {{"do": "UP"}}
Repeat: {{"do": "LEFT", "repeat": 3}}
Click: {{"do": "CLICK", "x": 36, "y": 3}}
Phases: {{"phase": "name", "steps": [...]}}
Predictions: {{"do": "SEL", "expect": "frame_change > 40px"}}

Use action NAMES: UP, DOWN, LEFT, RIGHT, SEL, CLICK
Use the probe data to choose actions that produce real change.
Include "expect" on key steps so we can verify your predictions.

Respond with ONLY the JSON plan (no other text):
[
  ...your steps here...
]"""


REVISE_PROMPT = """Your plan failed. Here's what happened:

## Previous plan
{prev_plan_json}

## Execution result
Outcome: {outcome}
Steps executed: {steps_executed}/{steps_total}
{failure_detail}

## Current state (from probes)
{probes}

## Game mechanics (reminder)
{wm_text}

## Your task
Fix the plan. The failure tells you exactly what went wrong.
Keep steps that worked. Fix or replace the step that failed.

Respond with ONLY the revised JSON plan:
[
  ...your fixed steps...
]"""


def generate_plan(
    llm: OllamaClient,
    wm: GameWorldModel,
    env, fd,
    prev_result: Optional[Dict] = None,
    prev_plan: Optional[List] = None,
    initial_frame: Optional[np.ndarray] = None,
) -> List[Dict]:
    """Generate or revise a plan using the LLM.

    If prev_result is provided, this is a revision (re-plan after failure).
    Otherwise, this is initial plan generation.
    initial_frame: the frame from game start — used for goal proximity tracking.
    """
    frame = np.array(fd.frame)[-1]
    png = render_frame_png(frame, scale=4)
    wm_text = wm.render(budget_tokens=400)
    probes = run_grounding_probes(env, fd, reference_frame=initial_frame)

    if prev_result and prev_plan:
        # Revision
        failure_detail = ""
        if prev_result.get("failed_step") is not None:
            failure_detail = (
                f"Failed at step {prev_result['failed_step']}"
                f" (phase: {prev_result.get('failed_phase', 'none')})\n"
                f"  Expected: {prev_result.get('failed_expect', '?')}\n"
                f"  Actual: {prev_result.get('failed_actual', '?')}"
            )
        else:
            failure_detail = f"Plan completed but level not cleared (outcome: {prev_result['outcome']})"

        prev_plan_json = json.dumps(prev_plan, indent=2)[:800]
        prompt = REVISE_PROMPT.format(
            prev_plan_json=prev_plan_json,
            outcome=prev_result["outcome"],
            steps_executed=prev_result["steps_executed"],
            steps_total=prev_result["steps_total"],
            failure_detail=failure_detail,
            probes=probes,
            wm_text=wm_text,
        )
    else:
        # Initial generation
        prompt = PLAN_PROMPT.format(wm_text=wm_text, probes=probes)

    t0 = time.time()
    response = llm.chat(prompt, images_png=[png])
    elapsed = time.time() - t0

    plan = parse_plan_from_text(response)
    print(f"[PLAN] Generated {len(plan)} steps in {elapsed:.1f}s")
    return plan


# ── Main loop: generate → execute → revise ──────────────────────────

def play_with_plans(
    game: str,
    game_id: str,
    wm: GameWorldModel,
    llm_model: str = "gemma4:26b",
    max_steps: int = 500,
    max_revisions: int = 5,
) -> Dict:
    """Play a game using the plan generation → execution → revision loop."""
    from arc_agi import Arcade

    arcade = Arcade(operation_mode='offline')
    env = arcade.make(game_id)
    if env is None:
        for ei in arcade.get_environments():
            if game in ei.game_id:
                env = arcade.make(ei.game_id)
                game_id = ei.game_id
                break
    if env is None:
        return {"error": f"Game {game_id} not found"}

    fd = env.reset()
    llm = OllamaClient(model=llm_model)
    initial_frame = np.array(fd.frame)[-1].copy()  # reference for goal proximity

    total_steps = 0
    total_log = []
    revisions = []
    level = 0

    print(f"PLAN DISPATCH: {game_id}, max {max_steps} steps, {max_revisions} revisions")
    print(f"WM: {wm.game} L{wm.level}, {len(wm.objects)} objects, {len(wm.causal_rules)} rules")

    # Initial plan
    plan = generate_plan(llm, wm, env, fd, initial_frame=initial_frame)
    if not plan:
        return {"error": "Failed to generate initial plan"}
    revisions.append({"step": 0, "type": "initial", "plan_len": len(plan)})

    for rev in range(max_revisions + 1):
        if fd.state.name in ("WON", "GAME_OVER", "LOST"):
            break
        if total_steps >= max_steps:
            break

        # Execute
        budget = min(max_steps - total_steps, 200)  # per-plan budget
        result = execute_plan(env, fd, plan, max_game_steps=budget)
        total_steps += result["steps_executed"]
        total_log.extend(result["execution_log"])

        # Get current fd from the executor's result
        # execute_plan steps env in-place; we need a fresh fd reference
        fd = result.get("_fd", fd)  # executor passes this back

        if result["outcome"] == "level_advance":
            print(f"[PLAY] ★ LEVEL {level} → {result['final_level']} at step {total_steps}")
            level = result["final_level"]
            # Update WM for new level
            wm.level = level
            wm.revision_count += 1
            # Generate fresh plan for new level
            plan = generate_plan(llm, wm, env, fd, initial_frame=initial_frame)
            revisions.append({"step": total_steps, "type": "level_advance", "plan_len": len(plan)})
            continue

        if result["outcome"] in ("game_over", "crash", "won"):
            break

        # Revise
        if rev < max_revisions:
            print(f"[PLAY] Revising (rev {rev+1}/{max_revisions}), outcome={result['outcome']}")

            # Update WM with observations from execution
            for entry in result["execution_log"]:
                if entry["px_diff"] > 10:
                    wm.observe(
                        action=entry["action"],
                        condition=f"step {entry['game_step']}, phase {entry.get('phase', '')}",
                        actual_effect=f"{entry['px_diff']}px change",
                        frame_delta_pct=entry["px_diff"] / 4096 * 100,
                    )
                if not entry.get("expect_passed", True):
                    wm.add_failure(
                        f"{entry['action']} expected '{entry.get('expect','')}' "
                        f"but got '{entry.get('expect_actual','')}'"
                    )

            plan = generate_plan(llm, wm, env, fd, prev_result=result, prev_plan=plan, initial_frame=initial_frame)
            revisions.append({"step": total_steps, "type": "revision", "plan_len": len(plan)})

    final_state = fd.state.name if fd and hasattr(fd, 'state') and fd.state else "CRASHED"
    final_levels = fd.levels_completed if fd and hasattr(fd, 'levels_completed') else 0

    return {
        "game": game, "game_id": game_id,
        "dispatch": "plan_generator",
        "n_steps": total_steps,
        "final_state": final_state,
        "final_levels": final_levels,
        "revision_count": len(revisions) - 1,
        "revisions": revisions,
        "wm_rules_discovered": len(wm.causal_rules),
        "wm_failures_recorded": len(wm.failed_attempts),
        "execution_log_len": len(total_log),
    }


# ── Load WM from existing world model files ─────────────────────────

def load_wm_from_prose(game: str) -> GameWorldModel:
    """Load a GameWorldModel by parsing existing prose world model files.

    Falls back to an empty WM if no file is found.
    """
    for root in [
        Path(os.environ.get("SHARED_CONTEXT_DIR", "")),
        Path.home() / "ai-workspace" / "shared-context",
    ]:
        wm_path = root / "arc-agi-3" / "world-models" / f"{game}.md"
        if wm_path.exists():
            text = wm_path.read_text()
            return _parse_prose_wm(game, text)

    return GameWorldModel(game=game, discovery_source="empty")


def _parse_prose_wm(game: str, text: str) -> GameWorldModel:
    """Parse a prose world model .md file into a GameWorldModel."""
    sections = {}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line.strip()
        elif current:
            sections.setdefault(current, []).append(line)

    objects = []
    for line in sections.get("## Objects", []):
        stripped = line.strip()
        if stripped.startswith("- **"):
            # Extract object name from "- **Name**: description"
            name_end = stripped.find("**", 4)
            if name_end > 0:
                objects.append(stripped[4:name_end] + stripped[name_end+2:])

    actions = {}
    for line in sections.get("## Rules", []):
        stripped = line.strip()
        if stripped.startswith("- ACTION") or stripped.startswith("- LAUNCH") or stripped.startswith("- CLICK"):
            # Try to extract action name and description
            for aname in ["UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK", "LAUNCH", "ACTION5", "ACTION1-4"]:
                if aname in stripped.upper():
                    actions[aname.replace("LAUNCH", "SEL").replace("ACTION5", "SEL").replace("ACTION1-4", "directional")] = stripped.lstrip("- ")
                    break

    win_condition = " ".join(sections.get("## Win Condition", [])).strip()[:200]

    from sage.cognition.thalamic_router.wm_schema import CausalRule

    causal_rules = []
    for line in sections.get("## Rules", []):
        stripped = line.strip()
        if stripped.startswith("- ") and any(kw in stripped for kw in ["→", "produces", "paints", "moves", "triggers"]):
            causal_rules.append(CausalRule(
                action="various",
                condition="general",
                predicted_effect=stripped.lstrip("- ")[:100],
                confidence=0.7,
                source="human",
            ))

    strategy = " ".join(sections.get("## Strategy", [])).strip()[:200]
    return GameWorldModel(
        game=game,
        objects=objects,
        actions=actions,
        win_condition=win_condition,
        causal_rules=causal_rules,
        current_strategy=strategy,
        discovery_source="prose_wm",
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True)
    p.add_argument("--game-id", default=None)
    p.add_argument("--llm-model", default="gemma4:26b")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--max-revisions", type=int, default=5)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    os.environ.setdefault("ARC_SAGE_DIR", os.path.expanduser("~/ai-workspace/arc-sage"))
    os.environ.setdefault("SHARED_CONTEXT_DIR", os.path.expanduser("~/ai-workspace/shared-context"))

    wm = load_wm_from_prose(args.game)
    print(f"Loaded WM: {wm.game}, {len(wm.objects)} objects, {len(wm.causal_rules)} rules")
    print(f"  Objects: {wm.objects[:3]}")
    print(f"  Win: {wm.win_condition[:80]}")

    result = play_with_plans(
        game=args.game,
        game_id=args.game_id or args.game,
        wm=wm,
        llm_model=args.llm_model,
        max_steps=args.max_steps,
        max_revisions=args.max_revisions,
    )

    print(f"\nResult: L{result.get('final_levels', 0)} {result.get('final_state', '?')} "
          f"({result.get('n_steps', 0)} steps, {result.get('revision_count', 0)} revisions)")
    print(f"WM: {result.get('wm_rules_discovered', 0)} rules discovered, "
          f"{result.get('wm_failures_recorded', 0)} failures recorded")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Wrote: {args.json_out}")
