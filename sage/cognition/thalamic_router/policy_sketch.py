#!/usr/bin/env python3
"""
Policy sketch dispatch — the model writes rules, executor runs them.

The model's natural register for causal understanding is writing code.
Policy sketches are tiny interpreted programs: observe → reason → encode → execute → revise.

Architecture:
  Phase 1 (Sketch): LLM sees frame + lookahead, writes a numbered step sequence
  Phase 2 (Execute): Step executor runs the sequence, tracking outcomes
  Phase 3 (Revise): When stuck or step fails, LLM sees what happened and rewrites

This is solver-writing compressed to game-time scale.
"""

import os
import re
import sys
import time
import json
import copy
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sage.cognition.thalamic_router.llm_dispatch import (
    OllamaClient, render_frame_pair_png, render_frame_png,
    N_ACTIONS, ACTION_NAMES, FRAME_H, FRAME_W,
)
from sage.cognition.thalamic_router.lookahead import lookahead

ACTION_KEY = "1=UP 2=DOWN 3=LEFT 4=RIGHT 5=SEL 6=CLICK"

# ── Phase 1: Sketch ─────────────────────────────────────────────────

SKETCH_PROMPT = """Look at this game frame. You need to write a POLICY — a numbered sequence of steps to play this game.

Actions: {action_key}

What each action does RIGHT NOW from the current state:
{lookahead}

Write a NUMBERED SEQUENCE of 4-6 steps. Each step says:
- Which action to do (by number)
- How many times to do it (or what to watch for)
- When to move to the next step

Format each step EXACTLY like this:
STEP N: action=X repeat=Y next_if=<condition>

Conditions: "frame_changed", "no_change_3x", "done"
repeat: a number (how many times to do this action before moving on)

Example:
STEP 1: action=3 repeat=3 next_if=frame_changed
STEP 2: action=5 repeat=1 next_if=done
STEP 3: action=4 repeat=3 next_if=frame_changed
STEP 4: action=5 repeat=1 next_if=done

Think about what this game IS and what sequence of actions would make progress. Be specific."""


def generate_sketch(llm: OllamaClient, frame: np.ndarray, la_text: str) -> List[Dict]:
    """Ask LLM to write a policy sketch. Returns list of step dicts."""
    png = render_frame_png(frame, scale=4)
    prompt = SKETCH_PROMPT.format(action_key=ACTION_KEY, lookahead=la_text)

    t0 = time.time()
    response = llm.chat(prompt, images_png=[png])
    elapsed = time.time() - t0

    steps = parse_sketch(response)
    print(f"[SKETCH] Generated {len(steps)} steps in {elapsed:.1f}s")
    for s in steps:
        print(f"  STEP {s['step']}: action={s['action']} repeat={s['repeat']} next_if={s['next_if']}")
    return steps


def parse_sketch(text: str) -> List[Dict]:
    """Parse STEP N: action=X repeat=Y next_if=Z from LLM response."""
    steps = []
    for line in text.split('\n'):
        m = re.search(r'STEP\s+(\d+).*?action\s*=\s*(\d+).*?repeat\s*=\s*(\d+).*?next_if\s*=\s*(\w+)', line, re.IGNORECASE)
        if m:
            steps.append({
                "step": int(m.group(1)),
                "action": int(m.group(2)),
                "repeat": int(m.group(3)),
                "next_if": m.group(4).lower(),
            })
    # Fallback: if no steps parsed, make a generic explore policy
    if not steps:
        print("[SKETCH] Parse failed — using fallback explore policy")
        steps = [
            {"step": 1, "action": 3, "repeat": 3, "next_if": "frame_changed"},
            {"step": 2, "action": 5, "repeat": 1, "next_if": "done"},
            {"step": 3, "action": 4, "repeat": 3, "next_if": "frame_changed"},
            {"step": 4, "action": 5, "repeat": 1, "next_if": "done"},
            {"step": 5, "action": 1, "repeat": 2, "next_if": "frame_changed"},
            {"step": 6, "action": 5, "repeat": 1, "next_if": "done"},
        ]
    return steps


# ── Phase 2: Execute ────────────────────────────────────────────────

def execute_sketch(
    env, fd, steps: List[Dict], max_game_steps: int = 200,
) -> Tuple[Any, List[Dict]]:
    """Execute a policy sketch, returning (final_fd, execution_log)."""
    from arcengine import GameAction
    GA = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
          4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6}

    log = []
    game_step = 0
    policy_idx = 0
    repeat_count = 0
    no_change_count = 0
    prev_frame = np.array(fd.frame)[-1].copy()
    level = fd.levels_completed

    while game_step < max_game_steps and policy_idx < len(steps):
        if fd.state.name in ("WON", "GAME_OVER", "LOST"):
            break

        # Check for level advance
        if fd.levels_completed > level:
            print(f"  [EXEC] LEVEL {level} → {fd.levels_completed} at game step {game_step}")
            level = fd.levels_completed

        rule = steps[policy_idx]
        action = rule["action"]

        # Execute
        if action in GA:
            try:
                fd = env.step(GA[action])
            except Exception:
                pass
        if fd is None:
            break

        # Measure outcome
        curr_frame = np.array(fd.frame)[-1]
        px_diff = int(np.sum(curr_frame != prev_frame))
        changed = px_diff > 10

        log.append({
            "game_step": game_step,
            "policy_step": rule["step"],
            "action": action,
            "action_name": ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action),
            "px_diff": px_diff,
            "changed": changed,
            "level": fd.levels_completed,
        })

        if game_step < 10 or game_step % 20 == 0:
            aname = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
            print(f"  {game_step:3d}: {aname} (S{rule['step']}) Δ{px_diff}px L{fd.levels_completed}")

        # Advance policy state
        repeat_count += 1
        if changed:
            no_change_count = 0
        else:
            no_change_count += 1

        advance = False
        if rule["next_if"] == "frame_changed" and changed:
            advance = True
        elif rule["next_if"] == "no_change_3x" and no_change_count >= 3:
            advance = True
        elif rule["next_if"] == "done":
            advance = True
        elif repeat_count >= rule["repeat"]:
            advance = True

        if advance:
            policy_idx += 1
            repeat_count = 0
            no_change_count = 0
            # Wrap around if at end of policy
            if policy_idx >= len(steps):
                policy_idx = 0

        prev_frame = curr_frame
        game_step += 1

    return fd, log


# ── Phase 3: Revise ─────────────────────────────────────────────────

REVISE_PROMPT = """Your policy sketch produced these results:

{execution_summary}

The game is at level {level}, step {step}. {status}.

Here is what each action does NOW:
{lookahead}

Your previous policy was:
{previous_policy}

Write a REVISED policy. Keep what worked, change what didn't.
Same format: STEP N: action=X repeat=Y next_if=<condition>
Conditions: "frame_changed", "no_change_3x", "done"
"""


def generate_revision(
    llm: OllamaClient, frame: np.ndarray, la_text: str,
    prev_steps: List[Dict], exec_log: List[Dict],
    level: int, game_step: int, status: str,
) -> List[Dict]:
    """Revise policy based on execution results."""
    # Summarize execution
    action_counts = Counter()
    total_change = 0
    for entry in exec_log:
        action_counts[entry["action_name"]] += 1
        total_change += entry["px_diff"]

    productive = sum(1 for e in exec_log if e["changed"])
    summary = (
        f"Actions taken: {dict(action_counts)}\n"
        f"Productive steps (frame changed): {productive}/{len(exec_log)}\n"
        f"Total pixel change: {total_change}\n"
        f"Levels completed: {exec_log[-1]['level'] if exec_log else 0}"
    )

    prev_policy = "\n".join(
        f"STEP {s['step']}: action={s['action']} repeat={s['repeat']} next_if={s['next_if']}"
        for s in prev_steps
    )

    png = render_frame_png(frame, scale=4)
    prompt = REVISE_PROMPT.format(
        execution_summary=summary,
        level=level,
        step=game_step,
        status=status,
        lookahead=la_text,
        previous_policy=prev_policy,
    )

    t0 = time.time()
    response = llm.chat(prompt, images_png=[png])
    elapsed = time.time() - t0

    steps = parse_sketch(response)
    print(f"[REVISE] Generated {len(steps)} revised steps in {elapsed:.1f}s")
    for s in steps:
        print(f"  STEP {s['step']}: action={s['action']} repeat={s['repeat']} next_if={s['next_if']}")
    return steps


# ── Main loop ───────────────────────────────────────────────────────

def play_policy_sketch(
    game: str, game_id: str,
    llm_model: str = "gemma4:26b",
    max_steps: int = 200,
    max_revisions: int = 3,
    steps_per_sketch: int = 50,
) -> Dict:
    """Play a game using policy sketch dispatch."""
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

    total_step = 0
    total_log = []
    revisions = []
    level = 0

    print(f"POLICY SKETCH: {game_id}, max {max_steps} steps, {max_revisions} revisions")

    # Phase 1: Initial sketch
    frame = np.array(fd.frame)[-1]
    la_text, _ = lookahead(env, fd, actions=[1, 2, 3, 4, 5])
    policy = generate_sketch(llm, frame, la_text)
    revisions.append({"step": 0, "type": "initial", "policy": policy})

    revision_count = 0
    while total_step < max_steps and revision_count <= max_revisions:
        if fd.state.name in ("WON", "GAME_OVER", "LOST"):
            break

        # Phase 2: Execute current sketch
        budget = min(steps_per_sketch, max_steps - total_step)
        fd, exec_log = execute_sketch(env, fd, policy, max_game_steps=budget)
        total_log.extend(exec_log)
        total_step += len(exec_log)

        if fd is None or fd.state.name in ("WON", "GAME_OVER", "LOST"):
            break

        # Check if we made progress
        new_level = fd.levels_completed
        if new_level > level:
            print(f"[MAIN] LEVEL ADVANCE {level} → {new_level}")
            level = new_level
            # Re-sketch for new level
            frame = np.array(fd.frame)[-1]
            la_text, _ = lookahead(env, fd, actions=[1, 2, 3, 4, 5])
            policy = generate_sketch(llm, frame, la_text)
            revisions.append({"step": total_step, "type": "level_advance", "policy": policy})
            continue

        # Phase 3: Always revise after each chunk — the policy needs to learn
        # from what it just did. No level cleared = strategy needs work.
        revision_count += 1
        productive = sum(1 for e in exec_log if e["changed"])
        productive_rate = productive / max(len(exec_log), 1)
        print(f"[MAIN] Revising ({productive_rate:.0%} productive, L{level}), revision {revision_count}/{max_revisions}")
        frame = np.array(fd.frame)[-1]
        la_text, _ = lookahead(env, fd, actions=[1, 2, 3, 4, 5])
        policy = generate_revision(
            llm, frame, la_text, policy, exec_log,
            level=level, game_step=total_step,
            status=f"Still L{level} after {total_step} steps. {productive_rate:.0%} of actions changed the frame but no level cleared. Your policy needs to do something fundamentally different.",
        )
        revisions.append({"step": total_step, "type": "revision", "policy": policy})

    final_state = fd.state.name if fd and hasattr(fd, 'state') and fd.state else "CRASHED"
    final_levels = fd.levels_completed if fd and hasattr(fd, 'levels_completed') else 0

    # Action distribution
    action_counts = Counter(e["action_name"] for e in total_log)
    productive_total = sum(1 for e in total_log if e["changed"])

    result = {
        "game": game, "game_id": game_id,
        "dispatch": "policy_sketch",
        "n_steps": total_step,
        "final_state": final_state,
        "final_levels": final_levels,
        "action_counts": dict(action_counts),
        "productive_steps": productive_total,
        "productivity_rate": productive_total / max(total_step, 1),
        "revision_count": revision_count,
        "revisions": [{"step": r["step"], "type": r["type"]} for r in revisions],
    }

    print(f"\nResult: L{final_levels}, {total_step} steps, {final_state}")
    print(f"Actions: {dict(action_counts)}")
    print(f"Productive: {productive_total}/{total_step} ({result['productivity_rate']:.0%})")
    print(f"Revisions: {revision_count}")

    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True)
    p.add_argument("--game-id", default=None)
    p.add_argument("--llm-model", default="gemma4:26b")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--max-revisions", type=int, default=3)
    p.add_argument("--steps-per-sketch", type=int, default=50)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    os.environ.setdefault("ARC_SAGE_DIR", os.path.expanduser("~/ai-workspace/arc-sage"))

    game_id = args.game_id or args.game
    result = play_policy_sketch(
        game=args.game, game_id=game_id,
        llm_model=args.llm_model,
        max_steps=args.max_steps,
        max_revisions=args.max_revisions,
        steps_per_sketch=args.steps_per_sketch,
    )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Wrote: {args.json_out}")
