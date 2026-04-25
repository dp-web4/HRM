#!/usr/bin/env python3
"""
Plan executor — runs structured plans from the Codification Project format spec.

Simple things simple: [{"do": "UP"}, {"do": "RIGHT"}]
Complex things possible: phases, expectations, conditions, resources, modes.

Plans are predictions. Each step's `expect` is checked against reality.
Expectation failure is the SNARC signal to re-plan.
"""

import copy
import json
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Action name → engine action index
ACTION_MAP = {
    "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SEL": 5, "CLICK": 6,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
}


def _get_action_index(step: Dict) -> int:
    """Extract action index from a step dict."""
    do = str(step.get("do", "")).upper()
    return ACTION_MAP.get(do, 0)


# ── Expectation checking ────────────────────────────────────────────

def check_expect(expect: str, px_diff: int, new_level: int, old_level: int,
                 state_name: str, frame: Optional[np.ndarray] = None) -> Tuple[bool, str]:
    """Check an expectation predicate against reality.

    Returns (passed, description_of_actual).
    """
    if not expect:
        return True, ""

    expect_lower = expect.lower().strip()

    # frame_change > Npx
    m = re.match(r'frame_change\s*>\s*(\d+)\s*px', expect_lower)
    if m:
        threshold = int(m.group(1))
        passed = px_diff > threshold
        return passed, f"frame_change = {px_diff}px"

    # frame_change < Npx
    m = re.match(r'frame_change\s*<\s*(\d+)\s*px', expect_lower)
    if m:
        threshold = int(m.group(1))
        passed = px_diff < threshold
        return passed, f"frame_change = {px_diff}px"

    # level_advance
    if 'level_advance' in expect_lower:
        passed = new_level > old_level
        return passed, f"level {'advanced' if passed else 'unchanged'}"

    # state = X
    m = re.match(r'state\s*=\s*(\w+)', expect_lower)
    if m:
        expected_state = m.group(1).upper()
        passed = state_name == expected_state
        return passed, f"state = {state_name}"

    # color_at(x,y) = C
    m = re.match(r'color_at\((\d+),\s*(\d+)\)\s*=\s*(\d+)', expect_lower)
    if m and frame is not None:
        x, y, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            actual = int(frame[y, x])
            return actual == c, f"color_at({x},{y}) = {actual}"

    # Unknown predicate — pass by default (don't block on parse issues)
    return True, f"unknown_predicate: {expect}"


# ── Plan normalization ──────────────────────────────────────────────

def normalize_plan(raw: Any) -> Tuple[List[Dict], Dict[str, int]]:
    """Normalize plan input into a flat list of steps + resource dict.

    Accepts:
    - List of step dicts (flat plan)
    - List of phase dicts (each with "steps" key)
    - Dict with "plan" key (+ optional "resources")
    """
    resources = {}

    if isinstance(raw, dict):
        resources = raw.get("resources", {})
        raw = raw.get("plan", raw.get("steps", []))

    flat_steps = []
    for item in raw:
        if "phase" in item and "steps" in item:
            # Phase grouping — flatten but tag each step with phase name
            phase_name = item["phase"]
            for step in item["steps"]:
                step = dict(step)  # copy
                step["_phase"] = phase_name
                flat_steps.append(step)
        elif "do" in item:
            flat_steps.append(item)
        # else: skip unrecognized items

    return flat_steps, resources


# ── Main executor ───────────────────────────────────────────────────

def execute_plan(
    env, fd,
    plan: Any,
    max_game_steps: int = 500,
) -> Dict:
    """Execute a structured plan against a game environment.

    Returns an execution result dict with per-step outcomes,
    expectation checks, and overall status.
    """
    from arcengine import GameAction
    GA = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
          4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6}

    steps, resources = normalize_plan(plan)
    resource_counters = dict(resources)  # mutable copy

    log = []
    game_step = 0
    plan_idx = 0
    repeat_count = 0
    level = fd.levels_completed
    prev_frame = np.array(fd.frame)[-1].copy()
    mode = "default"
    outcome = "completed"
    failed_step = None
    failed_expect = None
    failed_actual = None

    while game_step < max_game_steps and plan_idx < len(steps):
        if fd.state.name in ("WON", "GAME_OVER", "LOST"):
            outcome = fd.state.name.lower()
            break

        step = steps[plan_idx]
        action_idx = _get_action_index(step)

        # Check condition (if present)
        cond = step.get("if")
        if cond and log:
            last = log[-1]
            cond_passed, _ = check_expect(cond, last["px_diff"],
                                           last.get("new_level", level), level,
                                           fd.state.name)
            if not cond_passed:
                # Use else_do or else_goto
                else_do = step.get("else_do")
                else_goto = step.get("else_goto")
                if else_do:
                    action_idx = ACTION_MAP.get(else_do.upper(), action_idx)
                elif else_goto is not None:
                    plan_idx = int(else_goto)
                    repeat_count = 0
                    continue
                else:
                    # Condition failed, no alternative — skip step
                    plan_idx += 1
                    repeat_count = 0
                    continue

        # Check resource cost
        resource_name = step.get("resource")
        cost = step.get("cost", 0)
        if resource_name and cost:
            remaining = resource_counters.get(resource_name, 0)
            if remaining < cost:
                outcome = "resource_exhausted"
                failed_step = plan_idx
                break
            resource_counters[resource_name] = remaining - cost

        # Mode tracking
        if "mode_enter" in step:
            mode = step["mode_enter"]

        # Execute action
        if action_idx in GA:
            try:
                if action_idx == 6:
                    coords = {}
                    if "x" in step and "y" in step:
                        coords = {"x": step["x"], "y": step["y"]}
                    elif "target" in step:
                        # Symbolic target — would be resolved by WM object registry
                        # For now, use center as fallback
                        coords = {"x": 32, "y": 32}
                    if coords:
                        fd = env.step(GA[action_idx], data=coords)
                    else:
                        fd = env.step(GA[action_idx])
                else:
                    fd = env.step(GA[action_idx])
            except Exception:
                try:
                    fd = env.step(GA[action_idx])
                except Exception:
                    pass

        if fd is None:
            outcome = "crash"
            break

        # Measure outcome
        curr_frame = np.array(fd.frame)[-1]
        px_diff = int(np.sum(curr_frame != prev_frame))
        new_level = fd.levels_completed

        # Check expectation
        expect = step.get("expect", "")
        expect_passed, expect_actual = check_expect(
            expect, px_diff, new_level, level, fd.state.name, curr_frame
        )

        action_name = step.get("do", str(action_idx))
        phase = step.get("_phase", "")

        entry = {
            "game_step": game_step,
            "plan_step": plan_idx,
            "phase": phase,
            "action": action_name,
            "px_diff": px_diff,
            "new_level": new_level,
            "mode": mode,
            "expect": expect,
            "expect_passed": expect_passed,
            "expect_actual": expect_actual,
        }
        log.append(entry)

        # Print progress
        if game_step < 10 or game_step % 20 == 0 or not expect_passed or new_level > level:
            phase_tag = f" [{phase}]" if phase else ""
            expect_tag = f" ✓" if expect and expect_passed else (f" ✗ ({expect_actual})" if expect and not expect_passed else "")
            lvl_tag = f" ★L{new_level}" if new_level > level else ""
            print(f"  {game_step:3d}: {action_name:6s}{phase_tag} Δ{px_diff}px{expect_tag}{lvl_tag}")

        # Track expectation failure
        if expect and not expect_passed:
            if failed_step is None:  # record first failure
                failed_step = plan_idx
                failed_expect = expect
                failed_actual = expect_actual

        # Level advance
        if new_level > level:
            level = new_level
            outcome = "level_advance"
            break

        # Advance plan pointer
        repeat_target = step.get("repeat", 1)
        repeat_count += 1

        # Check goto
        then_goto = step.get("then_goto")
        if then_goto is not None and expect_passed:
            plan_idx = int(then_goto)
            repeat_count = 0
        elif repeat_count >= repeat_target:
            plan_idx += 1
            repeat_count = 0
        # else: stay on same step (repeating)

        prev_frame = curr_frame
        game_step += 1

    if game_step >= max_game_steps:
        outcome = "step_limit"

    return {
        "steps_executed": game_step,
        "steps_total": len(steps),
        "outcome": outcome,
        "final_level": fd.levels_completed if fd else 0,
        "final_state": fd.state.name if fd and hasattr(fd, 'state') and fd.state else "CRASHED",
        "failed_step": failed_step,
        "failed_expect": failed_expect,
        "failed_actual": failed_actual,
        "failed_phase": steps[failed_step].get("_phase", "") if failed_step is not None and failed_step < len(steps) else None,
        "resources_remaining": resource_counters,
        "execution_log": log,
    }


# ── Convenience: parse plan from LLM text ───────────────────────────

def parse_plan_from_text(text: str) -> List[Dict]:
    """Best-effort parse of plan from LLM text output.

    Tries JSON first, falls back to line-by-line step extraction.
    """
    # Try JSON parse
    try:
        # Find JSON array or object in the text
        for start in range(len(text)):
            if text[start] in '[{':
                bracket_count = 0
                for end in range(start, len(text)):
                    if text[end] in '[{':
                        bracket_count += 1
                    elif text[end] in ']}':
                        bracket_count -= 1
                    if bracket_count == 0:
                        candidate = text[start:end+1]
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list):
                            return parsed
                        elif isinstance(parsed, dict):
                            return [parsed]
                        break
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract STEP lines (from policy_sketch format)
    steps = []
    for line in text.split('\n'):
        m = re.search(r'STEP\s+\d+.*?action\s*=\s*(\d+)', line, re.IGNORECASE)
        if m:
            action = int(m.group(1))
            step = {"do": str(action)}
            # Extract repeat
            rm = re.search(r'repeat\s*=\s*(\d+)', line, re.IGNORECASE)
            if rm:
                step["repeat"] = int(rm.group(1))
            # Extract coords
            xm = re.search(r'x\s*=\s*(\d+)', line, re.IGNORECASE)
            ym = re.search(r'y\s*=\s*(\d+)', line, re.IGNORECASE)
            if xm and ym:
                step["x"] = int(xm.group(1))
                step["y"] = int(ym.group(1))
            # Extract expect
            em = re.search(r'expect\s*=\s*"([^"]+)"', line, re.IGNORECASE)
            if em:
                step["expect"] = em.group(1)
            steps.append(step)

    # Last fallback: extract bare action numbers
    if not steps:
        for line in text.split('\n'):
            line = line.strip()
            if line in ('1', '2', '3', '4', '5', '6'):
                names = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'SEL', 6: 'CLICK'}
                steps.append({"do": names.get(int(line), line)})

    return steps


if __name__ == "__main__":
    # Quick self-test with a simple plan
    print("Plan executor self-test")
    plan = [
        {"do": "LEFT", "repeat": 2, "expect": "frame_change > 100px"},
        {"do": "SEL", "expect": "frame_change > 30px"},
        {"do": "RIGHT", "repeat": 2},
        {"do": "SEL", "expect": "level_advance"},
    ]
    steps, resources = normalize_plan(plan)
    print(f"Normalized: {len(steps)} steps, resources: {resources}")
    for s in steps:
        print(f"  {s}")
