#!/usr/bin/env python3
"""
SAGE Puzzle Solver v5 — Fixes the understanding→action gap.

Root causes addressed (from 2026-04-04 analysis):
1. NAMING: Objects named "color_ID" unambiguously. Parser requires exact match.
2. INTERACTIVE-FIRST: Plan prompt forces known-interactive objects first.
3. INTERLEAVED: Execute 2 actions, observe, replan. Not batch-then-observe.
4. MOVEMENT: Explicitly asks "how should you move?" when movement available.
5. SCOPE: "Next 3 actions" not "complete solution."

Architecture: PROBE → STRATEGIZE → (ACT 2 → OBSERVE → REPLAN) loop

Usage:
    cd /Users/dennispalatov/repos/SAGE
    .venv/bin/python3 arc-agi-3/experiments/sage_solver_v5.py --game lp85 -v
    .venv/bin/python3 arc-agi-3/experiments/sage_solver_v5.py --all --attempts 5
"""

import sys, os, time, json, re, random
import numpy as np
import requests

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, background_color, color_name, find_color_regions
from arc_spatial import SpatialTracker
from arc_action_model import ActionEffectModel

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "")

def _detect_model():
    """Auto-detect best available Ollama model."""
    global MODEL
    if MODEL:
        return
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        models = []
    for preferred in ["gemma4:e4b", "gemma3:12b", "phi4:14b", "qwen3.5:0.8b",
                       "qwen2.5:3b", "qwen3.5:2b"]:
        if preferred in models:
            MODEL = preferred
            return
    chat_models = [m for m in models if "embed" not in m]
    MODEL = chat_models[0] if chat_models else "gemma4:e4b"

ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                5: "SELECT", 6: "CLICK", 7: "UNDO"}


# ─── LLM ───

def _is_thinking_model():
    """Check if current model supports native thinking (gemma4, etc.)."""
    return "gemma4" in MODEL

def ask_llm(prompt: str, max_tokens: int = -1) -> str:
    """Chat API call. Works with thinking models and small models alike."""
    opts = {"temperature": 0.3}
    # Cap tokens for small models to avoid runaway generation
    if max_tokens == -1 and any(s in MODEL for s in ["0.8b", "0.5b", "1b", "2b", "3b"]):
        opts["num_predict"] = 300
    elif max_tokens > 0:
        opts["num_predict"] = max_tokens

    payload = {
        "model": MODEL, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": opts,
    }
    # Disable thinking for non-thinking models (qwen3.5 generates huge CoT otherwise)
    if not _is_thinking_model():
        payload["think"] = False

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            return content if content else data.get("message", {}).get("thinking", "").strip()
    except Exception as e:
        return f"[error: {e}]"
    return ""


# ─── NAMING: Unambiguous object names ───

def obj_name(obj):
    """Unambiguous name: color_ID. Parser matches exactly."""
    return f"{color_name(obj.color)}_{obj.id}"


def obj_catalog(tracker):
    """Build object catalog with clear status markers."""
    lines = []
    # Interactive FIRST (this is what the model should use)
    interactive = tracker.get_interactive_objects()
    if interactive:
        lines.append("INTERACTIVE (these work — use them):")
        for o in interactive:
            lines.append(f"  {obj_name(o)} ({o.w}x{o.h}) at ({o.cx},{o.cy}) — {o.click_effectiveness:.0%} effective")

    # Static (don't waste clicks)
    static = tracker.get_static_objects()
    if static:
        lines.append(f"NO EFFECT ({len(static)} objects — do NOT click these):")
        for o in static[:3]:
            lines.append(f"  {obj_name(o)}")
        if len(static) > 3:
            lines.append(f"  ... and {len(static)-3} more")

    # Untested (explore if interactive exhausted)
    untested = tracker.get_untested_objects()
    if untested:
        # Group by size — small objects more likely buttons
        small = [o for o in untested if o.size <= 64]
        large = [o for o in untested if o.size > 64]
        if small:
            lines.append(f"UNTESTED SMALL ({len(small)} — possible buttons):")
            for o in sorted(small, key=lambda x: x.size)[:5]:
                lines.append(f"  {obj_name(o)} ({o.w}x{o.h}) at ({o.cx},{o.cy})")
        if large:
            lines.append(f"UNTESTED LARGE ({len(large)} — likely background/labels)")

    return "\n".join(lines)


# ─── PROBE ───

def probe(env, fd, grid, available, budget=18):
    """Smart probing: each non-click action 2x, then click each unique-size object."""
    model = ActionEffectModel()
    tracker = SpatialTracker()
    tracker.update(grid)
    steps = 0
    start_levels = fd.levels_completed
    level_ups = 0

    # Phase 1: Try non-click actions 2x each
    for action in [a for a in available if a != 6]:
        for _ in range(2):
            if steps >= budget:
                break
            prev = grid.copy()
            try:
                fd = env.step(INT_TO_GAME_ACTION[action])
            except:
                continue
            if fd is None:
                return model, tracker, fd, grid, steps, level_ups
            grid = get_frame(fd)
            steps += 1
            diff = tracker.update(grid)
            model.observe(action, prev, grid, diff)
            if fd.levels_completed > start_levels + level_ups:
                level_ups = fd.levels_completed - start_levels
            new_avail = [a.value if hasattr(a, "value") else int(a) for a in (fd.available_actions or [])]
            if new_avail:
                available = new_avail
            if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                return model, tracker, fd, grid, steps, level_ups

    # Phase 2: Click one object per unique (size, color)
    # Interleave small and large — buttons can be any size
    if 6 in available:
        regions = find_color_regions(grid, min_size=4)
        by_size = sorted(regions, key=lambda x: x["size"])
        # Interleave: small, large, small, large... covers full range quickly
        interleaved = []
        left, right = 0, len(by_size) - 1
        while left <= right:
            interleaved.append(by_size[left])
            if left != right:
                interleaved.append(by_size[right])
            left += 1
            right -= 1
        seen = set()
        for r in interleaved:
            key = (r["color"], r["w"], r["h"])
            if key in seen:
                continue
            seen.add(key)
            if steps >= budget:
                break
            prev = grid.copy()
            data = {'x': r["cx"], 'y': r["cy"]}
            try:
                fd = env.step(GameAction.ACTION6, data=data)
            except:
                continue
            if fd is None:
                return model, tracker, fd, grid, steps, level_ups
            grid = get_frame(fd)
            steps += 1
            diff = tracker.update(grid)
            model.observe(6, prev, grid, diff)
            changed = not np.array_equal(prev, grid)
            n_px = int(np.sum(prev != grid))
            tracker.record_click(data['x'], data['y'], changed, n_px)
            if fd.levels_completed > start_levels + level_ups:
                level_ups = fd.levels_completed - start_levels
            new_avail = [a.value if hasattr(a, "value") else int(a) for a in (fd.available_actions or [])]
            if new_avail:
                available = new_avail
            if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                break

    return model, tracker, fd, grid, steps, level_ups


# ─── STRATEGIZE ───

def strategize(action_model, tracker, grid, available, levels, win_levels):
    """LLM: What's the puzzle? What's the goal?"""
    catalog = obj_catalog(tracker)
    model_desc = action_model.describe()
    game_type = action_model.infer_game_type()
    has_move = any(a in available for a in [1, 2, 3, 4])
    has_click = 6 in available

    prompt = f"""Analyze this puzzle game.

WHAT EACH ACTION DOES (tested):
{model_desc}
Inferred type: {game_type}

OBJECTS ON GRID:
{catalog}

{"MOVEMENT is available and effective." if has_move else "NO movement — click only."}
{"CLICK is available." if has_click else ""}

Level: {levels}/{win_levels}

What type of puzzle is this? What is the win condition? What objects matter?
Reply as JSON: {{"puzzle_type": "...", "win_condition": "...", "key_objects": ["name_id", ...], "strategy": "..."}}"""

    return ask_llm(prompt)


# ─── PLAN (scoped: next 3 actions, not complete solution) ───

def plan_next(strategy, catalog, action_model, available, levels, win_levels,
              last_actions=None, last_results=None):
    """LLM: What are the NEXT 3 actions? Not the complete solution."""
    has_move = any(a in available for a in [1, 2, 3, 4])

    feedback = ""
    if last_actions and last_results:
        feedback = "\nLAST ACTIONS AND RESULTS:\n"
        for act, res in zip(last_actions[-3:], last_results[-3:]):
            feedback += f"  {act}: {res}\n"
        feedback += "Learn from these results.\n"

    movement_hint = ""
    if has_move:
        movement_hint = "\nYou can MOVE (UP/DOWN/LEFT/RIGHT). Consider positioning before clicking."

    prompt = f"""You are solving a puzzle. Your analysis:
{strategy[:300]}

OBJECTS:
{catalog}
{movement_hint}
{feedback}
Level: {levels}/{win_levels}

RULES:
- Use INTERACTIVE objects FIRST. They are proven to work.
- Do NOT click NO-EFFECT objects.
- Use exact object names: e.g., "CLICK cyan_23" (not "CLICK cyan")
- "CLICK ALL cyan" clicks every cyan object.
- "REPEAT 5 CLICK cyan_23" clicks it 5 times.

What are the NEXT 3 actions? (Not the full solution — just the next 3.)

ACTIONS:"""

    return ask_llm(prompt)


# ─── PARSE ───

def parse_actions(plan_text, tracker, available):
    """Parse semantic plan to (action, data) pairs. Exact ID matching."""
    actions = []
    # Build name→object lookup
    name_map = {}
    for obj in tracker.objects.values():
        name_map[obj_name(obj).upper()] = obj
        # Also map just color name for "CLICK ALL color"
        cname = color_name(obj.color).upper()
        if cname not in name_map:
            name_map[cname] = []
        if isinstance(name_map.get(cname), list):
            name_map[cname].append(obj)

    for line in plan_text.split("\n"):
        line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
        upper = line.upper().strip()
        if not upper:
            continue

        # REPEAT N <action>
        m = re.match(r'REPEAT\s+(\d+)\s+(.+)', upper)
        if m:
            n = min(int(m.group(1)), 30)  # cap at 30
            sub = parse_actions(m.group(2), tracker, available)
            actions.extend(sub * n)
            continue

        # CLICK ALL <color>
        m = re.match(r'CLICK\s+ALL\s+(\w+)', upper)
        if m and 6 in available:
            cname = m.group(1)
            objs = name_map.get(cname, [])
            if isinstance(objs, list):
                for o in objs:
                    actions.append((6, {'x': o.cx, 'y': o.cy}))
            continue

        # CLICK INTERACTIVE
        if 'CLICK INTERACTIVE' in upper and 6 in available:
            for o in tracker.get_interactive_objects():
                actions.append((6, {'x': o.cx, 'y': o.cy}))
            continue

        # CLICK <name_id> — exact match
        m = re.match(r'CLICK\s+(\w+_\d+)', upper)
        if m and 6 in available:
            name = m.group(1)
            obj = name_map.get(name)
            if obj and not isinstance(obj, list):
                actions.append((6, {'x': obj.cx, 'y': obj.cy}))
            continue

        # CLICK <color> — first match of that color
        m = re.match(r'CLICK\s+(\w+)', upper)
        if m and 6 in available:
            cname = m.group(1)
            objs = name_map.get(cname, [])
            if isinstance(objs, list) and objs:
                o = objs[0]
                actions.append((6, {'x': o.cx, 'y': o.cy}))
            continue

        # Directional
        for name, num in [("UP", 1), ("DOWN", 2), ("LEFT", 3), ("RIGHT", 4)]:
            if name in upper and num in available:
                actions.append((num, None))
                break
        else:
            if any(w in upper for w in ["SELECT", "SUBMIT"]) and 5 in available:
                actions.append((5, None))
            elif "UNDO" in upper and 7 in available:
                actions.append((7, None))

    return actions


# ─── EXECUTE 2, OBSERVE, REPLAN loop ───

def solve_game(arcade, game_id, max_attempts=5, budget=300, verbose=False):
    """Solve with interleaved execution: act 2, observe, replan."""
    prefix = game_id.split("-")[0]
    best_levels = 0

    for attempt in range(max_attempts):
        env = arcade.make(game_id)
        fd = env.reset()
        grid = get_frame(fd)
        available = [a.value if hasattr(a, "value") else int(a) for a in (fd.available_actions or [])]
        total_steps = 0

        if verbose:
            print(f"\n  Attempt {attempt+1}/{max_attempts} | Actions: {available} | Levels: 0/{fd.win_levels}")

        # PROBE
        probe_budget = min(budget // 3, 30)
        action_model, tracker, fd, grid, probe_steps, probe_levels = \
            probe(env, fd, grid, available, budget=probe_budget)
        total_steps += probe_steps

        if verbose:
            interactive = tracker.get_interactive_objects()
            static = tracker.get_static_objects()
            untested = tracker.get_untested_objects()
            print(f"  Probe: {probe_steps} steps | {len(interactive)} interactive, {len(static)} static, {len(untested)} untested")
            for o in interactive:
                print(f"    ✓ {obj_name(o)} at ({o.cx},{o.cy}) — {o.click_effectiveness:.0%}")
            if not interactive and verbose:
                # Show what was clicked and what happened
                clicked = [o for o in tracker.objects.values() if o.click_responses]
                for o in clicked:
                    hits = sum(1 for r in o.click_responses if r["changed"])
                    print(f"    ? {obj_name(o)} at ({o.cx},{o.cy}): {hits}/{len(o.click_responses)} clicks caused change")

        if fd is None or fd.state.name in ("WON", "LOST", "GAME_OVER"):
            if fd and fd.levels_completed > best_levels:
                best_levels = fd.levels_completed
            continue

        # STRATEGIZE (once per attempt)
        t0 = time.time()
        strategy = strategize(action_model, tracker, grid, available,
                              fd.levels_completed, fd.win_levels)
        if verbose:
            print(f"  Strategy ({time.time()-t0:.0f}s): {strategy[:120]}...")

        # INTERLEAVED LOOP: plan 3, execute 2-3, observe, replan
        remaining = budget - total_steps
        action_log = []
        result_log = []
        replan_count = 0

        while remaining > 0 and replan_count < 15 and fd.state.name not in ("WON", "LOST", "GAME_OVER"):
            replan_count += 1

            # Update catalog for current state
            catalog = obj_catalog(tracker)

            # PLAN next 3 actions
            t0 = time.time()
            plan_text = plan_next(strategy, catalog, action_model, available,
                                  fd.levels_completed, fd.win_levels,
                                  last_actions=action_log, last_results=result_log)
            plan_actions = parse_actions(plan_text, tracker, available)
            plan_time = time.time() - t0

            if not plan_actions:
                # Fallback: click interactive, or random
                interactive = tracker.get_interactive_objects()
                if interactive:
                    o = random.choice(interactive)
                    plan_actions = [(6, {'x': o.cx, 'y': o.cy})]
                else:
                    a = random.choice(available)
                    if a == 6:
                        bg = background_color(grid)
                        non_bg = np.argwhere(grid != bg)
                        if len(non_bg) > 0:
                            r, c = random.choice(non_bg.tolist())
                            plan_actions = [(6, {'x': int(c), 'y': int(r)})]
                    if not plan_actions:
                        plan_actions = [(random.choice(available), None)]

            if verbose:
                names = []
                for a, d in plan_actions[:5]:
                    if d:
                        names.append(f"CLICK({d['x']},{d['y']})")
                    else:
                        names.append(ACTION_NAMES.get(a, f"A{a}"))
                print(f"  [{total_steps}] Plan ({plan_time:.0f}s, {len(plan_actions)} actions): {' → '.join(names)}")

            # EXECUTE (2-3 actions max per cycle)
            exec_count = min(3, len(plan_actions), remaining)
            for i in range(exec_count):
                action_int, data = plan_actions[i]
                prev_grid = grid.copy()
                prev_levels = fd.levels_completed

                try:
                    if data:
                        fd = env.step(INT_TO_GAME_ACTION[action_int], data=data)
                    else:
                        fd = env.step(INT_TO_GAME_ACTION[action_int])
                except Exception:
                    continue

                if fd is None:
                    break

                grid = get_frame(fd)
                total_steps += 1
                remaining -= 1
                changed = not np.array_equal(prev_grid, grid)
                n_px = int(np.sum(prev_grid != grid))

                # Update tracker
                diff = tracker.update(grid)
                if action_int == 6 and data:
                    tracker.record_click(data['x'], data['y'], changed, n_px)

                # Log for replan feedback
                act_name = ACTION_NAMES.get(action_int, f"A{action_int}")
                if data:
                    act_name = f"CLICK({data['x']},{data['y']})"
                result = f"{n_px}px changed" if changed else "no change"
                action_log.append(act_name)
                result_log.append(result)

                if verbose and changed:
                    print(f"    {act_name}: {result}")

                # Level up!
                if fd.levels_completed > prev_levels:
                    if verbose:
                        print(f"    ★ LEVEL {fd.levels_completed}/{fd.win_levels}!")
                    # Re-probe for new level
                    if remaining > 10:
                        action_model, tracker, fd, grid, ps, _ = \
                            probe(env, fd, grid, available, budget=min(8, remaining))
                        total_steps += ps
                        remaining -= ps
                    # Re-strategize
                    strategy = strategize(action_model, tracker, grid, available,
                                          fd.levels_completed, fd.win_levels)
                    action_log.clear()
                    result_log.clear()
                    break

                # Update available
                new_avail = [a.value if hasattr(a, "value") else int(a) for a in (fd.available_actions or [])]
                if new_avail:
                    available = new_avail

                if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                    break

        if fd and fd.levels_completed > best_levels:
            best_levels = fd.levels_completed

        if verbose:
            print(f"  Result: {fd.levels_completed if fd else 0}/{fd.win_levels if fd else '?'} in {total_steps} steps")

    return {"game_id": game_id, "best_levels": best_levels,
            "win_levels": fd.win_levels if fd else 0}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAGE Puzzle Solver v5")
    parser.add_argument("--game", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--model", default=None, help="Ollama model override")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    global MODEL
    _detect_model()
    if args.model:
        MODEL = args.model

    print("=" * 60)
    print(f"SAGE Puzzle Solver v5 — {MODEL}")
    print(f"Thinking: {'native' if _is_thinking_model() else 'disabled'}")
    print("Interleaved: act 2-3, observe, replan")
    print("=" * 60)

    # Warmup
    print("\nWarming up...", end=" ", flush=True)
    t = ask_llm("ready")
    print("OK" if "error" not in t.lower() else f"WARN: {t[:40]}")

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in e.game_id]
    elif args.all:
        targets = envs
    else:
        targets = envs[:5]

    print(f"\n{len(targets)} games, {args.attempts} attempts, budget={args.budget}\n")

    total_levels = 0
    scored = {}

    for env_info in targets:
        game_id = env_info.game_id
        prefix = game_id.split("-")[0]
        try:
            result = solve_game(arcade, game_id, max_attempts=args.attempts,
                               budget=args.budget, verbose=args.verbose)
        except Exception as e:
            if args.verbose:
                print(f"  {prefix}: CRASHED ({e})")
            continue

        levels = result["best_levels"]
        total_levels += levels
        if levels > 0:
            scored[prefix] = levels
            print(f"  ★ {prefix:6s}  {levels}/{result['win_levels']}")
        else:
            print(f"    {prefix:6s}  0/{result['win_levels']}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_levels} levels across {len(scored)} games")
    if scored:
        for g, l in sorted(scored.items(), key=lambda x: -x[1]):
            print(f"  {g}: {l}")


if __name__ == "__main__":
    main()
