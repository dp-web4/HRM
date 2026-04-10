#!/usr/bin/env python3
"""
Solver Loop — all game-solving modes.

Part of SAGE Solver v11 modular architecture.

Three entry points:
  solve_game_autonomous  — LLM-driven loop with world model, discovery, federation
  solve_game_interactive — Claude Code as reasoning engine (init/step/look/summarize)
  replay_actions         — execute a known action sequence with full visual capture

All modes use SessionWriter for persistent visual memory and live viewer.
State directory: /tmp/sage_solver/
"""

import sys
import os
import time
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from arcengine import GameAction
from arc_perception import get_frame, get_all_frames, color_name
from arc_vision import ARC_PALETTE, grid_to_image_b64

from solver_actions import (
    ACTION_NAMES, obj_name, obj_catalog, parse_actions)
from solver_context import (
    assemble_context_and_plan, METACOGNITIVE, ContextBudget)
from solver_probe import probe, discover_mechanics

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}

# Shared context paths
_SHARED_CONTEXT = os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'shared-context', 'arc-agi-3')
_FLEET_LEARNING = os.path.join(_SHARED_CONTEXT, 'fleet-learning')
_GAME_MECHANICS = os.path.join(_SHARED_CONTEXT, 'game-mechanics')
_SOLUTIONS_DIR = os.path.dirname(__file__)


def _load_fleet_knowledge(game_prefix):
    """Load fleet learning, mechanics, solutions for a game.
    Used by interactive mode to show context on init."""
    import glob
    sections = []

    # Game mechanics doc (first 80 lines)
    mech = os.path.join(_GAME_MECHANICS, f"{game_prefix}.md")
    if os.path.exists(mech):
        with open(mech) as f:
            sections.append(f"=== MECHANICS ===\n{''.join(f.readlines()[:80])}")

    # Solutions JSON
    sol = os.path.join(_SOLUTIONS_DIR, f"{game_prefix}_solutions.json")
    if os.path.exists(sol):
        with open(sol) as f:
            sections.append(f"=== SOLUTIONS ===\n{f.read()[:3000]}")

    # Fleet learning JSONL (all machines, last 10 entries each)
    for machine_dir in glob.glob(os.path.join(_FLEET_LEARNING, "*")):
        machine = os.path.basename(machine_dir)
        for jl in glob.glob(os.path.join(machine_dir, f"{game_prefix}*.jsonl")):
            with open(jl) as f:
                entries = f.readlines()[-10:]
            if entries:
                sections.append(f"=== Fleet ({machine}) ===\n{''.join(entries)}")
        for md in glob.glob(os.path.join(machine_dir, f"{game_prefix}*.md")):
            with open(md) as f:
                sections.append(f"=== {machine}/{os.path.basename(md)} ===\n{f.read()[:2000]}")

    # Cross-game patterns
    patterns = os.path.join(_SHARED_CONTEXT, 'cross-game-patterns.md')
    if os.path.exists(patterns):
        with open(patterns) as f:
            sections.append(f"=== CROSS-GAME PATTERNS ===\n{f.read()[:2000]}")

    # Exploration playbook
    playbook = os.path.join(_SHARED_CONTEXT, 'exploration-playbook.md')
    if os.path.exists(playbook):
        with open(playbook) as f:
            sections.append(f"=== EXPLORATION PLAYBOOK ===\n{f.read()[:2000]}")

    return "\n\n".join(sections) if sections else None


# ─── Consolidated Fleet Learning ───

def _load_consolidated_learning(game_prefix):
    """Load solutions + insights from shared-context consolidated data."""
    for base in [
        os.path.join(os.path.dirname(__file__), '..', '..', '..',
                     'shared-context', 'arc-agi-3', 'consolidated'),
        os.path.expanduser(
            '~/repos/shared-context/arc-agi-3/consolidated'),
        os.path.expanduser(
            '~/ai-workspace/shared-context/arc-agi-3/consolidated'),
    ]:
        if not os.path.exists(base):
            continue

        lines = []

        solutions_path = os.path.join(base, 'level_solutions.jsonl')
        if os.path.exists(solutions_path):
            with open(solutions_path) as f:
                for line in f:
                    try:
                        sol = json.loads(line.strip())
                        if sol.get('game', '') == game_prefix:
                            level = sol.get('level', '?')
                            steps = sol.get('steps', '?')
                            actions = sol.get('actions', [])
                            if actions:
                                lines.append(
                                    f"  SOLUTION L{level}: "
                                    f"{', '.join(str(a) for a in actions[:10])}"
                                    f" ({steps} steps)")
                            else:
                                lines.append(
                                    f"  SOLUTION L{level}: {steps} steps "
                                    f"(actions not recorded)")
                    except Exception:
                        pass

        insights_path = os.path.join(base, 'game_insights.jsonl')
        if os.path.exists(insights_path):
            with open(insights_path) as f:
                for line in f:
                    try:
                        ins = json.loads(line.strip())
                        if ins.get('game', '') == game_prefix:
                            lines.append(
                                f"  INSIGHT: {ins.get('insight', '')[:150]}")
                    except Exception:
                        pass

        patterns_path = os.path.join(base, 'structural_patterns.jsonl')
        if os.path.exists(patterns_path):
            with open(patterns_path) as f:
                for line in f:
                    try:
                        pat = json.loads(line.strip())
                        lines.append(
                            f"  PATTERN: {pat.get('pattern', '')[:150]}")
                    except Exception:
                        pass

        if lines:
            return ("CONSOLIDATED FLEET LEARNING "
                    "(proven solutions and insights):\n"
                    + "\n".join(lines))
        break

    return ""


# ─── World-Model-Driven Planning ───

def _plan_with_world_model(backend, world_model, plan, narrative,
                           tracker, action_model,
                           scene_desc="", fleet_context="",
                           verbose=False):
    """Ask LLM to update world model + plan, then return next action."""
    from arc_world_model import PLAN_UPDATE_PROMPT
    prompt = PLAN_UPDATE_PROMPT.format(
        world_model=world_model.to_context(),
        plan_state=plan.to_context(),
        scene_description=(f"SCENE:\n{scene_desc}"
                           if scene_desc else "(no scene description)"),
        fleet_context=(fleet_context if fleet_context
                       else "(no fleet knowledge for this game)"),
    )
    return backend.query(prompt)


# ─── Vision Analysis ───

def _vision_analysis(backend, grid, interactive, available, fd, verbose=False):
    """Ask a vision-capable model to analyze the game screen."""
    if not backend.supports_vision():
        return ""
    try:
        img_b64 = grid_to_image_b64(grid)
        vision_prompt = (
            "You are solving a puzzle game. This is the current screen.\n\n"
            f"Known interactive objects: "
            f"{', '.join(obj_name(o) for o in interactive)}\n"
            f"Available actions: "
            f"{', '.join(ACTION_NAMES.get(a, f'A{a}') for a in available)}\n"
            f"Levels completed: {fd.levels_completed}/{fd.win_levels}\n\n"
            "Look at this image carefully. What do you see?\n"
            "1. What is the spatial layout? (clusters, lines, patterns, symmetry)\n"
            "2. What might the GOAL STATE look like? (what should change to win)\n"
            "3. What sequence of actions would transform the current state "
            "toward the goal?\n"
            "Be specific about positions and colors."
        )
        result = backend.query(vision_prompt, max_tokens=500,
                               image_b64=img_b64)
        if verbose and result:
            print(f"  Vision ({len(result)} chars): {result[:120]}...")
        return result
    except Exception as e:
        if verbose:
            print(f"  Vision failed: {e}")
        return ""


# ─── Scene Description Helper ───

def _build_scene_desc(tracker, grid):
    """Build scene description from tracker state. Returns text or empty."""
    try:
        from sage.irp.plugins.grid_vision_irp import GridVisionIRP, GridObservation
        gv = GridVisionIRP.__new__(GridVisionIRP)
        gv._buffer = []
        gv._frame_count = 0
        gv._prev_frame = None
        objects_list = [
            {"id": o.id, "color": o.color,
             "bbox": [o.y, o.x, o.y + o.h, o.x + o.w],
             "centroid": [o.cy, o.cx], "size": o.size}
            for o in tracker.objects.values()]
        obs = GridObservation(
            frame_raw=grid, objects=objects_list,
            changes=[], moved=[], step_number=0, level_id="")
        return gv.describe_scene(obs)
    except Exception:
        return ""


# ─── Autonomous Solve Loop ───

def solve_game_autonomous(arcade, game_id, backend, config):
    """Solve a game using the autonomous world-model-driven loop.

    Merges v7 logic: discover/probe -> vision -> world-model loop -> federation.

    Args:
        arcade: Arcade instance
        game_id: full game identifier
        backend: ModelBackend instance
        config: SolverConfig instance

    Returns:
        dict with game_id, best_levels, win_levels
    """
    from arc_narrative import SessionNarrative
    from arc_world_model import WorldModel, PlanState, parse_plan_update

    prefix = game_id.split("-")[0]
    best_levels = 0
    verbose = config.verbose

    # Context constructor (adaptive membot queries)
    ctx = None
    if not config.kaggle:
        try:
            from arc_context import ContextConstructor
            ctx = ContextConstructor(prefix)
        except Exception:
            pass

    # Federation (fleet knowledge from all machines)
    fed = None
    fleet_context = ""
    if not config.kaggle:
        try:
            from arc_federation import FederatedKnowledge
            fed = FederatedKnowledge()
            fleet_context = fed.build_context(prefix)
        except Exception:
            pass

    # Consolidated fleet learning
    if not config.kaggle:
        consolidated = _load_consolidated_learning(prefix)
        if consolidated:
            fleet_context = (
                fleet_context + "\n\n" + consolidated
                if fleet_context else consolidated)

    if verbose:
        if fed:
            print(f"  {fed.summary()}")
        if fleet_context:
            print(f"  Fleet context: {len(fleet_context)} chars")

    # Session writer (viewer + persistent visual memory)
    session_writer = None
    _SessionWriter = None
    if not config.kaggle:
        try:
            from arc_session_writer import SessionWriter as _SW
            _SessionWriter = _SW
        except Exception:
            pass

    for attempt in range(config.attempts):
        if ctx:
            ctx.new_attempt()

        env = arcade.make(game_id)
        fd = env.reset()
        grid = get_frame(fd)
        available = [
            a.value if hasattr(a, "value") else int(a)
            for a in (fd.available_actions or [])]
        total_steps = 0

        # Initialize session writer on first attempt
        if _SessionWriter and session_writer is None:
            try:
                session_writer = _SessionWriter(
                    game_id, win_levels=fd.win_levels,
                    available_actions=available,
                    baseline=config.budget or 0,
                    player=config.model or "autonomous")
            except Exception:
                pass
        elif session_writer:
            session_writer.new_attempt(attempt + 1)

        if verbose:
            print(f"\n  Attempt {attempt + 1}/{config.attempts}"
                  f" | Actions: {available}"
                  f" | Levels: 0/{fd.win_levels}")

        # ── DISCOVERY / PROBE ──
        mechanics = None
        if config.use_discovery:
            result = discover_mechanics(
                env, fd, grid, available,
                budget=config.budget, verbose=verbose)
            if result is not None:
                (action_model, tracker, fd, grid,
                 probe_steps, probe_levels, mechanics) = result
            else:
                # Fallback to basic probe
                probe_budget = min(config.budget // 3, 25)
                (action_model, tracker, fd, grid,
                 probe_steps, probe_levels) = probe(
                    env, fd, grid, available, budget=probe_budget)
        else:
            probe_budget = min(config.budget // 3, 25)
            (action_model, tracker, fd, grid,
             probe_steps, probe_levels) = probe(
                env, fd, grid, available, budget=probe_budget)

        total_steps += probe_steps

        # Initialize narrative
        narrative = SessionNarrative(grid)
        interactive = tracker.get_interactive_objects()

        if verbose:
            if mechanics:
                print(f"  Discovery: {probe_steps} steps"
                      f" | {mechanics.game_type}"
                      f" | confidence={mechanics.confidence}")
                for line in mechanics.to_context().split("\n")[:8]:
                    print(f"    {line}")
            else:
                print(f"  Probe: {probe_steps} steps"
                      f" | {len(interactive)} interactive")
            for o in interactive:
                print(f"    + {obj_name(o)} at ({o.cx},{o.cy})"
                      f" -- {o.click_effectiveness:.0%}")

        # Layer 3: query membot with probe results
        game_type = action_model.infer_game_type()
        if ctx:
            ctx.on_game_start(available, game_type)
            ctx.on_probe_complete(
                [obj_name(o) for o in interactive],
                action_model.describe(), game_type)

        # ── VISION ANALYSIS ──
        vision_insight = ""
        if config.vision and backend.supports_vision():
            vision_insight = _vision_analysis(
                backend, grid, interactive, available, fd, verbose)

        if fd is None or fd.state.name in ("WON", "LOST", "GAME_OVER"):
            if fd and fd.levels_completed > best_levels:
                best_levels = fd.levels_completed
            continue

        # ── BUILD WORLD MODEL ──
        wm = WorldModel()
        wm.update_from_probe(tracker, action_model)
        wm.current_level = fd.levels_completed
        if mechanics:
            wm.mechanics_context = mechanics.to_context()
        plan = PlanState()

        # Build scene description
        scene_desc = _build_scene_desc(tracker, grid)
        if verbose and scene_desc:
            print(f"  Scene: {scene_desc[:120]}...")

        # ── WORLD-MODEL-DRIVEN LOOP ──
        remaining = config.budget - total_steps

        while remaining > 0 and fd.state.name not in (
                "WON", "LOST", "GAME_OVER"):
            # Ask LLM: update plan + choose next action
            if config.use_world_model:
                t0 = time.time()
                response = _plan_with_world_model(
                    backend, wm, plan, narrative, tracker, action_model,
                    scene_desc=scene_desc, fleet_context=fleet_context,
                    verbose=verbose)
                plan_time = time.time() - t0

                prev_hypothesis = plan.hypothesis
                parse_plan_update(response, wm, plan)

                plan_actions = parse_actions(
                    plan.next_action, tracker, available)
                if not plan_actions:
                    plan_actions = parse_actions(
                        response, tracker, available)
            else:
                # Context-plan mode (no world model)
                t0 = time.time()
                response = assemble_context_and_plan(
                    backend, narrative, tracker, action_model, available,
                    fd.levels_completed, fd.win_levels, grid=grid,
                    ctx=ctx, attempt_num=attempt + 1, verbose=verbose,
                    fleet_context=fleet_context,
                    vision_insight=vision_insight,
                    mechanics=mechanics)
                plan_time = time.time() - t0
                plan_actions = parse_actions(
                    response, tracker, available)

            # Fallback: random interactive object or random action
            if not plan_actions:
                interactive = tracker.get_interactive_objects()
                if interactive:
                    o = random.choice(interactive)
                    plan_actions = [
                        (6, {'x': o.cx, 'y': o.cy}, obj_name(o))]
                else:
                    a = random.choice(available)
                    plan_actions = [
                        (a, None, ACTION_NAMES.get(a, f"A{a}"))]

            if verbose:
                aname = (plan_actions[0][2]
                         if len(plan_actions[0]) > 2
                         else str(plan_actions[0][0]))
                if config.use_world_model:
                    hyp_changed = (
                        " [NEW]"
                        if plan.hypothesis != prev_hypothesis
                        and prev_hypothesis else "")
                    print(f"  [{total_steps}] {aname} ({plan_time:.0f}s)"
                          f" expect=[{plan.expected_outcome[:60]}]"
                          f" hyp=[{plan.hypothesis[:60]}]{hyp_changed}")
                else:
                    print(f"  [{total_steps}] {aname} ({plan_time:.0f}s)")

            # ── EXECUTE ONE ACTION ──
            action_int = plan_actions[0][0]
            data = plan_actions[0][1]
            target_name = (
                plan_actions[0][2]
                if len(plan_actions[0]) > 2
                else ACTION_NAMES.get(action_int, f"A{action_int}"))

            prev_grid = grid.copy()
            prev_levels = fd.levels_completed

            try:
                if data:
                    fd = env.step(
                        INT_TO_GAME_ACTION[action_int], data=data)
                else:
                    fd = env.step(INT_TO_GAME_ACTION[action_int])
            except Exception:
                remaining -= 1
                total_steps += 1
                if config.use_world_model:
                    plan.record_outcome(
                        target_name, plan.expected_outcome,
                        "ERROR: action failed", 0, False)
                continue

            if fd is None:
                break

            all_frames = get_all_frames(fd)
            grid = all_frames[-1]
            # Normalize grid shape
            if grid.ndim == 3:
                grid = grid[-1]
            if grid.ndim != 2 or grid.shape == (0,):
                grid = np.zeros((64, 64), dtype=np.int8)
            total_steps += 1
            remaining -= 1

            # Persist every frame to visual memory
            if session_writer:
                session_writer.record_step(
                    action_int, grid,
                    all_frames=all_frames if len(all_frames) > 1 else None,
                    levels_completed=fd.levels_completed,
                    x=data.get('x') if data else None,
                    y=data.get('y') if data else None,
                    state=fd.state.name if hasattr(fd.state, 'name')
                        else str(fd.state))

            # Observe what happened
            diff = tracker.update(grid)
            if prev_grid.shape != grid.shape:
                prev_grid = np.zeros_like(grid)
            n_changed = int(np.sum(prev_grid != grid))
            if action_int == 6 and data:
                changed = not np.array_equal(prev_grid, grid)
                tracker.record_click(
                    data['x'], data['y'], changed, n_changed)

            # Record in narrative
            event = narrative.record(
                total_steps, action_int, target_name, data,
                grid, prev_levels, fd.levels_completed)

            # Compare expectation to reality
            level_up = fd.levels_completed > prev_levels
            actual = (event.observation if event.observation
                      else f"{n_changed}px changed")

            if config.use_world_model:
                plan.record_outcome(
                    target_name, plan.expected_outcome, actual,
                    n_changed, level_up)
                wm.record_action(target_name, n_changed, level_up)

            if verbose:
                sym = ("*" if level_up
                       else ("+" if n_changed > 10 else "-"))
                print(f"    {sym} {actual} ({n_changed}px)")

            # ── LEVEL UP ──
            if fd.levels_completed > prev_levels:
                if verbose:
                    print(f"    * LEVEL {fd.levels_completed}/{fd.win_levels}!")

                # Store winning sequence
                winning_actions = [
                    ev.target for ev in narrative.events[-10:]
                    if ev.changed]
                if ctx:
                    ctx.on_level_up(fd.levels_completed, winning_actions)

                if config.use_world_model:
                    wm.on_level_up(fd.levels_completed)
                    plan = PlanState()

                # Re-probe for new level
                if remaining > 10:
                    narrative = SessionNarrative(grid)
                    (action_model, tracker, fd, grid, ps, _) = probe(
                        env, fd, grid, available,
                        budget=min(10, remaining))
                    total_steps += ps
                    remaining -= ps

                    if config.use_world_model:
                        wm.update_from_probe(tracker, action_model)

                    scene_desc = _build_scene_desc(tracker, grid)

            # Update available actions
            new_avail = [
                a.value if hasattr(a, "value") else int(a)
                for a in (fd.available_actions or [])]
            if new_avail:
                available = new_avail

            if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                break

        # ── POST-ATTEMPT ──
        if session_writer:
            session_writer.record_game_end(
                fd.state.name if fd else "UNKNOWN",
                fd.levels_completed if fd else 0)

        if fd and fd.levels_completed > best_levels:
            best_levels = fd.levels_completed

        if verbose:
            print(f"  Result: "
                  f"{fd.levels_completed if fd else 0}"
                  f"/{fd.win_levels if fd else '?'}"
                  f" in {total_steps} steps")
            patterns = narrative.detect_patterns()
            if patterns:
                print(f"  Patterns: {patterns[:120]}")

        # Game-end learning
        if ctx:
            ctx.on_game_end(
                fd.levels_completed if fd else 0,
                fd.win_levels if fd else 0,
                narrative.detect_patterns(),
                narrative.object_summary())

        # Federation: store discoveries for other machines
        if fed:
            patterns = narrative.detect_patterns()
            obj_summary = narrative.object_summary()
            interactive = tracker.get_interactive_objects()
            game_type = action_model.infer_game_type()

            if interactive:
                fed.add_discovery(
                    prefix,
                    f"Interactive objects: "
                    f"{', '.join(obj_name(o) for o in interactive[:5])}. "
                    f"Game type: {game_type}. {obj_summary[:100]}")

            if "STABLE" in patterns:
                fed.add_failure(
                    prefix,
                    f"Clicking interactive objects repeatedly doesn't "
                    f"advance levels. Grid similarity stays stable. "
                    f"Need correct sequence, not just correct objects.")

            if fd and fd.levels_completed > 0:
                winning = [
                    ev.target for ev in narrative.events if ev.leveled_up]
                fed.add_discovery(
                    prefix,
                    f"SOLVED Level {fd.levels_completed}: winning actions "
                    f"included {', '.join(winning[-5:])}. "
                    f"Game type: {game_type}.")
                fed.add_strategy(
                    game_type,
                    f"On {prefix}: level solved by targeting "
                    f"{', '.join(winning[-3:])}.")

            if attempt == config.attempts - 1:
                fed.add_meta_insight(
                    f"Game {prefix} ({game_type}): {patterns[:150]}")
                fed.save()

                if ctx:
                    ctx.store(
                        f"SAGE game experience summary ({prefix}): "
                        f"Played {config.attempts} attempts. "
                        f"Best: {best_levels}/{fd.win_levels if fd else '?'}"
                        f" levels. "
                        f"Key learning: {patterns[:150]}. "
                        f"Objects learned: {obj_summary[:150]}.")

    return {"game_id": game_id, "best_levels": best_levels,
            "win_levels": fd.win_levels if fd else 0}


# ─── Replay Mode (Direct API) ───

def replay_actions(game_id, actions, label="replay"):
    """Execute a known action sequence with full visual capture.

    This is the "direct API" mode — used when the solution is already computed
    (from source analysis, algorithmic solver, or previous session).
    Every frame is persisted via SessionWriter.

    Args:
        game_id: full game ID (e.g., 'tu93-2b534c15')
        actions: list of actions. Each is either:
            - int (direction: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=SELECT)
            - tuple (action_int, x, y) for clicks
            - str direction name ('UP', 'DOWN', etc.)
        label: player label for the run

    Returns:
        dict with levels_completed, win_levels, total_steps, state, run_dir
    """
    from arc_agi import Arcade
    from arc_session_writer import SessionWriter

    name_to_int = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4,
                   'SELECT': 5, 'CLICK': 6}

    arcade = Arcade()
    env = arcade.make(game_id)
    fd = env.reset()

    sw = SessionWriter(
        game_id, win_levels=fd.win_levels,
        available_actions=[
            a.value if hasattr(a, 'value') else int(a)
            for a in (fd.available_actions or [])],
        player=label)

    for item in actions:
        # Normalize action format
        if isinstance(item, str):
            upper = item.upper()
            action_int = name_to_int[upper] if upper in name_to_int else int(item)
            x = y = None
        elif isinstance(item, tuple):
            action_int, x, y = item
        else:
            action_int = int(item)
            x = y = None

        # Execute
        ga = INT_TO_GAME_ACTION[action_int]
        if x is not None:
            fd = env.step(ga, data={"x": x, "y": y})
        else:
            fd = env.step(ga)

        all_frames = get_all_frames(fd)
        grid = all_frames[-1]
        sw.record_step(
            action_int, grid,
            all_frames=all_frames if len(all_frames) > 1 else None,
            levels_completed=fd.levels_completed,
            x=x, y=y,
            state=fd.state.name if hasattr(fd.state, 'name')
                else str(fd.state))

        if fd.state.name in ("GAME_OVER", "WON"):
            break

    sw.record_game_end(fd.state.name, fd.levels_completed)

    return {
        "levels_completed": fd.levels_completed,
        "win_levels": fd.win_levels,
        "total_steps": sw.step,
        "state": fd.state.name,
        "run_dir": sw.run_dir,
    }


# ─── Interactive Solve Loop ───

STATE_DIR = "/tmp/sage_solver"


def solve_game_interactive(game_prefix, backend, config):
    """Interactive mode: Claude Code as the reasoning engine.

    Wraps sage_solver's init/step/look/summarize pattern.
    State persists to /tmp/sage_solver/ between calls.

    Args:
        game_prefix: game prefix to initialize
        backend: ClaudeInteractiveBackend instance
        config: SolverConfig instance

    Returns:
        dict with game_id, state_dir
    """
    from arc_agi import Arcade
    from arc_perception import find_color_regions
    from arc_vision import grid_to_image_b64

    os.makedirs(STATE_DIR, exist_ok=True)

    arcade = Arcade()
    matches = [
        e for e in arcade.get_environments()
        if game_prefix in e.game_id]
    if not matches:
        print(f"No game matching '{game_prefix}'")
        return {"game_id": None, "state_dir": STATE_DIR}

    env_info = matches[0]
    game_id = env_info.game_id
    env = arcade.make(game_id)
    fd = env.reset()
    grid = get_frame(fd)
    available = [
        a.value if hasattr(a, "value") else int(a)
        for a in (fd.available_actions or [])]

    # Render image
    try:
        from PIL import Image
        img_path = os.path.join(STATE_DIR, "frame.png")
        h, w = grid.shape
        scale = 4
        img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
        for r in range(h):
            for c in range(w):
                color = ARC_PALETTE.get(int(grid[r, c]), (128, 128, 128))
                img[r * scale:(r + 1) * scale,
                    c * scale:(c + 1) * scale] = color
        Image.fromarray(img).save(img_path)
    except Exception:
        img_path = "(image rendering failed)"

    # Save state for future step/look/summarize calls
    session = {
        "game_id": game_id,
        "game_prefix": game_prefix,
        "available_actions": available,
        "win_levels": fd.win_levels,
        "levels_completed": 0,
        "step": 0,
        "state": "NOT_FINISHED",
        "actions_log": [],
        "observations": [],
        "level_summaries": [],
        "level_solutions": {},
        "level_start_step": 0,
        "level_actions": [],
        "baseline": sum(env_info.baseline_actions),
    }
    with open(os.path.join(STATE_DIR, "session.json"), "w") as f:
        json.dump(session, f, indent=2)

    np.save(os.path.join(STATE_DIR, "current_grid.npy"), grid)

    avail_str = ", ".join(
        f"{a}={ACTION_NAMES.get(a, f'A{a}')}" for a in available)

    print(f"GAME: {game_id}")
    print(f"LEVELS: 0/{fd.win_levels}")
    print(f"ACTIONS: {avail_str}")
    print(f"BASELINE: {session['baseline']} actions for 100% efficiency")
    print(f"IMAGE: {img_path}")

    # Load fleet knowledge for this game
    fleet_kb = _load_fleet_knowledge(game_prefix)
    if fleet_kb:
        print(f"\nFLEET KNOWLEDGE:\n{fleet_kb}")

    print(f"\nUse: sage_solver.py --interactive step <action> [x] [y]")

    return {"game_id": game_id, "state_dir": STATE_DIR}


def interactive_step(action, x=None, y=None):
    """Execute one step in interactive mode. Handles animation capture.

    Mirrors sage_solver.py's cmd_step but uses v11 infrastructure.
    """
    from arc_agi import Arcade
    from PIL import Image
    import shutil

    session_path = os.path.join(STATE_DIR, "session.json")
    if not os.path.exists(session_path):
        print("No active session. Run with --interactive first.")
        return

    with open(session_path) as f:
        session = json.load(f)

    # Reconstruct game state by replaying
    arcade = Arcade()
    env = arcade.make(session["game_id"])
    fd = env.reset()

    INT_TO_GA = {a.value: a for a in GameAction}

    for entry in session["actions_log"]:
        a = entry["action"]
        if a == 6 and "x" in entry and "y" in entry:
            fd = env.step(INT_TO_GA[a], data={"x": entry["x"], "y": entry["y"]})
        else:
            fd = env.step(INT_TO_GA[a])

    # Execute new action
    prev_grid = get_frame(fd)
    prev_levels = fd.levels_completed

    if action == 6 and x is not None and y is not None:
        fd = env.step(INT_TO_GA[6], data={"x": int(x), "y": int(y)})
        action_entry = {"action": 6, "x": int(x), "y": int(y)}
        action_desc = f"CLICK({x},{y})"
    else:
        fd = env.step(INT_TO_GA[action])
        action_entry = {"action": action}
        action_desc = ACTION_NAMES.get(action, f"ACTION{action}")

    session["actions_log"].append(action_entry)
    session.setdefault("level_actions", []).append(action_entry)

    # Get ALL frames (animation support)
    all_frames = get_all_frames(fd)
    grid = all_frames[-1]
    anim_frames = all_frames[:-1] if len(all_frames) > 1 else []

    session["step"] += 1
    prev_levels_val = session["levels_completed"]
    session["levels_completed"] = fd.levels_completed
    session["state"] = fd.state.name

    # Render frame
    img_path = os.path.join(STATE_DIR, "frame.png")
    _render_grid_sdk(grid).save(img_path)

    # Save animation if present
    if anim_frames:
        anim_dir = os.path.join(STATE_DIR, "animations")
        os.makedirs(anim_dir, exist_ok=True)
        level = session["levels_completed"]
        step = session["step"]
        # Save as GIF
        try:
            gif_frames = [_render_grid_sdk(f, scale=2) for f in all_frames]
            gif_path = os.path.join(anim_dir, f"anim_L{level}_S{step}.gif")
            gif_frames[0].save(gif_path, save_all=True,
                append_images=gif_frames[1:], duration=200, loop=1)
        except Exception:
            pass
        session.setdefault("animations", []).append({
            "step": step, "level": level, "frames": len(all_frames),
        })

    # Normalize grid shapes (game may return empty or 3D frames)
    if grid.ndim == 3:
        grid = grid[-1]
    if grid.ndim != 2 or grid.shape == (0,):
        grid = np.zeros((64, 64), dtype=np.int8)
    if prev_grid.ndim == 3:
        prev_grid = prev_grid[-1]
    if prev_grid.shape != grid.shape:
        prev_grid = np.zeros_like(grid)

    # Diff
    n_changed = int(np.sum(prev_grid != grid))
    if anim_frames:
        diff_desc = f"ANIMATION ({len(all_frames)} frames): {n_changed}px changed"
    elif n_changed == 0:
        diff_desc = "NOTHING CHANGED"
    elif n_changed <= 2:
        diff_desc = f"tiny change ({n_changed}px)"
    elif n_changed < 50:
        diff_desc = f"small change ({n_changed}px)"
    elif n_changed < 200:
        diff_desc = f"moderate change ({n_changed}px)"
    else:
        diff_desc = f"large change ({n_changed}px)"

    # Level-up detection
    level_up = fd.levels_completed > prev_levels_val
    if level_up:
        # Save solved state + new level start
        np.save(os.path.join(STATE_DIR, f"level_{prev_levels_val}_final_grid.npy"), grid)
        np.save(os.path.join(STATE_DIR, f"level_{fd.levels_completed}_start_grid.npy"), grid)
        session.setdefault("level_solutions", {})[str(prev_levels_val)] = {
            "actions": session.get("level_actions", []),
            "steps": len(session.get("level_actions", [])),
        }
        session["level_actions"] = []

    # Save state
    np.save(os.path.join(STATE_DIR, "current_grid.npy"), grid)
    np.save(os.path.join(STATE_DIR, "previous_grid.npy"), prev_grid)
    with open(session_path, "w") as f:
        json.dump(session, f, indent=2)

    # Output
    print(f"STEP {session['step']}: {action_desc}")
    print(f"WHAT CHANGED: {diff_desc}")
    print(f"LEVELS: {fd.levels_completed}/{session['win_levels']}")
    print(f"IMAGE: {img_path}")
    if anim_frames:
        print(f"ANIMATION: {len(anim_frames)} frames captured")
    if level_up:
        print(f"\n★★★ LEVEL UP! Now at level {fd.levels_completed}/{session['win_levels']} ★★★")


# SDK palette renderer (matches arc_agi/rendering.py COLOR_MAP)
_SDK_PALETTE = {
    0: (255,255,255), 1: (204,204,204), 2: (153,153,153), 3: (102,102,102),
    4: (51,51,51), 5: (0,0,0), 6: (229,58,163), 7: (255,123,204),
    8: (249,60,49), 9: (30,147,255), 10: (136,216,241), 11: (255,220,0),
    12: (255,133,27), 13: (146,18,49), 14: (79,204,48), 15: (163,86,214),
}

def _render_grid_sdk(grid, scale=4):
    """Render grid using SDK palette. Returns PIL Image."""
    from PIL import Image
    if grid.ndim == 3:
        grid = grid[-1]
    if grid.ndim != 2:
        grid = np.zeros((64, 64), dtype=np.uint8)
    h, w = grid.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = _SDK_PALETTE.get(int(grid[r, c]), (128, 128, 128))
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color
    return Image.fromarray(img)
