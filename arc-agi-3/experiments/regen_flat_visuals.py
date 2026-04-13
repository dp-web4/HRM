#!/usr/bin/env python3
"""Regenerate L*_start.png and L*_solved.png for flat-view games.

Strategy:
  1. Run each game's solver via subprocess (or in-proc with monkey-patch).
  2. Wrap env.step to record: (action, data, pre_fd, post_fd).
  3. After the run, walk the recorded trace:
     - When pre_fd.levels_completed < post_fd.levels_completed:
       the pre_fd frame is the TRUE "L{pre_level}_solved" state.
     - The first frame of a new level is L{N}_start.

This captures the -2 state (frame before the winning action) for every
level transition, fixing the flat-view "end shows next level start" bug.
"""

import sys, os, importlib.util
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_all_frames
import numpy as np
from PIL import Image

VM = Path("/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory")

ARC_PALETTE = [
    (255,255,255),(204,204,204),(153,153,153),(102,102,102),(51,51,51),(0,0,0),
    (229,58,163),(255,123,204),(249,60,49),(30,147,255),(136,216,241),(255,220,0),
    (255,133,27),(146,18,49),(79,204,48),(163,86,214),
]


def grid_to_png(grid, scale=8):
    if grid.ndim == 3:
        grid = grid[-1]
    h, w = grid.shape
    img = np.zeros((h*scale, w*scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = ARC_PALETTE[int(grid[r,c]) % 16]
    return Image.fromarray(img)


def run_with_trace(game_id):
    """Run a fresh game. Return list of fd snapshots (one per step, plus initial)."""
    arcade = Arcade()
    env = arcade.make(game_id)
    fd = env.reset()
    trace = [fd]  # initial state
    orig_step = env.step

    def traced_step(action, data=None):
        nonlocal fd
        fd = orig_step(action, data=data)
        trace.append(fd)
        return fd

    env.step = traced_step
    return env, fd, trace


def capture_transitions(game_id, game_name, solver_runner, label_suffix="solved"):
    """Run solver_runner(env, fd) which drives the game via env.step.
    After the run, walk the trace and save start/solved frames per level.
    """
    env, fd, trace = run_with_trace(game_id)

    try:
        solver_runner(env, fd)
    except SystemExit:
        pass

    out = VM / game_name
    out.mkdir(exist_ok=True, parents=True)

    # Walk trace: detect level transitions
    prev_level = -1
    prev_fd = None
    captured_starts = set()
    for i, snap in enumerate(trace):
        lvl = snap.levels_completed
        # New level started — previous snap was the solved state of prev_level
        if prev_fd is not None and lvl > prev_fd.levels_completed:
            solved_level = prev_fd.levels_completed
            grid = get_all_frames(prev_fd)[-1]
            p = out / f"L{solved_level}_{label_suffix}.png"
            grid_to_png(grid).save(p)
            print(f"  {p.name}  (pre-transition of L{solved_level})")

        # Capture start frame for each level, the first time we see it
        if lvl not in captured_starts:
            # The start frame is the one right AFTER transition (if i>0) or the initial (i=0)
            grid = get_all_frames(snap)[-1]
            p = out / f"L{lvl}_start.png"
            grid_to_png(grid).save(p)
            print(f"  {p.name}  (initial state)")
            captured_starts.add(lvl)

        prev_fd = snap

    # If the final state is WIN, the last frame IS the true solved state of the final level
    if trace and trace[-1].state.name == "WIN":
        final_level = trace[-1].levels_completed - 1
        if final_level >= 0:
            grid = get_all_frames(trace[-1])[-1]
            p = out / f"L{final_level}_{label_suffix}.png"
            grid_to_png(grid).save(p)
            print(f"  {p.name}  (final WIN state)")

    print(f"{game_name}: {len(trace)-1} steps, {trace[-1].levels_completed if trace else 0} levels, final={trace[-1].state.name if trace else '?'}")


def load_solver_module(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py','').replace('/','_'),
        os.path.join(os.path.dirname(__file__), name))
    mod = importlib.util.module_from_spec(spec)
    return spec, mod


def regen_wa30():
    """wa30 solver uses KNOWN strings for L2,L3,L6,L7,L8 and beam search for L0,L1,L4,L5."""
    spec, mod = load_solver_module("wa30_solve_final.py")

    def runner(env, fd):
        # Swap the module's env with ours, then run its main solve loop
        mod_env = None
        spec.loader.exec_module(mod)

    try:
        capture_transitions('wa30-ee6fef47', 'wa30', runner, label_suffix="end")
    except Exception as e:
        print(f"wa30 FAIL: {e}")
        import traceback; traceback.print_exc()


def regen_from_solutions_json(game_name, game_id, label_suffix="solved"):
    """For games that store explicit per-level action sequences in solutions.json."""
    import json
    solutions_path = VM / game_name / "solutions.json"
    if not solutions_path.exists():
        print(f"{game_name}: no solutions.json")
        return

    with open(solutions_path) as f:
        sols = json.load(f)

    level_actions = sols.get('levels', sols)
    # Normalize keys to int level indexes
    def to_int(k):
        s = str(k).lstrip('L')
        try: return int(s)
        except: return None

    entries = []
    for k, v in level_actions.items():
        ki = to_int(k)
        if ki is not None:
            entries.append((ki, v))
    entries.sort(key=lambda x: x[0])

    if not entries:
        print(f"{game_name}: no valid level entries in solutions.json")
        return

    print(f"{game_name}: {len(entries)} level entries")

    INT_TO_GA = {a.value: a for a in GameAction}

    def runner(env, fd):
        for lv, actions_raw in entries:
            if not isinstance(actions_raw, list):
                continue
            for item in actions_raw:
                if isinstance(item, list) and len(item) == 3:
                    a, x, y = item
                    env.step(INT_TO_GA[a], data={'x': x, 'y': y})
                elif isinstance(item, int):
                    env.step(INT_TO_GA[item])

    try:
        capture_transitions(game_id, game_name, runner, label_suffix=label_suffix)
    except Exception as e:
        print(f"{game_name} FAIL: {e}")
        import traceback; traceback.print_exc()


def regen_via_solver_module(game_name, game_id, solver_file, label_suffix="solved"):
    """Run a solver module as-is, with monkey-patched env."""
    solver_path = os.path.join(os.path.dirname(__file__), solver_file)
    with open(solver_path) as f:
        src = f.read()

    def runner(env, fd):
        # Exec the solver in a namespace where 'env' and 'fd' are pre-set;
        # the solver should use those. This only works if the solver accepts
        # a pre-made env, which most don't.
        # FALLBACK: just execute the module top-to-bottom and hope it uses
        # arcade.make() — which creates a fresh env we can't intercept.
        ns = {'__file__': solver_path, '__name__': '__regen__'}
        exec(compile(src, solver_path, 'exec'), ns)

    # This won't intercept because solver makes its own env.
    # Better: run the solver normally, then re-run with captured actions.
    print(f"{game_name}: solver-module regen not reliable — use solutions.json path")


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'ka59'
    print(f"Regenerating flat visuals (target={target})")
    print("=" * 60)

    if target in ('ka59', 'all'):
        regen_from_solutions_json('ka59', 'ka59-9f096b4a', label_suffix="solved")
    if target in ('dc22', 'all'):
        regen_from_solutions_json('dc22', 'dc22-4c9bff3e', label_suffix="solved")

    print("=" * 60)
    print("Done.")
