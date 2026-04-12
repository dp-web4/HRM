#!/usr/bin/env python3
"""
Visual Memory Capture — Generate frame captures for solved ARC-AGI-3 games.

Strategy: Run solver with env.step monkey-patched to collect all actions,
then replay through a fresh env with SessionWriter for visual capture.

Games handled:
  - cd82: circular stamp painting puzzle (6 levels)
  - bp35: gravity platformer (9 levels)
  - m0r0: mirrored cursor maze (6 levels)
  - s5i5: arrow resize/rotate puzzle (8 levels)
  - lp85: cyclic rotation puzzle (8 levels)
  - r11l: creature movement puzzle (6 levels)

Usage:
  python3 capture_visuals.py [game_id]
  python3 capture_visuals.py          # all games
  python3 capture_visuals.py cd82     # just cd82
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_all_frames
from arc_session_writer import SessionWriter


# === Action Collector: monkey-patch env.step to collect all actions ===

class ActionCollector:
    """Wraps an env to collect all (action_id, data) pairs while solving."""

    def __init__(self, env):
        self._env = env
        self.actions = []
        self._orig_step = env.step

    def step(self, action, data=None):
        self.actions.append((action, data))
        return self._orig_step(action, data=data)

    def patch(self):
        self._env.step = self.step

    def unpatch(self):
        self._env.step = self._orig_step


def replay_with_capture(game_id, action_sequence, label="capture"):
    """Replay an action sequence through the game engine with visual capture.

    action_sequence: list of (GameAction, data_dict_or_None)
    Returns (levels_completed, win_levels, total_steps)
    """
    arcade = Arcade()
    env = arcade.make(game_id)
    fd = env.reset()

    sw = SessionWriter(game_id, win_levels=fd.win_levels,
                       available_actions=[
                           a.value if hasattr(a, 'value') else int(a)
                           for a in (fd.available_actions or [])],
                       player=label)

    for action, data in action_sequence:
        fd = env.step(action, data=data)
        all_frames = get_all_frames(fd)
        grid = all_frames[-1]

        # Determine action label
        action_id = action.value if hasattr(action, 'value') else int(action)
        x = data.get('x') if data else None
        y = data.get('y') if data else None

        sw.record_step(action_id, grid,
                       all_frames=all_frames if len(all_frames) > 1 else None,
                       levels_completed=fd.levels_completed,
                       x=x, y=y,
                       state=fd.state.name if hasattr(fd.state, 'name') else str(fd.state))
        if fd.state.name in ("GAME_OVER", "WIN"):
            break

    sw.record_game_end(fd.state.name, fd.levels_completed)
    files = [f for f in os.listdir(sw.run_dir) if f.endswith('.png')]
    gifs = [f for f in os.listdir(sw.run_dir) if f.endswith('.gif')]
    print(f"  {game_id}: {fd.levels_completed}/{fd.win_levels} "
          f"in {sw.step} steps, {len(files)} PNGs, {len(gifs)} GIFs")
    print(f"  Saved to: {sw.run_dir}")
    return fd.levels_completed, fd.win_levels, sw.step


def _solve_with_collection(game_id, solver_script, label):
    """Generic solver runner: exec solver with monkey-patched env.step,
    collect all actions, then replay with SessionWriter.

    Uses ActionCollector which captures ALL action types (not just clicks).
    """
    game_prefix = game_id.split('-')[0]
    print(f"\n=== Solving {game_prefix} (collecting actions) ===")

    solver_path = os.path.join(os.path.dirname(__file__), solver_script)

    with open(solver_path) as f:
        source = f.read()

    main_marker = "\ntotal_actions = 0\n"
    if main_marker not in source:
        print(f"  ERROR: Can't find main loop marker in {solver_script}")
        return None

    setup_code, main_code = source.split(main_marker, 1)
    setup_code += "\ntotal_actions = 0\n"

    ns = {'__file__': solver_path}
    exec(compile(setup_code, solver_path, 'exec'), ns)

    env = ns['env']
    collector = ActionCollector(env)
    collector.patch()

    exec(compile(main_code, solver_path, 'exec'), ns)

    collector.unpatch()

    print(f"  Collected {len(collector.actions)} actions")

    if collector.actions:
        replay_with_capture(game_id, collector.actions, label)

    return collector.actions


def solve_cd82():
    return _solve_with_collection('cd82-fb555c5d', 'cd82_solve.py', 'world-model-solver')


def solve_bp35():
    """BP35 uses pre-computed solutions — extract action sequence directly."""
    print(f"\n=== Solving bp35 (collecting actions) ===")

    # Import bp35 solution sequences
    solver_path = os.path.join(os.path.dirname(__file__), 'bp35_solve.py')
    with open(solver_path) as f:
        source = f.read()

    # Execute just the solution definitions (everything before "arcade = Arcade()")
    marker = "\narcade = Arcade()"
    setup = source.split(marker)[0]
    ns = {'__file__': solver_path}
    exec(compile(setup, solver_path, 'exec'), ns)

    # Now run through the game collecting actions
    arcade_inst = Arcade()
    env = arcade_inst.make('bp35-0a0ad940')
    fd = env.reset()

    collector = ActionCollector(env)
    collector.patch()

    # Import the helpers we need
    game_obj = env._game

    LEFT = GameAction.ACTION3
    RIGHT = GameAction.ACTION4
    CLICK = GameAction.ACTION6
    UNDO = GameAction.ACTION7

    def click_act_local(gx, gy):
        engine = game_obj.oztjzzyqoek
        cam_y = engine.camera.rczgvgfsfb[1]
        return env.step(CLICK, data={"x": gx*6, "y": gy*6 - cam_y})

    solutions = [
        ("L0", ns['L0_SOL'], 15),
        ("L1", ns['L1_SOL'], 72),
        ("L2", ns['L2_SOL'], 36),
        ("L3", ns['L3_SOL'], 31),
        ("L4", ns['L4_SOL'], 31),
        ("L5", ns['L5_SOL'], 48),
        ("L6", ns['L6_SOL'], 86),
        ("L7", ns['L7_SOL'], 155),
        ("L8", ns['L8_SOL'], 422),
    ]

    for name, sol, baseline in solutions:
        old_level = game_obj.level_index
        for m in sol:
            if m[0] == 'R':
                fd = env.step(RIGHT)
            elif m[0] == 'L':
                fd = env.step(LEFT)
            elif m[0] == 'C':
                fd = click_act_local(m[1], m[2])

        # Pump animation
        for i in range(10):
            if game_obj.level_index > old_level:
                break
            fd = env.step(LEFT)

    collector.unpatch()
    print(f"  Collected {len(collector.actions)} actions")

    if collector.actions:
        replay_with_capture('bp35-0a0ad940', collector.actions, 'world-model-solver')

    return collector.actions


def solve_m0r0():
    return _solve_with_collection('m0r0-dadda488', 'm0r0_solve.py', 'world-model-solver')


def solve_s5i5():
    return _solve_with_collection('s5i5-a48e4b1d', 's5i5_solve.py', 'bfs-solver')


def solve_lp85():
    pre = [os.path.join(os.path.dirname(__file__),
                        'environment_files', 'lp85', '305b61c3')]
    for p in pre:
        if p not in sys.path:
            sys.path.insert(0, p)
    return _solve_with_collection('lp85-305b61c3', 'lp85_solve.py', 'bfs-solver')


def solve_r11l():
    return _solve_with_collection('r11l-aa269680', 'r11l_solve.py', 'bfs-solver')


GAME_SOLVERS = {
    'cd82': solve_cd82,
    'bp35': solve_bp35,
    'm0r0': solve_m0r0,
    's5i5': solve_s5i5,
    'lp85': solve_lp85,
    'r11l': solve_r11l,
}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("Visual Memory Capture for ARC-AGI-3 Games")
    print("=" * 50)

    if target == "all":
        games = list(GAME_SOLVERS.keys())
    elif target in GAME_SOLVERS:
        games = [target]
    else:
        print(f"Unknown game: {target}")
        print(f"Available: {list(GAME_SOLVERS.keys())}")
        return

    for game in games:
        try:
            GAME_SOLVERS[game]()
        except Exception as e:
            print(f"  ERROR solving {game}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("Done. Check shared-context/arc-agi-3/visual-memory/ for outputs.")


if __name__ == "__main__":
    main()
