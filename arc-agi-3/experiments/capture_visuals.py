#!/usr/bin/env python3
"""
Visual Memory Capture — Generate frame captures for solved ARC-AGI-3 games.

Strategy: Run the BFS solver with env.step monkey-patched to collect click
coordinates, then replay through a fresh env.step() with SessionWriter to
capture frames as PNGs + GIFs.

Games handled:
  - s5i5: arrow resize/rotate puzzle (8 levels, BFS solver)
  - lp85: cyclic rotation puzzle (8 levels, BFS solver)
  - r11l: creature movement puzzle (6 levels, centroid BFS solver)

Usage:
  python3 capture_visuals.py [game_id]
  python3 capture_visuals.py          # all games
  python3 capture_visuals.py s5i5     # just s5i5
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_all_frames
from arc_session_writer import SessionWriter

CLICK = 6


def replay_with_capture(game_id, click_sequence, label="capture"):
    """Replay a click sequence through the game engine with visual capture.

    click_sequence: list of (x, y) display coordinates
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

    for x, y in click_sequence:
        fd = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
        all_frames = get_all_frames(fd)
        grid = all_frames[-1]
        sw.record_step(CLICK, grid,
                       all_frames=all_frames if len(all_frames) > 1 else None,
                       levels_completed=fd.levels_completed,
                       x=x, y=y,
                       state=fd.state.name if hasattr(fd.state, 'name') else str(fd.state))
        if fd.state.name in ("GAME_OVER", "WON"):
            break

    sw.record_game_end(fd.state.name, fd.levels_completed)
    files = [f for f in os.listdir(sw.run_dir) if f.endswith('.png')]
    gifs = [f for f in os.listdir(sw.run_dir) if f.endswith('.gif')]
    print(f"  {game_id}: {fd.levels_completed}/{fd.win_levels} "
          f"in {sw.step} steps, {len(files)} PNGs, {len(gifs)} GIFs")
    print(f"  Saved to: {sw.run_dir}")
    return fd.levels_completed, fd.win_levels, sw.step


# ═══════════════════════════════════════════════════════════
# ClickCollector: monkey-patch env.step to collect click coords
# ═══════════════════════════════════════════════════════════

class ClickCollector:
    """Wraps an env to collect all click coordinates while solving."""

    def __init__(self, env):
        self._env = env
        self.clicks = []
        self._orig_step = env.step

    def step(self, action, data=None):
        if data and 'x' in data:
            self.clicks.append((data['x'], data['y']))
        return self._orig_step(action, data=data)

    def patch(self):
        self._env.step = self.step

    def unpatch(self):
        self._env.step = self._orig_step


def _solve_with_collection(game_id, solver_script, label, pre_paths=None):
    """Generic solver runner: exec solver script with monkey-patched env.step,
    collect clicks, then replay through fresh game with SessionWriter.

    Args:
        game_id: full game ID (e.g. 's5i5-a48e4b1d')
        solver_script: filename in experiments/ (e.g. 's5i5_solve.py')
        label: player label for visual memory
        pre_paths: list of paths to add to sys.path before exec
    """
    game_prefix = game_id.split('-')[0]
    print(f"\n=== Solving {game_prefix} (collecting clicks) ===")

    solver_path = os.path.join(os.path.dirname(__file__), solver_script)

    with open(solver_path) as f:
        source = f.read()

    main_marker = "\ntotal_actions = 0\n"
    if main_marker not in source:
        print(f"  ERROR: Can't find main loop marker in {solver_script}")
        return None

    setup_code, main_code = source.split(main_marker, 1)
    setup_code += "\ntotal_actions = 0\n"

    # Add any required paths before exec
    if pre_paths:
        for p in pre_paths:
            if p not in sys.path:
                sys.path.insert(0, p)

    ns = {'__file__': solver_path}
    exec(compile(setup_code, solver_path, 'exec'), ns)

    env = ns['env']
    collector = ClickCollector(env)
    collector.patch()

    exec(compile(main_code, solver_path, 'exec'), ns)

    collector.unpatch()

    print(f"  Collected {len(collector.clicks)} clicks")

    if collector.clicks:
        replay_with_capture(game_id, collector.clicks, label)

    return collector.clicks


def solve_s5i5_with_collection():
    return _solve_with_collection('s5i5-a48e4b1d', 's5i5_solve.py', 'bfs-solver')


def solve_lp85_with_collection():
    pre = [os.path.join(os.path.dirname(__file__),
                        'environment_files', 'lp85', '305b61c3')]
    return _solve_with_collection('lp85-305b61c3', 'lp85_solve.py', 'bfs-solver',
                                  pre_paths=pre)


def solve_r11l_with_collection():
    return _solve_with_collection('r11l-aa269680', 'r11l_solve.py', 'bfs-solver')


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("Visual Memory Capture for ARC-AGI-3 Games")
    print("=" * 50)

    if target in ("all", "s5i5"):
        solve_s5i5_with_collection()

    if target in ("all", "lp85"):
        solve_lp85_with_collection()

    if target in ("all", "r11l"):
        solve_r11l_with_collection()

    print("\n" + "=" * 50)
    print("Done. Check shared-context/arc-agi-3/visual-memory/ for outputs.")


if __name__ == "__main__":
    main()
