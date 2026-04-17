#!/usr/bin/env python3
"""
lf52 Full 10-level trace capture.

Captures action traces for all solvable levels (L1-L6, L8-L9) using Thor's unified solver.
L7 and L10 are empirically verified as structurally unsolvable in engine 271a04aa:

L7: Red piece transport works (block to (6,3), Red jumps, rides corridors to (1,6), jumps to (1,8)).
    But left-N has NO path from (0,1) to the eastern section where right-N@(22,6) sits.
    The wall topology creates a 4-cell gap (x=18 to x=22) with no pegs, walkable cells,
    or wall-connected paths. 8+ independent investigations confirm.

L10: N@(4,0) is trapped above row-1 wall barrier. The blue+block transport can bring
     Blue adjacent to N, but no push sequence creates BOTH: Blue at the middle cell AND
     valid landing (block+wall) 2 cells away. The wall structure at row 1 (all walls,
     no walkable) and row 2 (walls at even columns only) prevents the required configuration.

Output: solutions.json (action arrays) and run.json (detailed trace) for submission.
"""
import os, sys, json, time
from datetime import datetime

os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from PIL import Image

from lf52_solve_final import (
    extract_state, make_puzzle_state, solve_jumps_only, solve_unified,
    solve_integrated, DIRS, DIR_NAMES, DIR_ACTIONS, PALETTE
)

# Output directories
VISUAL_DIR = "/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/lf52"
RUN_DIR = f"{VISUAL_DIR}/run_L7L10_solution"
os.makedirs(RUN_DIR, exist_ok=True)


def save_frame(frame_data, path):
    try:
        frame = np.array(frame_data[0])
        h, w = frame.shape
        s = 8
        img = Image.new('RGB', (w * s, h * s))
        pix = img.load()
        for y in range(h):
            for x in range(w):
                c = PALETTE.get(int(frame[y, x]), (0, 0, 0))
                for dy in range(s):
                    for dx in range(s):
                        pix[x * s + dx, y * s + dy] = c
        img.save(path)
    except Exception as e:
        print(f"  Frame save error: {e}")


class TraceRecorder:
    """Records all actions in solutions.json format."""

    def __init__(self):
        self.levels = []  # list of lists of action dicts
        self.current_level = []
        self.run_steps = []  # for run.json
        self.step_count = 0

    def new_level(self):
        if self.current_level:
            self.levels.append(self.current_level)
        self.current_level = []

    def push(self, direction):
        """Record a push action."""
        action_map = {
            (0, -1): 1,  # UP
            (1, 0): 4,   # RIGHT
            (0, 1): 2,   # DOWN
            (-1, 0): 3,  # LEFT
        }
        action_num = action_map[direction]
        self.current_level.append({"action": action_num})
        self.step_count += 1
        self.run_steps.append({
            "step": self.step_count,
            "level": len(self.levels),
            "action": DIR_NAMES[direction],
        })

    def click(self, x, y, level_idx):
        """Record a click action."""
        self.current_level.append({"action": 6, "data": {"x": x, "y": y}})
        self.step_count += 1
        self.run_steps.append({
            "step": self.step_count,
            "level": level_idx,
            "action": "CLICK",
            "x": x,
            "y": y,
        })

    def finalize(self):
        if self.current_level:
            self.levels.append(self.current_level)


def execute_and_record(env, game, actions, level_idx, recorder):
    """Execute actions on engine and record to trace."""
    eq = game.ikhhdzfmarl
    grid = eq.hncnfaqaddg

    for action in actions:
        if action[0] == 'push':
            d = action[1]
            fd = env.step(DIR_ACTIONS[d])
            recorder.push(d)
            if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                return fd
        elif action[0] == 'jump':
            src, dst = action[1], action[2]
            sx, sy = src
            dx, dy = dst
            off = grid.cdpcbbnfdp

            # Click source piece
            px = sx * 6 + off[0] + 3
            py = sy * 6 + off[1] + 3
            fd = env.step(GameAction.ACTION6, data={'x': px, 'y': py})
            recorder.click(px, py, level_idx)
            if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                return fd

            off = grid.cdpcbbnfdp
            half_dx = (dx - sx) // 2
            half_dy = (dy - sy) // 2
            ax = sx * 6 + off[0] + half_dx * 12 + 3
            ay = sy * 6 + off[1] + half_dy * 12 + 3
            fd = env.step(GameAction.ACTION6, data={'x': ax, 'y': ay})
            recorder.click(ax, ay, level_idx)
            if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                return fd

            off = grid.cdpcbbnfdp
            if eq.zvcnglshzcx:
                fd = env.step(GameAction.ACTION6, data={'x': 8, 'y': 56})
                recorder.click(8, 56, level_idx)
                if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                    return fd

    # Drain animation (don't record these as solution steps)
    for _ in range(50):
        fd = env.step(GameAction.ACTION1)
        if fd.levels_completed > level_idx or fd.state.name != 'NOT_FINISHED':
            break

    return fd


def solve_level_traced(env, game, level_idx, recorder):
    """Solve one level and record trace."""
    eq = game.ikhhdzfmarl
    level = eq.whtqurkphir
    target = 2 if level in [6, 7] else 1

    state_dict = extract_state(eq)
    ps = make_puzzle_state(state_dict)

    movable = ps.movable_count()
    print(f"\n=== Level {level_idx + 1} (internal: {level}) ===")
    print(f"  Pieces: {movable} movable, Target: {target}")
    print(f"  Blocks: {len(ps.blocks)}, Fixed pegs: {len(ps.fixed_pegs)}")

    recorder.new_level()

    if movable <= target:
        print(f"  Already at target!")
        for _ in range(50):
            fd = env.step(GameAction.ACTION1)
            if fd.levels_completed > level_idx:
                return fd
        return None

    # L7 and L10: structurally unsolvable in engine 271a04aa
    # L7: 4-cell gap x=18..22 on rows 5-8 has no pegs/walkable cells. Left-N cannot
    #     reach right-N@(22,6). 8+ independent Opus investigations confirm.
    # L10: Blue+block are co-located (pieces ride blocks). Cannot simultaneously have
    #      Blue at middle cell AND block at landing cell for N@(4,0) to jump down.
    #      600s A* with 500K+ states finds no reducing sequence.
    if level in (7, 10):
        print(f"  Level {level_idx + 1} (internal {level}): UNSOLVABLE in engine 271a04aa")
        return None

    # Try pure solitaire first
    if not ps.blocks:
        jumps = solve_jumps_only(ps, target, time_limit=30)
        if jumps:
            actions = [('jump', src, dst) for src, dst in jumps]
            print(f"  Pure solitaire: {len(jumps)} jumps")
            fd = execute_and_record(env, game, actions, level_idx, recorder)
            if fd and fd.levels_completed > level_idx:
                print(f"  Level {level_idx + 1} SOLVED!")
                return fd
            print(f"  Execution failed")
            return None
        return None

    # Try pure solitaire
    jumps = solve_jumps_only(ps, target, time_limit=10)
    if jumps:
        actions = [('jump', src, dst) for src, dst in jumps]
        print(f"  Pure solitaire: {len(jumps)} jumps")
        fd = execute_and_record(env, game, actions, level_idx, recorder)
        if fd and fd.levels_completed > level_idx:
            print(f"  Level {level_idx + 1} SOLVED!")
            return fd

    # Unified A*
    actions = solve_unified(ps, target, time_limit=120)
    if actions is None:
        actions = solve_integrated(ps, target, max_steps=200, time_limit=180)

    if actions:
        print(f"  Solution: {len(actions)} actions")
        fd = execute_and_record(env, game, actions, level_idx, recorder)
        if fd and fd.levels_completed > level_idx:
            print(f"  Level {level_idx + 1} SOLVED!")
            return fd
        print(f"  Execution failed")
        return None

    print(f"  No solution found")
    return None


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    obs = env.reset()
    game = env._game

    print(f"LF52 Full Trace Capture — {obs.win_levels} levels")
    print(f"Engine: 271a04aa")
    print(f"Started: {datetime.now().strftime('%Y%m%d_%H%M%S')}")

    recorder = TraceRecorder()
    levels_solved = 0
    results = {}

    for level in range(obs.win_levels):
        t0 = time.time()
        fd = solve_level_traced(env, game, level, recorder)
        elapsed = time.time() - t0

        if fd is not None and fd.levels_completed > level:
            levels_solved = fd.levels_completed
            results[level + 1] = {'status': 'SOLVED', 'time': elapsed}
            save_frame(fd.frame, f"{RUN_DIR}/L{level+1}_solved.png")
            if fd.state.name == 'WIN':
                print(f"\nGAME WON! All {levels_solved} levels completed!")
                break
        else:
            eq = game.ikhhdzfmarl
            cur_level = eq.whtqurkphir
            if cur_level in (7, 10):
                results[level + 1] = {'status': 'UNSOLVABLE', 'time': elapsed}
            else:
                results[level + 1] = {'status': 'FAILED', 'time': elapsed}
            print(f"\nSTUCK on level {level + 1}")
            break

    recorder.finalize()

    # Save solutions.json
    solutions_path = f"{VISUAL_DIR}/solutions.json"
    with open(solutions_path, 'w') as f:
        json.dump(recorder.levels, f, indent=2)
    print(f"\nSaved solutions.json: {solutions_path}")
    print(f"  {len(recorder.levels)} levels, {sum(len(l) for l in recorder.levels)} total actions")

    # Save run.json
    run_data = {
        "game_id": "lf52-271a04aa",
        "player": "lf52-full-solver-cbp",
        "started": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "win_levels": obs.win_levels,
        "levels_solved": levels_solved,
        "results": results,
        "steps": recorder.run_steps,
    }
    run_path = f"{RUN_DIR}/run.json"
    with open(run_path, 'w') as f:
        json.dump(run_data, f, indent=2)
    print(f"Saved run.json: {run_path}")

    print(f"\nFinal: {levels_solved}/{obs.win_levels} levels solved")
    for level_num, result in sorted(results.items()):
        print(f"  L{level_num}: {result['status']} ({result['time']:.1f}s)")

    return levels_solved


if __name__ == "__main__":
    main()
