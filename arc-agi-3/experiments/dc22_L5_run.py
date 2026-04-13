#!/usr/bin/env python3
"""Run solve_macro on L5 specifically with extended parameters."""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import solve_macro

VIS = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/dc22"

def replay_cached(env, cache_path, up_to_level):
    with open(cache_path) as f:
        raw = json.load(f)
    act_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
               3: GameAction.ACTION3, 4: GameAction.ACTION4,
               6: GameAction.ACTION6}
    for lvl_idx in range(up_to_level):
        for m in raw[lvl_idx]:
            a = act_map[m['action']]
            if 'data' in m:
                env.step(a, data=m['data'])
            else:
                env.step(a)

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    replay_cached(env, f"{VIS}/solutions.json", 4)
    sol = solve_macro(env, 4, timeout=1200, max_click_depth=20)
    print(f"Result: {'solved ' + str(len(sol)) + ' moves' if sol else 'FAILED'}")

if __name__ == "__main__":
    main()
