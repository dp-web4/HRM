#!/usr/bin/env python3
"""Verify what solve_level(6) actually reads as L7 pristine state, vs what
my empirical script read. Print the pristine initial state exactly."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
from arcengine import GameAction
import lf52_solve_final as solver

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game

    for lvl in range(6):
        fd = solver.solve_level(env, game, lvl)
        if fd is None: return

    # What solve_level(6) reads:
    eq = game.ikhhdzfmarl
    state = solver.extract_state(eq)
    print(f"L7 init from eq refetch:")
    print(f"  level={state['level']} offset={state['offset']}")
    print(f"  pieces={state['pieces']}")
    print(f"  blocks={sorted(state['pushable'])}")
    print(f"  pegs={sorted(state['fixed_pegs'])}")
    ps = solver.make_puzzle_state(state)
    print(f"  movable_count={ps.movable_count()}")
    print(f"  valid_jumps from init: {ps.get_valid_jumps()}")
    # Sanity: do any pegs overlap blocks?
    overlaps = set(state['pushable']) & set(state['fixed_pegs'])
    print(f"  peg-on-block overlaps: {sorted(overlaps)}")

if __name__ == "__main__":
    main()
