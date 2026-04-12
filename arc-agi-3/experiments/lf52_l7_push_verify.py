#!/usr/bin/env python3
"""
Incremental push verification: at each step, compare model vs engine.
Single env run. Pushes a BFS-like walk to cover many states.
Any divergence = missing mechanic.
"""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
from arcengine import GameAction
import lf52_solve_final as solver

DIR_ACTIONS = {(0,-1): GameAction.ACTION1, (0,1): GameAction.ACTION2,
               (-1,0): GameAction.ACTION3, (1,0): GameAction.ACTION4}
DIR_NAMES = {(0,-1):'UP',(0,1):'DOWN',(-1,0):'LEFT',(1,0):'RIGHT'}
DIRS = [(0,-1),(0,1),(-1,0),(1,0)]

def tup(ps):
    return (ps.pieces, ps.blocks, ps.fixed_pegs)

def state_to_ps(state):
    return solver.make_puzzle_state(state)

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        fd = solver.solve_level(env, game, lvl)
        if fd is None: return

    eq = game.ikhhdzfmarl
    init = solver.extract_state(eq)
    ps = solver.make_puzzle_state(init)
    print(f"L7 pristine blocks={sorted(ps.blocks)} pegs_on_blocks={sorted(set(ps.blocks)&set(ps.fixed_pegs))}")
    print(f"pieces={sorted(ps.pieces)}")

    # Walk: try each direction one at a time, compare each transition,
    # continue walking regardless of divergence (so we see full trajectory).
    mismatches = 0
    rng_seq = [
        (0,-1), (0,-1), (0,-1), (0,1), (0,1), (1,0), (1,0), (1,0),
        (-1,0), (-1,0), (0,-1), (0,1), (1,0), (-1,0), (0,-1), (0,-1),
        (1,0), (1,0), (1,0), (0,1), (0,1), (0,1), (-1,0), (-1,0),
    ]
    for i, d in enumerate(rng_seq):
        # Model prediction
        model_next = ps.apply_push(d)
        # Engine truth
        fd = env.step(DIR_ACTIONS[d])
        eq = game.ikhhdzfmarl
        eng_state = solver.extract_state(eq)
        eng_ps = solver.make_puzzle_state(eng_state)

        # If model says no change, set model_next = ps
        if model_next is None:
            model_next = ps

        if tup(model_next) != tup(eng_ps):
            mismatches += 1
            print(f"\n!!! STEP {i} PUSH {DIR_NAMES[d]} DIVERGENCE")
            b_m = sorted(model_next.blocks); b_e = sorted(eng_ps.blocks)
            p_m = sorted(model_next.pieces); p_e = sorted(eng_ps.pieces)
            g_m = sorted(model_next.fixed_pegs); g_e = sorted(eng_ps.fixed_pegs)
            if b_m != b_e:
                print(f"  blocks: model={b_m}")
                print(f"  blocks: eng  ={b_e}")
                print(f"  blocks: only_model={set(b_m)-set(b_e)} only_eng={set(b_e)-set(b_m)}")
            if p_m != p_e:
                print(f"  pieces: only_model={set(p_m)-set(p_e)} only_eng={set(p_e)-set(p_m)}")
            if g_m != g_e:
                print(f"  pegs: only_model={set(g_m)-set(g_e)} only_eng={set(g_e)-set(g_m)}")

        # Use engine state as next for further comparison
        ps = eng_ps
        print(f"step {i:2} PUSH {DIR_NAMES[d]:5} blocks={sorted(ps.blocks)} pegs_on_blocks={sorted(set(ps.blocks)&set(ps.fixed_pegs))}")

    print(f"\nTotal mismatches: {mismatches}/{len(rng_seq)}")

if __name__ == "__main__":
    main()
