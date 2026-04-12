#!/usr/bin/env python3
"""Test what clicking pegs with piece selected actually does by checking
piece position after click."""
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
        solver.solve_level(env, game, lvl)

    eq = game.ikhhdzfmarl
    state = solver.extract_state(eq)
    off = state['offset']
    print(f"L7 pristine: pieces={state['pieces']} off={off}")

    # Select right-N at (22,6): pixel = (22*6+5+3, 6*6+5+3) = (140, 44)
    # But frame is 64x64! Pixel (140,44) is out of bounds.
    # So the grid doesn't fit in the frame — the engine SCROLLS.
    # offset likely adjusts rendering.
    # Let's look for click responses at actual frame locations.

    # The frame is 64x64. Cells rendered at... let me find the actual cell-to-pixel.
    # offset=(5,5) might mean the top-left cell visible is at some position.
    # Actually the solver treats offset as pixel base: cell*6+offset+3.
    # If cells are 6px wide and frame is 64px, only ~9-10 cells fit horizontally.
    # But grid is 30 wide. So the camera scrolls.
    # After L6 ending at (20,8), camera likely scrolled right. Offset (5,5) may
    # mean "col 0 visible at pixel 5" which fits cells 0-9 (pixel 5-59).
    # Right-N at (22,6) would be at pixel 22*6+5=137 — off-screen.

    # Verify by clicking on (0,1) and (6,1) to see if selection works there:
    # Pieces on screen (cols 0-9): (0,1) and (6,1).

    # Actually, let's just try making a known-good move: push DOWN and see if
    # a new jump becomes available, then EXECUTE that jump via select+arrow.
    # First see jumps after pushes.

    ps = solver.make_puzzle_state(state)
    # Push down 1
    ps1 = ps.apply_push((0,1))
    print(f"After 1 DOWN push: blocks={sorted(ps1.blocks)}")
    print(f"  valid jumps: {ps1.get_valid_jumps()}")

    # Push left 5 (from pristine)
    ps2 = ps
    for _ in range(5):
        p = ps2.apply_push((-1,0))
        if p is None: break
        ps2 = p
    print(f"After 5 LEFT pushes: blocks={sorted(ps2.blocks)}")
    print(f"  valid jumps: {ps2.get_valid_jumps()}")

    # Execute on engine: push left 5
    for _ in range(5):
        env.step(GameAction.ACTION3)
    eq = game.ikhhdzfmarl
    state2 = solver.extract_state(eq)
    ps_eng = solver.make_puzzle_state(state2)
    print(f"Engine after 5 LEFT: blocks={sorted(ps_eng.blocks)}")
    print(f"  valid jumps: {ps_eng.get_valid_jumps()}")

    # Now try LEFT LEFT UP UP LEFT LEFT LEFT (the 7-push to get first jump per earlier)
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        solver.solve_level(env, game, lvl)
    seq = [(-1,0),(-1,0),(0,-1),(0,-1),(-1,0),(-1,0),(-1,0)]
    for d in seq:
        act = {(0,-1):GameAction.ACTION1,(0,1):GameAction.ACTION2,
               (-1,0):GameAction.ACTION3,(1,0):GameAction.ACTION4}[d]
        env.step(act)
    eq = game.ikhhdzfmarl
    state3 = solver.extract_state(eq)
    ps3 = solver.make_puzzle_state(state3)
    print(f"\nAfter 7-push sequence: blocks={sorted(ps3.blocks)}")
    print(f"  pegs on blocks: {sorted(set(ps3.blocks)&set(ps3.fixed_pegs))}")
    print(f"  pieces: {sorted(ps3.pieces)}")
    print(f"  valid jumps: {ps3.get_valid_jumps()}")
    print(f"  offset: {state3['offset']}")

if __name__ == "__main__":
    main()
