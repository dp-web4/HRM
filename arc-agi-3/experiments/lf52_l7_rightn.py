#!/usr/bin/env python3
"""Test what happens when we click right-N at (22,6) and try various clicks.
Key: the engine probably scrolls camera to center piece when selected.
Check offset change after selection."""
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
    print(f"L7 pristine: pieces={state['pieces']} offset={state['offset']}")
    # Do 1 no-op push that doesn't move anything to verify engine is ready
    fd = env.step(GameAction.ACTION1)  # UP
    eq = game.ikhhdzfmarl
    state_after_up = solver.extract_state(eq)
    print(f"After UP push: pieces={state_after_up['pieces']} offset={state_after_up['offset']} blocks={sorted(state_after_up['pushable'])}")

    # The frame is 64x64 but the grid is 30x30 cells at 6px each = 180px.
    # So camera scrolls. offset = pixel position of cell (0,0)?
    # (0,0) → pixel (5,5). (9,9) → pixel (5+9*6, 5+9*6)=(59,59). So cells 0-9 visible.
    # Right-N at (22,6) is off-screen. To interact, we must trigger camera scroll.
    # Let's click at the far-right edge of the visible frame repeatedly to trigger scroll.

    # Before any action:
    print(f"Frame-visible: cells ~0-9 with offset (5,5)")
    print(f"left-N@(0,1) pixel = (8, 14) — visible")
    print(f"right-N@(22,6) pixel = (140, 44) — off-screen")

    # Try clicking at the far right edge of frame (x=63, y=39-49) to see if scrolls
    # Actually, maybe select left-N first, then navigate.
    # ACTION6 at (8, 14) = click left-N
    fd = env.step(GameAction.ACTION6, data={'x': 8, 'y': 14})
    eq = game.ikhhdzfmarl
    state2 = solver.extract_state(eq)
    print(f"After click left-N: offset={state2['offset']} pieces={state2['pieces']}")

    # Try clicking arrow direction down: cell (0,3) → pixel (8, 26)
    fd = env.step(GameAction.ACTION6, data={'x': 8, 'y': 26})
    eq = game.ikhhdzfmarl
    state3 = solver.extract_state(eq)
    print(f"After click (0,3) arrow: offset={state3['offset']} pieces={state3['pieces']}")

    # Maybe click the completion/next button at (8, 56)
    fd = env.step(GameAction.ACTION6, data={'x': 8, 'y': 56})
    eq = game.ikhhdzfmarl
    state4 = solver.extract_state(eq)
    print(f"After click (8,56): offset={state4['offset']} pieces={state4['pieces']}")
    print(f"  levels_completed={fd.levels_completed} state={fd.state.name}")

    # Now try ACTION7 (undo) after having moved
    fd = env.step(GameAction.ACTION7)
    eq = game.ikhhdzfmarl
    state5 = solver.extract_state(eq)
    print(f"After ACTION7: offset={state5['offset']} pieces={state5['pieces']}")
    print(f"  blocks={sorted(state5['pushable'])}")

    # Try ACTION5 to see what it does
    fd = env.step(GameAction.ACTION5)
    eq = game.ikhhdzfmarl
    state6 = solver.extract_state(eq)
    print(f"After ACTION5: offset={state6['offset']} pieces={state6['pieces']}")

if __name__ == "__main__":
    main()
