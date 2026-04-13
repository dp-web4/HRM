#!/usr/bin/env python3
"""Use the engine as oracle: after push sequence L L U U L L L, try to jump
N@(0,1) DOWN to (0,3). Verify engine actually performs the jump."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
from arcengine import GameAction
import lf52_solve_final as solver


def dump_state(eq, label):
    s = solver.extract_state(eq)
    print(f"[{label}] pieces={s['pieces']} blocks={sorted(s['pushable'])} off={s['offset']} steps={eq.asqvqzpfdi}")
    return s


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        solver.solve_level(env, game, lvl)

    eq = game.ikhhdzfmarl
    dump_state(eq, "pristine")

    # Push sequence to get block at (0,3): L L U U L L L
    seq = [('LEFT', GameAction.ACTION3), ('LEFT', GameAction.ACTION3),
           ('UP', GameAction.ACTION1), ('UP', GameAction.ACTION1),
           ('LEFT', GameAction.ACTION3), ('LEFT', GameAction.ACTION3),
           ('LEFT', GameAction.ACTION3)]
    for n, a in seq:
        env.step(a)
    dump_state(game.ikhhdzfmarl, "after LLUULLL")

    # Select N@(0,1) — pixel (8, 14) at cs=6 offset=(5,5)
    eq = game.ikhhdzfmarl
    off = eq.hncnfaqaddg.cdpcbbnfdp
    print(f"offset now: {off}")
    px = 0*6 + off[0] + 3
    py = 1*6 + off[1] + 3
    print(f"select N@(0,1) at pixel ({px},{py})")
    env.step(GameAction.ACTION6, data={'x': px, 'y': py})
    dump_state(game.ikhhdzfmarl, "after select")
    print(f"  sel obj: {getattr(game.ikhhdzfmarl, 'aufxjsaidrw', None)}")

    # Click DOWN arrow: src (0,1), dst (0,3), half_dy=1
    ax = 0*6 + off[0] + 0*12 + 3
    ay = 1*6 + off[1] + 1*12 + 3
    print(f"click DOWN arrow at ({ax},{ay})")
    env.step(GameAction.ACTION6, data={'x': ax, 'y': ay})
    dump_state(game.ikhhdzfmarl, "after DOWN arrow")

    # Also try UP arrow (should be invalid), LEFT/RIGHT arrow
    env.step(GameAction.ACTION6, data={'x': px, 'y': py})  # re-select
    print(f"\nRe-selected. Trying UP arrow at ({px},{py-12}):")
    env.step(GameAction.ACTION6, data={'x': px, 'y': py-12})
    dump_state(game.ikhhdzfmarl, "after UP arrow")

    # Now try: pristine L7 — can we just click RIGHT arrow on N@(0,1) without
    # any pushes? Middle (1,1) is empty — should fail.
    while game.ikhhdzfmarl.asqvqzpfdi > 0:
        env.step(GameAction.ACTION7)
    print(f"\n=== reset to pristine L7 ===")
    dump_state(game.ikhhdzfmarl, "reset")

    eq = game.ikhhdzfmarl
    off = eq.hncnfaqaddg.cdpcbbnfdp
    px = 0*6 + off[0] + 3
    py = 1*6 + off[1] + 3
    env.step(GameAction.ACTION6, data={'x': px, 'y': py})
    print(f"Selected (0,1). Now try all 4 arrows:")
    for name, (adx, ady) in [('UP',(0,-1)),('DOWN',(0,1)),('LEFT',(-1,0)),('RIGHT',(1,0))]:
        ax = 0*6 + off[0] + adx*12 + 3
        ay = 1*6 + off[1] + ady*12 + 3
        # Re-select first
        env.step(GameAction.ACTION6, data={'x': px, 'y': py})
        sb = dump_state(eq, f"before {name}")
        env.step(GameAction.ACTION6, data={'x': ax, 'y': ay})
        sa = dump_state(eq, f"after  {name} @({ax},{ay})")
        if sb['pieces'] != sa['pieces']:
            print(f"  !!! {name} MOVED PIECE !!!")


if __name__ == "__main__":
    main()
