#!/usr/bin/env python3
"""Empirically test what counts as a 'jumpable middle'. Is a block alone
enough? A peg? Both?"""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
from arcengine import GameAction
import lf52_solve_final as solver


def click(env, x, y):
    env.step(GameAction.ACTION6, data={'x': x, 'y': y})


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        solver.solve_level(env, game, lvl)
    eq = game.ikhhdzfmarl

    # Pristine L7: N@(0,1), block NOT at (0,2) (peg there). Let's try:
    # Push block to (14,1) — right? pristine has (14,2) block, UP push → (14,1) block.
    # After UP: blocks at (14,1),(18,2),(19,2),(20,2),(22,3),(5,5)
    # Can R@(6,1) jump somewhere new? Middle cells of R are (6,0),(6,2),(5,1),(7,1).
    # None changed.
    # How about N@(0,1)? Middle cells are (0,0),(0,2),(-1,1),(1,1). Only (0,2) has peg.
    # Landing (0,3) — NO block yet (block is at (5,5)). And (0,3) has kraubslpehi-L (wall, not walkable). So not a valid landing.
    # UNLESS we push a block TO (0,3). That's what LLUULLL achieves.

    # Interesting: let's push (5,5) block LEFT 5 times.
    # (5,5) → (4,5)? needs (4,5) to have wall. yes, (4,5) has kraubslpehi. → (4,5).
    # → (3,5)? (3,5) has kraubslpehi-t → (3,5). → (2,5)? (2,5) kraubslpehi → (2,5).
    # → (1,5)? (1,5) kraubslpehi-< → (1,5). → (0,5)? (0,5) empty. STOP.
    # So block reaches (1,5). Not (0,5) because no wall.
    # But we need it at (0,3) for left-N jump.
    # UP from (1,5)? (1,4) empty. STOP.
    # Actually the solver's LLUULLL trace brought (5,5) → (0,3). How?
    # Trace: L L U U L L L.
    # (5,5) L→ (4,5) L→ (3,5). UP: (3,5)→(3,4)? (3,4) has kraubslpehi-up. → (3,4). UP: (3,4)→(3,3)? (3,3) kraubslpehi-T. → (3,3). L: (3,3)→(2,3)? (2,3) kraubslpehi. → (2,3). L: →(1,3) kraubslpehi. L: →(0,3) kraubslpehi-L.
    # So (5,5) → (0,3) via those pushes. Good.
    #
    # Now N@(0,1) DOWN jump: middle (0,2)=peg, landing (0,3)=block+kraubslpehi.
    # Is (0,3) a valid landing? Rule 2: len==2 and hupkpseyuim2 in names → block exists, so YES.
    # We verified this works: N jumps (0,1) → (0,3).

    # OK, so pursue the original push path + explore from there.
    # After N at (0,3), what can it do?
    # Need to find: can N at (0,3) reach (6,1) area? Can it meet R?
    # Then: can R be pushed/moved/removed?

    print("=== Push LLUULLL then jump N DOWN ===")
    for a in [GameAction.ACTION3, GameAction.ACTION3, GameAction.ACTION1,
              GameAction.ACTION1, GameAction.ACTION3, GameAction.ACTION3,
              GameAction.ACTION3]:
        env.step(a)
    eq = game.ikhhdzfmarl
    st_pre = solver.extract_state(eq)
    print(f"pre: pieces={st_pre['pieces']}")
    print(f"pre: blocks={sorted(st_pre['pushable'])}")

    # Select N@(0,1) at (8,14)
    click(env, 8, 14)
    click(env, 8, 26)  # DOWN arrow
    st_post = solver.extract_state(game.ikhhdzfmarl)
    print(f"post jump: pieces={st_post['pieces']}")
    print(f"post: blocks={sorted(st_post['pushable'])}")

    # Now N is at (0,3). Compute valid jumps from (0,3) per model
    ps = solver.make_puzzle_state(st_post)
    print(f"\nValid jumps from post-state: {ps.get_valid_jumps()}")

    # Try each — empirically test what the engine actually accepts
    for src, dst in ps.get_valid_jumps():
        print(f"\n  trying engine jump {src}->{dst}")
        eq = game.ikhhdzfmarl
        off = eq.hncnfaqaddg.cdpcbbnfdp
        sx, sy = src
        dx, dy = dst
        px = sx*6 + off[0] + 3
        py = sy*6 + off[1] + 3
        if not (0<=px<64 and 0<=py<64):
            print(f"    src px({px},{py}) off-screen")
            continue
        env.step(GameAction.ACTION7)  # undo arrow click if any
        click(env, px, py)
        half_dx = (dx - sx) // 2
        half_dy = (dy - sy) // 2
        ax = sx*6 + off[0] + half_dx*12 + 3
        ay = sy*6 + off[1] + half_dy*12 + 3
        if not (0<=ax<64 and 0<=ay<64):
            print(f"    arrow px({ax},{ay}) off-screen")
            continue
        st_b = solver.extract_state(eq)
        click(env, ax, ay)
        st_a = solver.extract_state(eq)
        moved = st_b['pieces'] != st_a['pieces']
        print(f"    moved={moved} new_pieces={st_a['pieces']}")
        if moved:
            # undo to restore
            env.step(GameAction.ACTION7)
            env.step(GameAction.ACTION7)


if __name__ == "__main__":
    main()
