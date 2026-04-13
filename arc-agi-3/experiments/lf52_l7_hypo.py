#!/usr/bin/env python3
"""Test hypotheses for L7 hidden mechanics:
H1: pushing a block INTO a piece removes the piece
H2: jumping onto an occupied cell removes target
H3: peg-on-piece overlap can be created and reduces count
H4: clicking a piece with another piece selected does something
H5: red piece can jump ANYTHING (including N) and the jumper matches rule is
    actually "piece kinds are compatible" not "exactly equal"
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


def st(eq):
    return solver.extract_state(eq)


def select(env, game, grid_pos, off=None):
    eq = game.ikhhdzfmarl
    if off is None:
        off = eq.hncnfaqaddg.cdpcbbnfdp
    px = grid_pos[0]*6 + off[0] + 3
    py = grid_pos[1]*6 + off[1] + 3
    if not (0 <= px < 64 and 0 <= py < 64):
        return False
    env.step(GameAction.ACTION6, data={'x': px, 'y': py})
    eq = game.ikhhdzfmarl
    return getattr(eq, 'aufxjsaidrw', None) is not None


def click_arrow(env, game, grid_src, dir_dxdy, off=None):
    eq = game.ikhhdzfmarl
    if off is None:
        off = eq.hncnfaqaddg.cdpcbbnfdp
    ax = grid_src[0]*6 + off[0] + dir_dxdy[0]*12 + 3
    ay = grid_src[1]*6 + off[1] + dir_dxdy[1]*12 + 3
    env.step(GameAction.ACTION6, data={'x': ax, 'y': ay})


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        solver.solve_level(env, game, lvl)
    eq = game.ikhhdzfmarl
    print(f"pristine: {st(eq)['pieces']}")

    # === H1: Push block INTO a piece
    # On pristine L7: push DOWN twice moves (22,4)→(22,6)? No, (22,6) has N. What happens?
    print("\n=== H1: push DOWN ===")
    s0 = st(eq)
    env.step(GameAction.ACTION2)  # DOWN
    s1 = st(eq)
    print(f"  blocks before: {s0['pushable']}")
    print(f"  blocks after:  {s1['pushable']}")
    print(f"  pieces after:  {s1['pieces']}")
    env.step(GameAction.ACTION7)  # undo

    # push DOWN DOWN DOWN on pristine — see if block reaches right-N
    for _ in range(5):
        fd = env.step(GameAction.ACTION2)
    s = st(game.ikhhdzfmarl)
    print(f"\n  after 5x DOWN: blocks={s['pushable']} pieces={s['pieces']}")

    # Reset
    while game.ikhhdzfmarl.asqvqzpfdi > 0:
        env.step(GameAction.ACTION7)
    print(f"\n=== reset: steps={game.ikhhdzfmarl.asqvqzpfdi} ===")

    # === H2: select a piece and click ANOTHER piece (not an arrow)
    print("\n=== H2: select N@(0,1), click R@(6,1) directly ===")
    select(env, game, (0, 1))
    eq = game.ikhhdzfmarl
    print(f"  sel after N click: {getattr(eq,'aufxjsaidrw',None) is not None}")
    off = eq.hncnfaqaddg.cdpcbbnfdp
    # Click R@(6,1) = pixel (44, 14)
    env.step(GameAction.ACTION6, data={'x': 44, 'y': 14})
    s = st(game.ikhhdzfmarl)
    print(f"  pieces after: {s['pieces']}")

    # Reset
    while game.ikhhdzfmarl.asqvqzpfdi > 0:
        env.step(GameAction.ACTION7)

    # === H3: select red, click DOWN arrow
    print("\n=== H3: can red jump? red has no valid jumps in the model. ===")
    select(env, game, (6, 1))
    click_arrow(env, game, (6, 1), (0, 1))  # DOWN
    s = st(game.ikhhdzfmarl)
    print(f"  after red DOWN: {s['pieces']}")
    env.step(GameAction.ACTION7)
    env.step(GameAction.ACTION7)

    select(env, game, (6, 1))
    click_arrow(env, game, (6, 1), (-1, 0))  # LEFT — middle (5,1)?
    s = st(game.ikhhdzfmarl)
    print(f"  after red LEFT: {s['pieces']}")

    while game.ikhhdzfmarl.asqvqzpfdi > 0:
        env.step(GameAction.ACTION7)

    # === H4: PUSH blocks around right-N to see if any reach it
    print(f"\n=== H4: can we PUSH blocks to (22,5),(21,6),(22,7) to unlock right-N? ===")
    # (22,5) is walkable-only: can blocks settle there? Probably not since no wall.
    # But right-N's adjacent (22,5) is a peg? No. pegs list: (0,2),(1,7),(2,8),...
    s = st(game.ikhhdzfmarl)
    print(f"  pegs near right-N: {[p for p in s['fixed_pegs'] if abs(p[0]-22)<=3 and abs(p[1]-6)<=3]}")
    print(f"  walls near right-N: {sorted([w for w in s['walls'] if abs(w[0]-22)<=3 and abs(w[1]-6)<=3])}")
    print(f"  walkable near right-N: {sorted([w for w in s['walkable'] if abs(w[0]-22)<=3 and abs(w[1]-6)<=3])}")
    print(f"  blocks near right-N: {[b for b in s['pushable'] if abs(b[0]-22)<=3 and abs(b[1]-6)<=3]}")

    # === H5: Try to DIRECTLY select right-N (off-screen click). What if we set
    # a click at larger pixel — the engine clamps to 64x64 frame, so no.
    # But we CAN click in scene-graph space via env.step's data dict.
    # Maybe other keys are accepted?
    print(f"\n=== H5: try clicking right-N at px=(137,41) beyond frame ===")
    env.step(GameAction.ACTION6, data={'x': 137, 'y': 41})
    eq = game.ikhhdzfmarl
    print(f"  sel: {getattr(eq,'aufxjsaidrw',None) is not None}")
    # Or via grid coords?
    env.step(GameAction.ACTION6, data={'x': 22, 'y': 6, 'grid_x': 22, 'grid_y': 6})
    eq = game.ikhhdzfmarl
    print(f"  sel with grid_x/y: {getattr(eq,'aufxjsaidrw',None) is not None}")

    # === H6: What if the cwyrzsciwms button exists and clicking it with a
    # selected piece does something special?
    while game.ikhhdzfmarl.asqvqzpfdi > 0:
        env.step(GameAction.ACTION7)
    print(f"\n=== H6: look for cwyrzsciwms in scene graph ===")
    eq = game.ikhhdzfmarl
    grid = eq.hncnfaqaddg
    for y in range(30):
        for x in range(30):
            for o in grid.ijpoqzvnjt(x, y):
                if 'cwyrzsciwms' in o.name:
                    print(f"  found cwyrzsciwms@({x},{y}) name={o.name}")
    print(f"  zvcnglshzcx flag: {getattr(eq, 'zvcnglshzcx', 'N/A')}")

    # H7: scan the ENTIRE scene graph for any object-type we haven't seen
    print(f"\n=== H7: all unique object names in L7 scene graph ===")
    names = {}
    for y in range(30):
        for x in range(30):
            for o in grid.ijpoqzvnjt(x, y):
                names.setdefault(o.name, []).append((x, y))
    for n, ps in sorted(names.items()):
        print(f"  {n}: {len(ps)} at {ps[:5]}{'...' if len(ps)>5 else ''}")


if __name__ == "__main__":
    main()
