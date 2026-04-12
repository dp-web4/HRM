#!/usr/bin/env python3
"""
lf52 L7 empirical exploration.

RULE 0: Do not read source code. Treat engine as an oracle.
Advance to L7 via the existing solver, then probe systematically.

Key gotcha discovered: after solve_level(n) drains, extract_state returns a
stale transition state. The NEXT solve_level call's internal save_frame or
obs.frame access triggers the real level load. So we must do one pass of
frame access to refresh state before probing.
"""
import os, sys, time, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction


def extract_state(eq):
    grid = eq.hncnfaqaddg
    walkable, pushable, walls, pieces, fixed_pegs = set(), set(), set(), {}, set()
    for y in range(30):
        for x in range(30):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            if 'hupkpseyuim' in names:
                walkable.add((x, y))
            if 'hupkpseyuim2' in names:
                pushable.add((x, y))
            for name in names:
                if 'kraubslpehi' in name:
                    walls.add((x, y))
                if name == 'fozwvlovdui':
                    pieces[(x, y)] = 'fozwvlovdui'
                elif name == 'fozwvlovdui_red':
                    pieces[(x, y)] = 'fozwvlovdui_red'
                elif name == 'fozwvlovdui_blue':
                    pieces[(x, y)] = 'fozwvlovdui_blue'
                if name == 'dgxfozncuiz':
                    fixed_pegs.add((x, y))
    return {
        'walkable': walkable, 'pushable': pushable, 'walls': walls,
        'pieces': pieces, 'fixed_pegs': fixed_pegs,
        'level': eq.whtqurkphir, 'offset': grid.cdpcbbnfdp,
        'steps': eq.asqvqzpfdi,
    }


def state_signature(eq):
    grid = eq.hncnfaqaddg
    sig = {}
    for y in range(30):
        for x in range(30):
            names = tuple(sorted(o.name for o in grid.ijpoqzvnjt(x, y)))
            if names:
                sig[(x, y)] = names
    return sig


def diff_sigs(a, b):
    out = []
    for k in set(a) | set(b):
        if a.get(k) != b.get(k):
            out.append((k, a.get(k), b.get(k)))
    return sorted(out)


def print_grid(state):
    gc = [['.' for _ in range(30)] for _ in range(30)]
    for x, y in state['walkable']: gc[y][x] = ' '
    for x, y in state['walls']:    gc[y][x] = '#'
    for x, y in state['pushable']: gc[y][x] = 'B'
    for x, y in state['fixed_pegs']:gc[y][x] = 'P'
    for (x, y), n in state['pieces'].items():
        gc[y][x] = {'fozwvlovdui':'N','fozwvlovdui_red':'R','fozwvlovdui_blue':'b'}[n]
    print("    " + "".join(str(x % 10) for x in range(30)))
    for y in range(30):
        print(f"{y:3} " + "".join(gc[y]))


def px(cell, offset):
    return (cell[0]*6 + offset[0] + 3, cell[1]*6 + offset[1] + 3)


def main():
    import lf52_solve_final as solver
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    obs = env.reset()
    game = env._game
    eq = game.ikhhdzfmarl

    # Advance to L7
    for lvl in range(6):
        fd = solver.solve_level(env, game, lvl)
        if fd is None or fd.levels_completed <= lvl:
            print(f"FAIL reaching L7 at L{lvl+1}")
            return

    # Maybe game.ikhhdzfmarl refreshes on access
    print(f"\neq before refresh: id={id(eq)} level={eq.whtqurkphir}")
    eq2 = game.ikhhdzfmarl
    print(f"eq after game.ikhhdzfmarl: id={id(eq2)} level={eq2.whtqurkphir}")
    # Try re-getting game
    game2 = env._game
    eq3 = game2.ikhhdzfmarl
    print(f"eq via env._game: id={id(eq3)} level={eq3.whtqurkphir}")
    # Try obs
    print(f"obs.frame access...")
    _ = env.observation_space.frame
    eq4 = game.ikhhdzfmarl
    print(f"eq after frame: id={id(eq4)} level={eq4.whtqurkphir}")
    # DO NOT step — that mutates. Just re-fetch eq.
    eq = eq2

    state = extract_state(eq)
    print(f"\n=== L7 state ===")
    print(f"level={state['level']} offset={state['offset']} steps={state['steps']}")
    print(f"pieces ({len(state['pieces'])}): {state['pieces']}")
    print(f"blocks ({len(state['pushable'])}): {sorted(state['pushable'])}")
    print(f"fixed_pegs ({len(state['fixed_pegs'])}): {sorted(state['fixed_pegs'])}")
    print(f"walkable ({len(state['walkable'])})")
    print(f"walls ({len(state['walls'])}): {sorted(state['walls'])[:20]}")
    print_grid(state)

    if len(state['pieces']) != 3:
        print("!!! Not L7 yet, aborting probe")
        return

    offset = state['offset']
    base_sig = state_signature(eq)

    # EXPT A: ACTION7 (undo) before any action
    print("\n--- A: ACTION7 from fresh L7 ---")
    fd = env.step(GameAction.ACTION7)
    sig = state_signature(eq)
    d = diff_sigs(base_sig, sig)
    print(f"  diffs={len(d)} state={fd.state.name} steps={eq.asqvqzpfdi}")
    for x in d[:10]: print(f"    {x}")

    # EXPT B: ACTION5 no selection
    print("\n--- B: ACTION5 no select ---")
    sigb = state_signature(eq)
    fd = env.step(GameAction.ACTION5)
    siga = state_signature(eq)
    d = diff_sigs(sigb, siga)
    print(f"  diffs={len(d)} state={fd.state.name} steps={eq.asqvqzpfdi}")
    for x in d[:10]: print(f"    {x}")

    # EXPT C: Click each fixed peg directly (no selection)
    print("\n--- C: click each fixed peg (no selection) ---")
    for peg in sorted(state['fixed_pegs']):
        sigb = state_signature(eq)
        p = px(peg, state['offset'])
        fd = env.step(GameAction.ACTION6, data={'x': p[0], 'y': p[1]})
        siga = state_signature(eq)
        d = diff_sigs(sigb, siga)
        if d or fd.state.name != 'NOT_FINISHED':
            print(f"  peg {peg} px{p}: diffs={len(d)} state={fd.state.name}")
            for x in d[:8]: print(f"    {x}")

    # Refresh offset (may have scrolled)
    state = extract_state(eq)
    offset = state['offset']
    print(f"\n  After peg clicks: offset={offset}")
    if len(state['pieces']) < 3:
        print(f"  !!! piece count changed: {state['pieces']}")

    # EXPT D: Click blocks
    print("\n--- D: click each block (no selection) ---")
    for blk in sorted(state['pushable']):
        sigb = state_signature(eq)
        p = px(blk, offset)
        fd = env.step(GameAction.ACTION6, data={'x': p[0], 'y': p[1]})
        siga = state_signature(eq)
        d = diff_sigs(sigb, siga)
        if d:
            print(f"  block {blk}: diffs={len(d)} state={fd.state.name}")
            for x in d[:8]: print(f"    {x}")

    # EXPT E: Select each piece, click each peg
    print("\n--- E: select piece, click each peg ---")
    state = extract_state(eq)
    offset = state['offset']
    for pp, pn in list(state['pieces'].items()):
        sigb = state_signature(eq)
        p = px(pp, offset)
        fd = env.step(GameAction.ACTION6, data={'x': p[0], 'y': p[1]})
        siga = state_signature(eq)
        sd = diff_sigs(sigb, siga)
        print(f"\n  select {pn}@{pp}: diffs={len(sd)} state={fd.state.name}")
        for x in sd[:8]: print(f"    {x}")

        for peg in sorted(state['fixed_pegs']):
            sb = state_signature(eq)
            pp2 = px(peg, offset)
            fd = env.step(GameAction.ACTION6, data={'x': pp2[0], 'y': pp2[1]})
            sa = state_signature(eq)
            dd = diff_sigs(sb, sa)
            if dd and len(dd) > 2:
                print(f"    -> peg {peg}: {len(dd)} diffs state={fd.state.name}")
                for x in dd[:6]: print(f"      {x}")
            if fd.state.name == 'WIN' or fd.levels_completed > 6:
                print("    !!! WIN / level advance !!!")
                return

    # EXPT F: ACTION5 after selecting each piece
    print("\n--- F: ACTION5 after select ---")
    state = extract_state(eq)
    offset = state['offset']
    for pp in list(state['pieces'].keys()):
        p = px(pp, offset)
        env.step(GameAction.ACTION6, data={'x': p[0], 'y': p[1]})
        sb = state_signature(eq)
        fd = env.step(GameAction.ACTION5)
        sa = state_signature(eq)
        d = diff_sigs(sb, sa)
        if d:
            print(f"  ACTION5@{pp}: diffs={len(d)} state={fd.state.name}")
            for x in d[:8]: print(f"    {x}")

    # EXPT G: click each piece with another piece already selected
    print("\n--- G: select piece1, click piece2 ---")
    state = extract_state(eq)
    offset = state['offset']
    pieces_list = list(state['pieces'].items())
    for i, (p1_pos, p1_n) in enumerate(pieces_list):
        for j, (p2_pos, p2_n) in enumerate(pieces_list):
            if i == j: continue
            # Select p1
            p1 = px(p1_pos, offset)
            env.step(GameAction.ACTION6, data={'x': p1[0], 'y': p1[1]})
            sb = state_signature(eq)
            # Click p2
            p2 = px(p2_pos, offset)
            fd = env.step(GameAction.ACTION6, data={'x': p2[0], 'y': p2[1]})
            sa = state_signature(eq)
            d = diff_sigs(sb, sa)
            if d:
                print(f"  sel {p1_n}@{p1_pos} -> click {p2_n}@{p2_pos}: {len(d)} diffs")
                for x in d[:8]: print(f"    {x}")

    # EXPT H: scroll experimentation (click at various UI positions)
    print("\n--- H: UI pixel clicks ---")
    for pos in [(8,8),(8,56),(56,8),(56,56),(5,5),(0,0),(63,63),(32,32),(48,48),(16,16)]:
        sb = state_signature(eq)
        fd = env.step(GameAction.ACTION6, data={'x': pos[0], 'y': pos[1]})
        sa = state_signature(eq)
        d = diff_sigs(sb, sa)
        if d:
            print(f"  pixel {pos}: {len(d)} diffs state={fd.state.name}")
            for x in d[:6]: print(f"    {x}")

    # EXPT I: push blocks in all 4 directions (may reveal something)
    print("\n--- I: push each direction twice ---")
    for act, nm in [(GameAction.ACTION1,'UP'),(GameAction.ACTION2,'DOWN'),
                    (GameAction.ACTION3,'LEFT'),(GameAction.ACTION4,'RIGHT')]:
        sb = state_signature(eq)
        fd = env.step(act)
        sa = state_signature(eq)
        d = diff_sigs(sb, sa)
        print(f"  {nm}: {len(d)} diffs state={fd.state.name} steps={eq.asqvqzpfdi}")

    # Final state
    state = extract_state(eq)
    print(f"\n=== Final ===")
    print(f"pieces={state['pieces']}")
    print(f"blocks={sorted(state['pushable'])}")
    print(f"pegs={len(state['fixed_pegs'])}")
    print(f"steps={eq.asqvqzpfdi}")
    print(f"state={fd.state.name}")


if __name__ == "__main__":
    main()
