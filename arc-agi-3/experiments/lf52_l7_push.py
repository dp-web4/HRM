#!/usr/bin/env python3
"""Try each push direction from L7 pristine, see what valid jumps appear.
Then go 2 pushes deep with BFS, looking for first reducing jump."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
import lf52_solve_final as solver

DIR_NAMES = {(0,-1):'UP',(0,1):'DOWN',(-1,0):'LEFT',(1,0):'RIGHT'}
DIRS = [(0,-1),(0,1),(-1,0),(1,0)]

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        fd = solver.solve_level(env, game, lvl)
        if fd is None: return

    eq = game.ikhhdzfmarl
    state = solver.extract_state(eq)
    ps = solver.make_puzzle_state(state)
    print(f"Pristine L7: mc={ps.movable_count()} jumps={len(ps.get_valid_jumps())}")
    print(f"Blocks: {sorted(ps.blocks)}")
    print(f"Pegs: {sorted(ps.fixed_pegs)}")
    print(f"Pieces: {sorted(ps.pieces)}")

    # 1-push expansion
    print("\n=== 1-push expansion ===")
    for d in DIRS:
        nxt = ps.apply_push(d)
        if nxt is None:
            print(f"  PUSH {DIR_NAMES[d]}: no effect")
            continue
        j = nxt.get_valid_jumps()
        new_blocks = sorted(nxt.blocks)
        block_deltas = set(nxt.blocks) - set(ps.blocks)
        print(f"  PUSH {DIR_NAMES[d]}: blocks_moved={len(block_deltas)} jumps={len(j)}")
        if j:
            for src, dst in j[:5]:
                print(f"    jump {src}->{dst}")

    # 2-push expansion
    print("\n=== 2-push depth search for any jump ===")
    from collections import deque
    visited = {ps.key()}
    queue = deque([(ps, [])])
    found_jumps = []
    while queue and len(visited) < 50000:
        cur, path = queue.popleft()
        if len(path) > 6: continue
        for d in DIRS:
            nxt = cur.apply_push(d)
            if nxt is None or nxt.key() in visited: continue
            visited.add(nxt.key())
            j = nxt.get_valid_jumps()
            if j and not found_jumps:
                print(f"  after {len(path)+1} pushes ({[DIR_NAMES[x] for x in path+[d]]}):")
                for src, dst in j:
                    print(f"    jump {src}->{dst}")
                found_jumps = j
                # Keep exploring a bit more
            queue.append((nxt, path + [d]))
    print(f"  explored {len(visited)} push-only states")

    # Now: full solver on this pristine state with extended budget
    print("\n=== Full unified solve with larger budget ===")
    actions = solver.solve_unified(ps, 2, time_limit=180, max_depth=300)
    if actions:
        print(f"  SOLVED with {len(actions)} actions")
        for a in actions:
            print(f"    {a}")

if __name__ == "__main__":
    main()
