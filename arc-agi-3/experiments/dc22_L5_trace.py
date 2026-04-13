#!/usr/bin/env python3
"""
dc22 L5 trace: attempt the conjectured solution path.

Plan:
1. b-click: open wbze1 at (26,34), keeps wbze2 at (10,38) walkable (both INT)
2. Walk from (32,34) to (26,20) — need a vertical path up through (26, 24..33)
   kbqq pads: (26,30), (26,20). These are 4x4 at those positions. Gap (26..29, 24..29)
   is unfilled — need crane bridge or something.
3. Walk to itki1 (24,6), teleport to itki2 (10,34)
4. Walk down (10,38) wbze2 → (10,42) kbqq
5. Click vckz-jpug×4 to cycle vckz to vckz-5 (solid)
   From (10..13, 42..45), walk right to (14..29, 44..45) vckz-5
6. Step on zbhi (10,18) — but player is at y=42+, can't reach
   Actually zbhi is at (10,18). Need to go UP. After teleport player at (10,34).
   Walk up to (10,18): (10,34→30) via wbze2 etc. Hmm, column x=10 at y=20..33 — is there floor?
   kbqq (10,18) 4x4 → (10..13, 18..21). kbqq (10,34) → (10..13, 34..37). Gap (10..13, 22..33).
   Crane bridge at (8,22) covers (10..13, 22..33). So after placing bridge, x=10..13 column connected.
7. So walk up from (10,34) to zbhi (10,18), activates jpug-adnw
8. Return to... need to click 'd' AFTER zbhi activation. Click jpug-adnw.
9. d-click: vucz-2 → vucz-1 (unwalkable), p-1 → p-2 (walkable)
10. Walk to p-2 (26..29, 46..53) — need to get there from reach. From (26, 20)? or via vckz-5?
    vckz (14..29, 44..45) connects to (26..29, 44..45) which is edge of p. (26..29, 46) p-2. Walkable.
11. Walk down p-2 to (26..29, 54..55) qgdz.
12. Walk left along qgdz to (14..23, 54..55).
13. Click d AGAIN: vucz-1 → vucz-2 (walkable), p-2 → p-1 (blocking).
    But wait — cycling in wqsthrbpfk takes the current named sprite and finds next number.
    vucz_1 → vucz_2, p_2 → p_1. After second d-click, we're back to initial d state.
    But zbhi has been consumed... does jpug-adnw still work? Yes, it's INTANGIBLE.
14. Walk up qgdz → vucz-2 (now walkable) → kbqq (10,50) goal.

Let me try to execute this step by step and see.
"""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import (
    save_game_state, restore_game_state, player_reachable_cells,
    reconstruct_moves, find_click_targets, save_frame,
)

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
            env.step(a, data=m.get('data', {}))

def dump_reach(game, label):
    p = player_reachable_cells(game)
    print(f"  [{label}] player=({game.fdvakicpimr.x},{game.fdvakicpimr.y}) reach={len(p)} cells")
    # Print cells summarized
    if len(p) < 50:
        print(f"    cells: {sorted(p.keys())}")
    else:
        ys = sorted(set(y for _,y in p.keys()))
        for y in ys:
            xs = sorted(x for x,yy in p.keys() if yy==y)
            print(f"    y={y:3d}: x={xs}")

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    replay_cached(env, f"{VIS}/solutions.json", 4)
    game = env._game
    ct = {n: {'x': cx, 'y': cy} for k, n, cx, cy in find_click_targets(game)}
    print("click targets:", list(ct.keys()))

    dump_reach(game, 'init')

    # Step 1: b-click
    env.step(GameAction.ACTION6, data=ct['jpug-jbyz'])
    dump_reach(game, 'after b')

    # Step 2: walk to (26,34)
    parents = player_reachable_cells(game)
    if (26, 34) in parents:
        moves = reconstruct_moves(parents, (26, 34))
        for m in moves:
            env.step(m)
        dump_reach(game, 'at (26,34)')
    else:
        print("  (26,34) NOT reachable")
        return

    # Step 3: crane sequence - need to grab object at (3,3)
    # Path: up×3, right×3, grab. But first, we need the object to actually be grabbable.
    # 'u' tag=zmjbupyjfyb, 'r'=ghqmfnmmgrz
    for _ in range(3):
        env.step(GameAction.ACTION6, data=ct['nxhz-zmjbupyjfyb'])
    for _ in range(3):
        env.step(GameAction.ACTION6, data=ct['nxhz-ghqmfnmmgrz'])
    env.step(GameAction.ACTION6, data=ct['nxhz-bynyvtuepbt'])  # grab
    print(f"  after grab: attached={game.nxhz_attached_kind}, nxhz=({game.nxhz_x},{game.nxhz_y})")

    # Move object to (0,0) via L3 D3 — left 3, down 3
    for _ in range(3):
        env.step(GameAction.ACTION6, data=ct['nxhz-vbdduyutyiw'])  # l
    for _ in range(3):
        env.step(GameAction.ACTION6, data=ct['nxhz-jffakyxiury'])  # d (visual down)
    for s in game.current_level.get_sprites():
        if 'bynyvtuepbt-object' in s.tags:
            print(f"  object at ({s.x},{s.y}), nxhz=({game.nxhz_x},{game.nxhz_y})")
            break
    dump_reach(game, 'after crane to (0,0)')

    # Step 4: walk to itki2 (10,34)? Actually we want to walk along the object bridge
    # From (26,34), can we walk to bridge? bridge is at (8..13, 22..33). Player needs
    # to reach (13, 33) via... need (26,34) connected to (13,32) somehow.
    # Hmm, (26,34) kbqq pad connects via (28..29, 34..37) to (30..33,34..37).
    # To get to x=13 we need a horizontal bridge at y=34 or somewhere. Nothing there.
    # BUT: via the crane-grabbed object spanning (10..13, 22..33), player can reach x=10..13 column
    # if they first get to the bridge. The bridge is "floating" at y=22..33 though — no
    # connection to y=34 row because y=34 is kbqq at (30,34) and (26,34) and (10,34),
    # isolated from x=10..13 at y=22..33 because (10..13, 22..33) bridge; (26..29, 22..33) void.
    # So bridge is connected only at its own footprint. To reach it, need another route.

    # ALTERNATIVE: walk UP via (26,20) kbqq pad — need bridge from (26,34) to (26,20).
    # (26, 24..29) gap. kbqq(26,30) fills (26..29, 30..33). Gap (26..29, 24..29). Need bridge.
    # With crane at (3,0), object at (20, 22..33). That fills (26..29... no, x=20..23+solid.
    # Hmm object is 6 wide solid at cols 2..5 so at (20,22) solid at (22..25, 22..33).
    # That's wrong x range.
    # At crane (2,0) → obj (16,22) → solid (18..21, 22..33). Not reaching x=26.
    # At crane (3,0) → obj (20,22) → solid (22..25, 22..33). (22..25, 22..33) + kbqq(26,30) at (26..29,30..33)
    # = connects to (26..29, 30..33) only via (25, 30) to (26, 30)? Yes! (25, 30) on obj bridge, (26, 30) on kbqq. Edge-adjacent. Walkable.
    # Then kbqq(26,30) to kbqq(26,20)? (26, 20..29) — kbqq(26,20) fills (26..29, 20..23), kbqq(26,30) fills (26..29, 30..33). Gap (26..29, 24..29) — 6 rows unfilled. Need another bridge.
    # Back to the bridge: obj at (20,22) covers (22..25, 22..33). (22..25, 22..29) fills gap partially but not at x=26. Player at (25, 29) walks to (26, 29)? Nothing at (26, 29). Void.
    # Hmm.

    # Try a different crane position. Let me think algorithmically — I need the macro BFS
    # to search over crane positions automatically. The macro already does this but may
    # have a bug in how it walks to bridge cells.
    print("\n=== Direct execution stopped — need smarter BFS ===")

if __name__ == "__main__":
    main()
