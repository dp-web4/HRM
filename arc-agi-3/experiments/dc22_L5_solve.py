#!/usr/bin/env python3
"""
dc22 L5 manual solver. Execute a handcrafted plan and verify.

Plan:
  1. b-click (opens wbze1 at 26,34)
  2. Walk to (26,34) via kbqq(30,34)+wbze(26,34)+kbqq(26,30)
     — actually walk-to-stable: (26,30) kbqq
  3. Crane sequence to grab and move object to (20, 22..33) — crane (3,0)
     - from (0,0): u×3 (to 0,3), r×3 (to 3,3), grab, l×3 (to 0,3), d×3 (to 0,0)
       wait — that's obj (8, 22..33). We need (20, 22..33). So: u×3, r×3, grab, keeps obj at (20, 10..21).
       Then we need to move to (3,0). From (3,3): l×3, d×3 = (0,3)→(0,0)→r×3=(3,0). Total 9 clicks after grab.
     Alternative: direct to (3,0) after grab: from (3,3), l×3 to (0,3), d×3 to (0,0), r×3 to (3,0). 9 clicks.
     Actually we want TWO crane configurations across the solve, so let me plan both:

Sequence A: get player to kbqq(26,20) — crane (3,0) with obj (20, 22..33)
Sequence B: move crane to (3,3) obj (20, 10..21) for player to walk to (22,6) kbqq
Sequence C: move crane to (0,0) obj (8, 22..33) for player at (10,34) to reach zbhi(10,18)

Walking sequence:
  start(32,34) → kbqq(26,30) [via b-bridge]
  → crane (3,0): player walks onto obj bridge at (22..25, 22..33) via (25,30..33)
  → then onto kbqq(26,20) at (26..29, 20..23) via (25, 22..23)↔(26, 22..23)
  → park on kbqq(26,20)
  → crane moves to (3,3) (requires l×3, u×3, r×3 — 9 clicks). Actually from (3,0) we can go l×3 then u×3 then r×3. 9 clicks.
  → obj now at (20, 10..21). Player still on kbqq(26,20).
  → walk player from (26,20..23) onto bridge (25, 20..21)↔(22..25, 10..21) → (22..25, 10..21) → up into (22..25, 9)↔(22..25, 6..9) kbqq(22,6)
  → walk to itki1 (24,6)
  → click jpug-bjuk (c) → teleport to itki2 (10,34)
  → crane moves to (0,0): l×3, d×3. From (3,3): l×3 to (0,3), d×3 to (0,0). 6 clicks.
  → obj at (8, 22..33). Player on kbqq(10,34), reach now includes (10..13, 18..45) via bridge.
  → walk to (10, 18) zbhi → triggers jpug-adnw
  → click jpug-adnw (d) → vucz_1 unwalk, p-2 walkable
  → click vckz-jpug 4x (e) → vckz-5 walkable at (14..29, 44..45)
  → click sprite-6 4x (f) → qgdz-5 walkable at (22..27, 54..55)
  → walk from (10,18) to (10..13, 42..45) → (14..29, 44..45) → (26..29, 45)↔(26..29, 46) p-2 → (26..29, 53)↔(26..27, 54..55) qgdz-5 — need to be at x∈{26,27} for qgdz-5 (which covers 22..27)
  → now click jpug-adnw (d) AGAIN → vucz_2 walk, p-1 block. Player on qgdz-5 at (26,54) or (27,54) — unaffected.
  → walk from (26..27, 54..55) left along qgdz-5 to (22..23, 54..55)
  → walk up to (22..23, 52..53) vucz_2 → left through vucz_2 to (14..15, 50..53)
  → walk left to (12..13, 50..53) kbqq goal → (10..11, 50..51) = goal

NOTE: the click sequence may kill the player if during any click the player is unsupported.
We must ensure the player is ALWAYS supported when clicking.
"""

import os, sys, json, time
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
    am = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
          3: GameAction.ACTION3, 4: GameAction.ACTION4,
          6: GameAction.ACTION6}
    for lvl_idx in range(up_to_level):
        for m in raw[lvl_idx]:
            a = am[m['action']]
            env.step(a, data=m.get('data', {}))

def find_ct(game):
    ct = {}
    for k, n, cx, cy in find_click_targets(game):
        ct[n] = {'x': cx, 'y': cy}
    return ct

RECORDED = []

def click(env, ct, name, label):
    fd = env.step(GameAction.ACTION6, data=ct[name])
    RECORDED.append((GameAction.ACTION6, ct[name]))
    game = env._game
    px, py = game.fdvakicpimr.x, game.fdvakicpimr.y
    print(f"    [{label}] click {name}: state={fd.state.name}, player=({px},{py}), nxhz=({game.nxhz_x},{game.nxhz_y}), attached={game.nxhz_attached_kind}")
    if fd.state.name == 'LOSE':
        raise RuntimeError(f"LOSE after click {name}")
    return fd

def walk_to(env, target, label):
    """Walk player from current pos to target using reachability."""
    game = env._game
    parents = player_reachable_cells(game)
    if target not in parents:
        print(f"    [{label}] target {target} NOT reachable. Current reach has {len(parents)} cells.")
        ys = sorted(set(y for _,y in parents.keys()))
        for y in ys:
            xs = sorted(x for x,yy in parents.keys() if yy==y)
            print(f"      y={y:3d}: x={xs}")
        return False
    moves = reconstruct_moves(parents, target)
    for m in moves:
        fd = env.step(m)
        RECORDED.append(m)
        if fd.state.name == 'LOSE':
            print(f"    [{label}] LOSE during walk to {target}")
            return False
    game = env._game
    print(f"    [{label}] walked to {target}, player=({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    return True

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    replay_cached(env, f"{VIS}/solutions.json", 4)
    game = env._game
    ct = find_ct(game)
    print("Click targets:", list(ct.keys()))
    save_frame(env.observation_space.frame, f"{VIS}/L5_begin.png")

    # ---- Step 1: b-click ----
    print("\n=== Step 1: b-click ===")
    click(env, ct, 'jpug-jbyz', 'b')

    # ---- Step 2: walk to kbqq(26,30) ====
    print("\n=== Step 2: walk to (26,30) kbqq ===")
    if not walk_to(env, (26, 30), 'kbqq(26,30)'):
        return

    # ---- Step 3: crane sequence to (3,0) with grab ----
    # Path: u3 r3 grab l3 u3 r3 (wait from 3,3 → L to 0,3 → d3 → 0,0 → r3 → 3,0. 9 clicks)
    # Actually easier: u3 r3 grab gets us to (3,3) with obj attached. Then to (3,0) we need l3 d3 r3 = 9 clicks.
    # Total crane clicks: 6 + 1 + 9 = 16
    print("\n=== Step 3: crane to (3,0) with obj ===")
    for i in range(3): click(env, ct, 'nxhz-zmjbupyjfyb', f'u{i+1}')   # u (y+=1)
    for i in range(3): click(env, ct, 'nxhz-ghqmfnmmgrz', f'r{i+1}')   # r
    click(env, ct, 'nxhz-bynyvtuepbt', 'grab')
    # At (3,3). Go l3 → (0,3), then d3 → (0,0), then r3 → (3,0)
    for i in range(3): click(env, ct, 'nxhz-vbdduyutyiw', f'l{i+1}')
    for i in range(3): click(env, ct, 'nxhz-jffakyxiury', f'd{i+1}')   # jffakyxiury (y-=1) — visual down
    for i in range(3): click(env, ct, 'nxhz-ghqmfnmmgrz', f'r{i+1}_b')
    # Now at (3,0), obj at (20, 22..33)
    for s in game.current_level.get_sprites():
        if 'bynyvtuepbt-object' in s.tags:
            print(f"  obj now at ({s.x},{s.y})")
            break

    # ---- Step 4: walk to kbqq(26,20) via bridge ----
    print("\n=== Step 4: walk onto bridge, up to kbqq(26,20) ===")
    if not walk_to(env, (26, 20), 'kbqq(26,20)'):
        return

    # ---- Step 5: crane to (3,3) obj (20, 10..21), player stays on kbqq(26,20) ----
    print("\n=== Step 5: crane to (3,3) ===")
    # From (3,0) to (3,3): l3 u3 r3 = 9 clicks.
    for i in range(3): click(env, ct, 'nxhz-vbdduyutyiw', f'l{i+1}')
    for i in range(3): click(env, ct, 'nxhz-zmjbupyjfyb', f'u{i+1}')
    for i in range(3): click(env, ct, 'nxhz-ghqmfnmmgrz', f'r{i+1}')
    for s in game.current_level.get_sprites():
        if 'bynyvtuepbt-object' in s.tags:
            print(f"  obj now at ({s.x},{s.y})")
            break

    # ---- Step 6: walk to itki1 (24,6) via new bridge ----
    print("\n=== Step 6: walk to itki1 (24,6) ===")
    if not walk_to(env, (24, 6), 'itki1'):
        return

    # ---- Step 7: click jpug-bjuk (c) to teleport ----
    print("\n=== Step 7: teleport to itki2 (10,34) ===")
    click(env, ct, 'jpug-bjuk', 'c')
    print(f"  player now at ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")

    # ---- Step 8: crane to (0,0) obj (8, 22..33) ----
    print("\n=== Step 8: crane to (0,0) ===")
    # From (3,3) to (0,0): l3 d3
    for i in range(3): click(env, ct, 'nxhz-vbdduyutyiw', f'l{i+1}')
    for i in range(3): click(env, ct, 'nxhz-jffakyxiury', f'd{i+1}')
    for s in game.current_level.get_sprites():
        if 'bynyvtuepbt-object' in s.tags:
            print(f"  obj now at ({s.x},{s.y})")
            break

    # ---- Step 8.5: b-click again to unblock (10,38) ----
    print("\n=== Step 8.5: b-click again (restore wbze2 at 10,38) ===")
    click(env, ct, 'jpug-jbyz', 'b2')
    # Verify
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED: continue
        if s.name in ('wbze-efzv1','wbze-efzv2'):
            print(f"    {s.name} ({s.x},{s.y}) {s.interaction.name}")

    # ---- Step 9: walk to zbhi (10,18) ----
    print("\n=== Step 9: walk to zbhi (10,18) ===")
    if not walk_to(env, (10, 18), 'zbhi'):
        return
    # Check jpug-adnw state
    for s in game.current_level.get_sprites():
        if s.name == 'jpug-adnw':
            print(f"  jpug-adnw: interaction={s.interaction.name}, visible={s.is_visible}")

    # ---- Step 10: d-click ----
    print("\n=== Step 10: d-click ===")
    click(env, ct, 'jpug-adnw', 'd')
    # Verify vucz/p state
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED: continue
        if s.name.startswith('wbze-efzv_vucz') or s.name.startswith('wbze-efzv-p'):
            print(f"    {s.name} ({s.x},{s.y}) {s.interaction.name}")

    # ---- Step 11: e×4 for vckz-5 ----
    print("\n=== Step 11: vckz-jpug ×4 (e) ===")
    for i in range(4):
        click(env, ct, 'vckz-jpug', f'e{i+1}')
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED: continue
        if s.name.startswith('vckz-') and s.name != 'vckz-jpug':
            print(f"    {s.name} ({s.x},{s.y}) {s.interaction.name}")

    # ---- Step 12: f×4 for qgdz-5 ----
    print("\n=== Step 12: sprite-6 ×4 (f) ===")
    for i in range(4):
        click(env, ct, 'sprite-6', f'f{i+1}')
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED: continue
        if s.name.startswith('qgdz-efzv-'):
            print(f"    {s.name} ({s.x},{s.y}) {s.interaction.name}")

    # Dump current reach
    parents = player_reachable_cells(game)
    print(f"\n  Current reach: {len(parents)} cells")
    ys = sorted(set(y for _,y in parents.keys()))
    for y in ys[-20:]:
        xs = sorted(x for x,yy in parents.keys() if yy==y)
        print(f"      y={y:3d}: x={xs}")

    # ---- Step 13: walk to qgdz-5 at (26, 54) ----
    print("\n=== Step 13: walk to qgdz-5 (26,54) via p-2 ===")
    if not walk_to(env, (26, 54), 'qgdz-5'):
        return

    # ---- Step 14: d-click again ----
    print("\n=== Step 14: d-click again (revert) ===")
    click(env, ct, 'jpug-adnw', 'd2')
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED: continue
        if s.name.startswith('wbze-efzv_vucz') or s.name.startswith('wbze-efzv-p'):
            print(f"    {s.name} ({s.x},{s.y}) {s.interaction.name}")

    # Dump reach again
    parents = player_reachable_cells(game)
    print(f"  Reach after d2: {len(parents)} cells")
    if (10, 50) in parents:
        print(f"  GOAL (10,50) is reachable!")
    else:
        print(f"  GOAL not yet reachable.")
        ys = sorted(set(y for _,y in parents.keys()))
        for y in ys:
            xs = sorted(x for x,yy in parents.keys() if yy==y)
            print(f"      y={y:3d}: x={xs}")

    # ---- Step 15: walk to goal ----
    print("\n=== Step 15: walk to goal (10, 50) ===")
    if not walk_to(env, (10, 50), 'goal'):
        return
    fd = env.observation_space
    print(f"  Final state: levels_completed={fd.levels_completed}, status={fd.state.name}")
    save_frame(fd.frame, f"{VIS}/L5_solved.png")

    # Save solution
    print(f"\nTotal recorded actions: {len(RECORDED)}")
    encoded = []
    for m in RECORDED:
        if isinstance(m, tuple):
            encoded.append({'action': m[0].value, 'data': m[1]})
        else:
            encoded.append({'action': m.value})
    # Load existing cache
    with open(f"{VIS}/solutions.json") as f:
        cache = json.load(f)
    if len(cache) >= 5:
        cache[4] = encoded
    else:
        cache.append(encoded)
    with open(f"{VIS}/solutions.json", 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"Saved L5 solution to {VIS}/solutions.json ({len(encoded)} moves)")

if __name__ == "__main__":
    main()
