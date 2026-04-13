#!/usr/bin/env python3
"""
dc22 L5 deeper probe: full sprite dump, per-letter tag set, crane motion tests,
and systematic exploration of (clicks + crane + walks) via the macro solver.
"""
import os, sys, json, time
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
from collections import deque

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
            if 'data' in m:
                env.step(a, data=m['data'])
            else:
                env.step(a)

def dump_all_sprites(game, label=''):
    print(f"\n== all sprites ({label}) ==")
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED:
            continue
        if 'wbze' in s.tags or 'kbqq' in s.name or 'jpug' in s.tags or 'bqxa' in s.tags:
            print(f"  {s.name:30s} ({s.x:3d},{s.y:3d}) {s.width}x{s.height:<2d} {s.interaction.name:<10s} tags={s.tags}")

def sprites_by_letter(game, letter):
    return [s.name for s in game.current_level.get_sprites()
            if letter in s.tags and s.interaction != InteractionMode.REMOVED]

def reach_key_cells(parents):
    """Extract player-reach cells on key rows."""
    return sorted(parents.keys())

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    obs = env.reset()
    replay_cached(env, f"{VIS}/solutions.json", 4)
    game = env._game
    click_targets = find_click_targets(game)
    ct_map = {}
    for k, n, cx, cy in click_targets:
        ct_map[n] = {'x': cx, 'y': cy}

    dump_all_sprites(game, 'initial L5')

    # Per-letter sprite groups
    for letter in 'abcdef':
        names = sprites_by_letter(game, letter)
        print(f"letter '{letter}': {len(names)} sprites -> {names}")

    init = save_game_state(game)

    # Test each jpug click once and see what sprites change
    print("\n== Effect of clicking each jpug ==")
    jpugs = [(k,n,cx,cy) for (k,n,cx,cy) in click_targets if k == 'jpug']
    for k, n, cx, cy in jpugs:
        restore_game_state(game, init)
        before = {s.name: (s.x, s.y, s.interaction.name, s.is_visible)
                  for s in game.current_level.get_sprites() if s.interaction != InteractionMode.REMOVED}
        env.step(GameAction.ACTION6, data={'x': cx, 'y': cy})
        after = {s.name: (s.x, s.y, s.interaction.name, s.is_visible)
                 for s in game.current_level.get_sprites() if s.interaction != InteractionMode.REMOVED}
        diffs = []
        for nm in set(before) | set(after):
            if before.get(nm) != after.get(nm):
                diffs.append(f"{nm}: {before.get(nm)} -> {after.get(nm)}")
        print(f"\nclick {n}:")
        for d in diffs[:15]:
            print(f"  {d}")
        parents = player_reachable_cells(game)
        print(f"  reach: {len(parents)} cells")

    # Test a full crane grab/move sequence
    print("\n== Crane exploration: move to (0,3), grab object, move around ==")
    restore_game_state(game, init)
    # up 3 times
    for _ in range(3):
        env.step(GameAction.ACTION6, data=ct_map['nxhz-jffakyxiury'])
    print(f"After up×3: nxhz=({game.nxhz_x},{game.nxhz_y}), attached={game.nxhz_attached_kind}")
    for s in game.current_level.get_sprites():
        if 'bynyvtuepbt-object' in s.tags:
            print(f"  object at ({s.x},{s.y})")
    # grab
    env.step(GameAction.ACTION6, data=ct_map['nxhz-bynyvtuepbt'])
    print(f"After grab: attached={game.nxhz_attached_kind}")
    for s in game.current_level.get_sprites():
        if 'bynyvtuepbt-object' in s.tags:
            print(f"  object at ({s.x},{s.y})")
    # right 3 times
    for _ in range(3):
        env.step(GameAction.ACTION6, data=ct_map['nxhz-ghqmfnmmgrz'])
    print(f"After right×3: nxhz=({game.nxhz_x},{game.nxhz_y})")
    for s in game.current_level.get_sprites():
        if 'bynyvtuepbt-object' in s.tags:
            print(f"  object at ({s.x},{s.y})")
    # down 3 times
    for _ in range(3):
        env.step(GameAction.ACTION6, data=ct_map['nxhz-zmjbupyjfyb'])
    print(f"After down×3: nxhz=({game.nxhz_x},{game.nxhz_y})")
    for s in game.current_level.get_sprites():
        if 'bynyvtuepbt-object' in s.tags:
            print(f"  object at ({s.x},{s.y})")

    # What crane positions allow bridging?
    print("\n== Enumerate all 10 crane positions and object coords ==")
    # 'u' = zmjbupyjfyb (nxhz_y+=1 -> visual up, since nxhz_start[1] - y*4)
    # 'd' = jffakyxiury (y-=1)
    # 'r' = ghqmfnmmgrz (x+=1), 'l' = vbdduyutyiw (x-=1)
    # Valid path: up at x=0 only; horiz at y=0 or y=3 only.
    crane_paths = [
        ('start', []),
        ('u1',['u']),('u2',['u','u']),('u3',['u','u','u']),
        ('u3r1',['u','u','u','r']),('u3r2',['u','u','u','r','r']),('u3r3',['u','u','u','r','r','r']),
        # back down
        ('u3r3d1',['u','u','u','r','r','r','d']),
        ('u3r3d2',['u','u','u','r','r','r','d','d']),
        ('u3r3d3',['u','u','u','r','r','r','d','d','d']),
        # only right along bottom
        ('r1',['r']),('r2',['r','r']),('r3',['r','r','r']),
    ]
    cmd_map = {'u': ct_map['nxhz-zmjbupyjfyb'],
               'd': ct_map['nxhz-jffakyxiury'],
               'l': ct_map['nxhz-vbdduyutyiw'],
               'r': ct_map['nxhz-ghqmfnmmgrz'],
               'g': ct_map['nxhz-bynyvtuepbt']}
    # First no-grab: where do we end up
    print("  [no grab]")
    for label, seq in crane_paths:
        restore_game_state(game, init)
        for c in seq:
            env.step(GameAction.ACTION6, data=cmd_map[c])
        print(f"    {label:10s}: crane=({game.nxhz_x},{game.nxhz_y})")

    # With grab AT (3,3): move first, then grab, then move object around
    print("  [move to (3,3), grab, move]")
    for label, seq in crane_paths:
        restore_game_state(game, init)
        # First u×3 r×3 to get to (3,3)
        for c in ['u','u','u','r','r','r']:
            env.step(GameAction.ACTION6, data=cmd_map[c])
        # Then grab
        env.step(GameAction.ACTION6, data=cmd_map['g'])
        attached_after_grab = game.nxhz_attached_kind
        for c in seq:
            env.step(GameAction.ACTION6, data=cmd_map[c])
        obj_pos = None
        for s in game.current_level.get_sprites():
            if 'bynyvtuepbt-object' in s.tags:
                obj_pos = (s.x, s.y)
                break
        print(f"    {label:10s}: crane=({game.nxhz_x},{game.nxhz_y}), attached_after_grab={attached_after_grab}, obj={obj_pos}")

    print("  [grab at (3,3), then navigate U-shape to each of the 10 positions]")
    # All 10 valid crane positions starting from (3,3) after grab:
    # (3,3) -> left (2,3) -> left (1,3) -> left (0,3) -> down (0,2) -> down (0,1) -> down (0,0) -> right (1,0) -> right (2,0) -> right (3,0)
    nav = [
        ('at33',[]),
        ('L1',['l']),('L2',['l','l']),('L3',['l','l','l']),
        ('L3D1',['l','l','l','d']),('L3D2',['l','l','l','d','d']),('L3D3',['l','l','l','d','d','d']),
        ('L3D3R1',['l','l','l','d','d','d','r']),
        ('L3D3R2',['l','l','l','d','d','d','r','r']),
        ('L3D3R3',['l','l','l','d','d','d','r','r','r']),
    ]
    for label, seq in nav:
        restore_game_state(game, init)
        for c in ['u','u','u','r','r','r']:
            env.step(GameAction.ACTION6, data=cmd_map[c])
        env.step(GameAction.ACTION6, data=cmd_map['g'])
        for c in seq:
            env.step(GameAction.ACTION6, data=cmd_map[c])
        obj_pos = None
        for s in game.current_level.get_sprites():
            if 'bynyvtuepbt-object' in s.tags:
                obj_pos = (s.x, s.y)
                break
        print(f"    {label:10s}: crane=({game.nxhz_x},{game.nxhz_y}), obj={obj_pos}")

    print("\n== d toggle check ==")
    restore_game_state(game, init)
    print(f"  initial d state:")
    for s in game.current_level.get_sprites():
        if s.name in ('wbze-efzv_vucz_1','wbze-efzv_vucz_2','wbze-efzv-p-1','wbze-efzv-p-2') and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) {s.interaction.name}")
    env.step(GameAction.ACTION6, data=ct_map['jpug-adnw'])
    print(f"  after d click:")
    for s in game.current_level.get_sprites():
        if s.name in ('wbze-efzv_vucz_1','wbze-efzv_vucz_2','wbze-efzv-p-1','wbze-efzv-p-2') and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) {s.interaction.name}")
    # Test actual walkability: plant player at (26,46) and see support
    game.fdvakicpimr.set_position(26, 46)
    supp = game.uxwpppoljm(26, 46, game.fdvakicpimr)
    print(f"  support at (26,46): {supp.name if supp else None}")
    supp = game.uxwpppoljm(14, 50, game.fdvakicpimr)
    print(f"  support at (14,50): {supp.name if supp else None}")

    print("\n== d toggle with zbhi activation ==")
    restore_game_state(game, init)
    # Activate jpug-adnw by triggering zbhi at (10,18)
    # The code at line 2913-2921: after a walk, if player at position with zbhi,
    # finds jpug with matching letter and sets INTANGIBLE. But just calling here:
    for s in game.current_level.get_sprites():
        if s.name == 'jpug-adnw':
            s.set_interaction(InteractionMode.INTANGIBLE)
            print(f"  Forced jpug-adnw -> INTANGIBLE: {s.name} ({s.x},{s.y}) {s.interaction.name}")
    # Remove zbhi (like walk-trigger does)
    for s in list(game.current_level.get_sprites()):
        if s.name == 'zbhi-jpug-adnw':
            game.current_level.remove_sprite(s)
    print("  Now click d:")
    env.step(GameAction.ACTION6, data=ct_map['jpug-adnw'])
    for s in game.current_level.get_sprites():
        if s.name.startswith('wbze-efzv_vucz') or s.name.startswith('wbze-efzv-p'):
            if s.interaction != InteractionMode.REMOVED:
                print(f"    {s.name} ({s.x},{s.y}) {s.interaction.name}")
    game.fdvakicpimr.set_position(26, 46)
    s1 = game.uxwpppoljm(26, 46, game.fdvakicpimr)
    s2 = game.uxwpppoljm(14, 50, game.fdvakicpimr)
    s3 = game.uxwpppoljm(14, 52, game.fdvakicpimr)
    s4 = game.uxwpppoljm(22, 50, game.fdvakicpimr)
    print(f"  support (26,46)={s1.name if s1 else None}  (14,50)={s2.name if s2 else None}  (14,52)={s3.name if s3 else None}  (22,50)={s4.name if s4 else None}")

    print("\n== b toggle cyclability ==")
    restore_game_state(game, init)
    print(f"  initial wbze state:")
    for s in game.current_level.get_sprites():
        if s.name in ('wbze-efzv1','wbze-efzv2') and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) {s.interaction.name}")
    env.step(GameAction.ACTION6, data=ct_map['jpug-jbyz'])
    print(f"  after b click:")
    for s in game.current_level.get_sprites():
        if s.name in ('wbze-efzv1','wbze-efzv2') and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) {s.interaction.name}")
    env.step(GameAction.ACTION6, data=ct_map['jpug-jbyz'])
    print(f"  after b click x2:")
    for s in game.current_level.get_sprites():
        if s.name in ('wbze-efzv1','wbze-efzv2') and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) {s.interaction.name}")
    for label, seq in crane_paths:
        restore_game_state(game, init)
        env.step(GameAction.ACTION6, data=cmd_map['g'])
        # Was grab successful? (should be if crane at (0,0) with object center nearby)
        attached_after_grab = game.nxhz_attached_kind
        for c in seq:
            env.step(GameAction.ACTION6, data=cmd_map[c])
        obj_pos = None
        for s in game.current_level.get_sprites():
            if 'bynyvtuepbt-object' in s.tags:
                obj_pos = (s.x, s.y)
                break
        print(f"    {label:10s}: crane=({game.nxhz_x},{game.nxhz_y}), attached_after_grab={attached_after_grab}, obj={obj_pos}")

if __name__ == "__main__":
    main()
