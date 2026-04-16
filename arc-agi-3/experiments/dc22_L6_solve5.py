#!/usr/bin/env python3
"""dc22 L6 - test walk-down-staircase then f-cycle-back strategy."""
import os, sys, json
os.chdir("/home/dp/ai-workspace/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
import numpy as np
from PIL import Image

VIS = "/home/dp/ai-workspace/shared-context/arc-agi-3/visual-memory/dc22"

PALETTE = {
    0:(255,255,255), 1:(220,220,220), 2:(255,0,0), 3:(128,128,128),
    4:(255,255,0), 5:(100,100,100), 6:(255,0,255), 7:(255,192,203),
    8:(200,0,0), 9:(128,0,0), 10:(0,0,255), 11:(135,206,250),
    12:(0,0,200), 13:(255,165,0), 14:(0,255,0), 15:(128,0,128),
}

def save_frame(frame_data, path):
    frame = np.array(frame_data[0])
    h, w = frame.shape
    scale = 8
    img = Image.new('RGB', (w*scale, h*scale))
    pix = img.load()
    for y in range(h):
        for x in range(w):
            c = PALETTE.get(int(frame[y,x]), (0,0,0))
            for dy in range(scale):
                for dx in range(scale):
                    pix[x*scale+dx, y*scale+dy] = c
    img.save(path)

def save_game_state(game):
    safe_sprites = []
    for s in game.current_level.get_sprites():
        c = s.clone()
        c.set_position(s.x, s.y)
        c.set_interaction(s.interaction)
        c._blocking = s._blocking
        c._tags = list(s.tags)
        safe_sprites.append(c)
    return {
        'sprites': safe_sprites,
        'nxhz_x': game.nxhz_x, 'nxhz_y': game.nxhz_y,
        'nxhz_attached_kind': game.nxhz_attached_kind,
        'attached_hhxv_prefix': game.attached_hhxv_prefix,
        'attached_hhxv_x': game.attached_hhxv_x,
        'attached_hhxv_y': game.attached_hhxv_y,
        'step_counter': game.step_counter_ui.rdnpeqedga,
        'current_steps': game.step_counter_ui.current_steps,
        'uuehztercxf': game.uuehztercxf,
        'pxicvzkjuui': game.pxicvzkjuui,
        'prbjhwkkxth': game.prbjhwkkxth,
        'fgxfjbqnmgt': game.fgxfjbqnmgt,
        'zemyudjnnqd': game.zemyudjnnqd,
        'jnmawhhrfhh': game.jnmawhhrfhh,
        'dimvmykkjbg': game.dimvmykkjbg,
    }

def restore_game_state(game, state):
    fresh_sprites = []
    for s in state['sprites']:
        c = s.clone()
        c.set_position(s.x, s.y)
        c.set_interaction(s.interaction)
        c._blocking = s._blocking
        c._tags = list(s.tags)
        fresh_sprites.append(c)
    game.lvnwxszdcv = {
        'sprites': fresh_sprites,
        'nxhz_x': state['nxhz_x'], 'nxhz_y': state['nxhz_y'],
        'nxhz_attached_kind': state['nxhz_attached_kind'],
        'attached_hhxv_prefix': state['attached_hhxv_prefix'],
        'attached_hhxv_x': state['attached_hhxv_x'],
        'attached_hhxv_y': state['attached_hhxv_y'],
    }
    game.ycfbtkckze()
    game.step_counter_ui.rdnpeqedga = state['step_counter']
    game.step_counter_ui.current_steps = state['current_steps']
    game.uuehztercxf = state['uuehztercxf']
    game.pxicvzkjuui = state['pxicvzkjuui']
    game.prbjhwkkxth = state['prbjhwkkxth']
    game.fgxfjbqnmgt = state['fgxfjbqnmgt']
    game.zemyudjnnqd = state['zemyudjnnqd']
    game.jnmawhhrfhh = state['jnmawhhrfhh']
    game.dimvmykkjbg = state['dimvmykkjbg']
    game.nxhz_x = state['nxhz_x']
    game.nxhz_y = state['nxhz_y']
    game.nxhz_attached_kind = state['nxhz_attached_kind']
    game.attached_hhxv_prefix = state['attached_hhxv_prefix']
    game.attached_hhxv_x = state['attached_hhxv_x']
    game.attached_hhxv_y = state['attached_hhxv_y']
    if state['nxhz_attached_kind'] == 'hhxv':
        try: game.dldhnotovw()
        except ValueError: pass
    elif state['nxhz_attached_kind'] == 'bynyvtuepbt-object' and game.ciuxrvkyndj:
        game.euqqhkqayni = game.ciuxrvkyndj

def player_reachable_cells(game):
    player = game.fdvakicpimr
    start = (player.x, player.y)
    orig_x, orig_y = player.x, player.y
    parents = {start: None}
    order = [start]
    head = 0
    deltas = [
        (0, -2, GameAction.ACTION1),
        (0,  2, GameAction.ACTION2),
        (-2, 0, GameAction.ACTION3),
        ( 2, 0, GameAction.ACTION4),
    ]
    while head < len(order):
        cx, cy = order[head]
        head += 1
        for dx, dy, act in deltas:
            player.set_position(cx, cy)
            collisions = game.try_move_sprite(player, dx, dy)
            if collisions:
                continue
            nx, ny = player.x, player.y
            if (nx, ny) == (cx, cy):
                continue
            if game.uxwpppoljm(nx, ny, player) is None:
                continue
            if (nx, ny) in parents:
                continue
            parents[(nx, ny)] = (cx, cy, act)
            order.append((nx, ny))
    player.set_position(orig_x, orig_y)
    return parents

def reconstruct_moves(parents, goal):
    moves = []
    cur = goal
    while parents.get(cur) is not None:
        px, py, act = parents[cur]
        moves.append(act)
        cur = (px, py)
    moves.reverse()
    return moves

def find_click_targets(game):
    targets = {}
    cam_h = game.camera._height
    y_offset = (64 - cam_h) // 2
    for s in game.current_level.get_sprites():
        if 'sys_click' in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2 + y_offset
            targets[s.name] = {'x': cx, 'y': cy}
    return targets

def replay_to_L6(env):
    cache_path = f"{VIS}/solutions.json"
    with open(cache_path) as f:
        raw = json.load(f)
    am = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
          3: GameAction.ACTION3, 4: GameAction.ACTION4,
          6: GameAction.ACTION6}
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            a = am[m['action']]
            env.step(a, data=m.get('data', {}))

RECORDED = []

def do_click(env, data, label=""):
    fd = env.step(GameAction.ACTION6, data=data)
    RECORDED.append((GameAction.ACTION6, data))
    game = env._game
    print(f"  [{label}] click: player=({game.fdvakicpimr.x},{game.fdvakicpimr.y}), "
          f"steps={game.step_counter_ui.current_steps}, state={fd.state.name}")
    if fd.state.name == 'LOSE':
        raise RuntimeError(f"LOSE after {label}")
    return fd

def walk_to(env, target, label=""):
    game = env._game
    parents = player_reachable_cells(game)
    if target not in parents:
        print(f"  [{label}] target {target} NOT reachable!")
        return False
    moves = reconstruct_moves(parents, target)
    for m in moves:
        fd = env.step(m)
        RECORDED.append(m)
        if fd.state.name == 'LOSE':
            print(f"  [{label}] LOSE during walk")
            return False
    game = env._game
    print(f"  [{label}] walked to {target}: player=({game.fdvakicpimr.x},{game.fdvakicpimr.y}), "
          f"steps={game.step_counter_ui.current_steps}, {len(moves)} moves")
    return True

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    replay_to_L6(env)
    game = env._game
    ct = find_click_targets(game)

    print(f"Player: ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    print(f"Steps: {game.step_counter_ui.current_steps}")

    state0 = save_game_state(game)

    # Strategy:
    # 1. f-click x4 to create staircase
    # 2. Walk down staircase to (8,58) area
    # 3. f-click x2 more to reset to f=0 pattern (cycle is 6)
    # 4. Now qgdz at y=54 has variant 1 (x=6..11), connecting kbqq(6,48) to qgdz
    # 5. But are we still supported at (8,58)?
    # 6. The qgdz at y=58 after f=6 total = f=0 = variant 4, nonblank x=12..17
    #    We'd be at (8,58) and variant 4 has nonblank at x=12..17 - NOT x=8!
    #    We'd fall!

    # Let me check all combinations: walk to position P after N f-clicks,
    # then do M more f-clicks, check if still supported

    # After 4 f-clicks: y=58 has qgdz-efzv-2 (nonblank x=8..13)
    # After 5 f-clicks: y=58 has qgdz-efzv-3 (nonblank x=10..15)
    # After 6 f-clicks: y=58 has qgdz-efzv-4 (nonblank x=12..17)

    # So if we walk to (8,58) after 4 f-clicks, then do 2 more f-clicks,
    # y=58 becomes variant 4 (x=12..17) and we're at x=8 - FALL!

    # What about standing on kbqq-efzv-1 at (32,56) which covers y=56..61?
    # After 4 f-clicks we can reach y=56 at x=12..20. Not x=32.

    # Key insight: the qgdz staircase goes LEFT, not RIGHT.
    # We need to go RIGHT (toward x=32) to reach the pressure plates.

    # What if we walk LEFT via the staircase to the bottom-left area,
    # then UP through the bridge to the top-left kbqq cluster,
    # then pick up zbhi-d at (6,18)?

    # For this to work: after 4 f-clicks, walk to (8,58).
    # From (8,58), go UP. Need support at (8,56).
    # qgdz at y=56 after 4 f-clicks: variant 4 (x=12..17). At x=8, no support!

    # After 5 f-clicks: y=56 has variant 5 (x=14..19). Still no x=8.
    # BUT with 5 f-clicks, y=56 also shows x=6,8,10 in the reach!
    # That's because there are TWO qgdz sprites at y=56. After 5 f-clicks:
    # qgdz-efzv-5 at y=56 (one of the pair) has nonblank at x=14..19
    # The second sprite at y=56 is qgdz-efzv-4 (from the initial pair qgdz-efzv-2+6)
    # Wait, let me check the sprite names at y=56 after 5 f-clicks

    print("\n=== DETAILED QGDZ AT Y=56 ===")
    for nf in range(7):
        restore_game_state(game, state0)
        for i in range(nf):
            env.step(GameAction.ACTION6, data=ct['sprite-6'])
        # List all active qgdz at y=56
        active = []
        for s in game.current_level.get_sprites():
            if 'qgdz' in s.name and s.interaction != InteractionMode.REMOVED and s.y == 56:
                pixels = s.render()
                nonblank = [col for col in range(pixels.shape[1]) if any(pixels[row][col] >= 0 for row in range(pixels.shape[0]))]
                active.append(f"{s.name}(x={[c+s.x for c in nonblank]})")
        print(f"  f={nf}: y=56 active: {active}")
        active58 = []
        for s in game.current_level.get_sprites():
            if 'qgdz' in s.name and s.interaction != InteractionMode.REMOVED and s.y == 58:
                pixels = s.render()
                nonblank = [col for col in range(pixels.shape[1]) if any(pixels[row][col] >= 0 for row in range(pixels.shape[0]))]
                active58.append(f"{s.name}(x={[c+s.x for c in nonblank]})")
        print(f"         y=58 active: {active58}")
        active54 = []
        for s in game.current_level.get_sprites():
            if 'qgdz' in s.name and s.interaction != InteractionMode.REMOVED and s.y == 54:
                pixels = s.render()
                nonblank = [col for col in range(pixels.shape[1]) if any(pixels[row][col] >= 0 for row in range(pixels.shape[0]))]
                active54.append(f"{s.name}(x={[c+s.x for c in nonblank]})")
        print(f"         y=54 active: {active54}")

    # REAL EXPERIMENT: after 4 f-clicks, walk to (8,58), then check what
    # f-click combinations let us walk up
    print("\n\n=== WALK DOWN THEN F-CYCLE BACK ===")
    restore_game_state(game, state0)

    # 4 f-clicks
    for i in range(4):
        do_click(env, ct['sprite-6'], f'f{i+1}')

    # Walk to bottom of staircase
    if not walk_to(env, (8, 58), 'staircase-bottom'):
        print("Can't reach (8,58)!")
        return

    save_frame(env.observation_space.frame, f"{VIS}/L6_at_8_58.png")

    # Now try f-clicking to find a configuration where we're still supported
    # AND can walk up
    state_at_bottom = save_game_state(game)

    for extra_f in range(1, 7):
        restore_game_state(game, state_at_bottom)
        # Check if f-click would cause fall
        supp_before = game.uxwpppoljm(8, 58, game.fdvakicpimr)
        print(f"\n  Extra f={extra_f}: support at (8,58) = {supp_before.name if supp_before else 'NONE'}")

        fd = env.step(GameAction.ACTION6, data=ct['sprite-6'])
        print(f"    After f-click: state={fd.state.name}, player=({game.fdvakicpimr.x},{game.fdvakicpimr.y})")

        if fd.state.name == 'LOSE':
            print("    FELL! Trying different positions...")
            continue

        # Check reach
        parents = player_reachable_cells(game)
        # Check key targets
        has_6_18 = (6, 18) in parents
        has_6_48 = (6, 48) in parents
        has_8_44 = (8, 44) in parents
        print(f"    Reach: {len(parents)} cells, (6,18)={has_6_18}, (6,48)={has_6_48}, (8,44)={has_8_44}")

        if len(parents) > 5:
            ys = sorted(set(y for _, y in parents.keys()))
            for y in ys:
                xs = sorted(x for x, yy in parents.keys() if yy == y)
                print(f"      y={y}: x={xs}")

    # Try different walk targets before extra f-clicks
    print("\n\n=== TRY DIFFERENT WALK TARGETS ===")
    # After 4 f-clicks, reachable at y=58: x=8,10,12
    # After 4 f-clicks, reachable at y=56: x=12,14,16,18,20
    # Let's try (12,56) which is on qgdz-efzv-4 (nonblank x=12..17)
    # Then f-click to see if we can stay on that tile

    for walk_target in [(12, 56), (10, 58), (12, 58), (18, 56), (20, 56)]:
        restore_game_state(game, state0)
        for i in range(4):
            env.step(GameAction.ACTION6, data=ct['sprite-6'])
        parents = player_reachable_cells(game)
        if walk_target not in parents:
            continue
        moves = reconstruct_moves(parents, walk_target)
        for m in moves:
            env.step(m)

        for extra_f in range(1, 7):
            state_wt = save_game_state(game)
            fd = env.step(GameAction.ACTION6, data=ct['sprite-6'])
            if fd.state.name != 'LOSE':
                parents2 = player_reachable_cells(game)
                has_kbqq_left = any((x, y) in parents2 for x in range(4, 12) for y in range(44, 54))
                has_plates = any((x, y) in parents2 for x in range(32, 38) for y in range(56, 62))
                total_f = 4 + extra_f
                if has_kbqq_left or has_plates or len(parents2) > 30:
                    print(f"\n  walk_to={walk_target}, +f={extra_f} (total_f={total_f}): "
                          f"reach={len(parents2)}, kbqq_left={has_kbqq_left}, plates={has_plates}")
                    ys = sorted(set(y for _, y in parents2.keys()))
                    for y in ys:
                        xs = sorted(x for x, yy in parents2.keys() if yy == y)
                        print(f"      y={y}: x={xs}")
            restore_game_state(game, state_wt)

    # What if we need c-click (bridge) + f-clicks + walk + more f-clicks?
    print("\n\n=== C-CLICK + F-CYCLE + WALK + F-CYCLE ===")
    # After c-click, bridge (variant 2) is walkable at (6..12, 24..42)
    # Combined with kbqq(4..11, 4..23): left side is fully connected to kbqq(6..11,44..53)
    # Then if f-cycle connects kbqq(6,48) down to qgdz, we could walk all the way up

    # c-click + 4 f-clicks: bridge walkable, staircase connects x=18 down to x=8
    # But does bridge + kbqq(6,48) + qgdz connect?

    restore_game_state(game, state0)
    # c-click for bridge
    env.step(GameAction.ACTION6, data=ct['jpug-bjuk'])
    # 4 f-clicks for staircase
    for i in range(4):
        env.step(GameAction.ACTION6, data=ct['sprite-6'])

    parents = player_reachable_cells(game)
    print(f"c=1 + f=4: reach={len(parents)} cells")
    if len(parents) > 20:
        ys = sorted(set(y for _, y in parents.keys()))
        for y in ys:
            xs = sorted(x for x, yy in parents.keys() if yy == y)
            print(f"  y={y}: x={xs}")

    # Check: does kbqq at (6,48) connect to qgdz at y=54 (variant 5)?
    # kbqq(6,48) covers y=48..53. qgdz at y=54 variant 5 has x=14..19
    # kbqq(6,48) has x=6..11. No overlap at y=54!
    # So they DON'T connect.

    # What about after walking down and then f-cycling?
    # If we start from x=18 at y=48, walk down to (8,58) via staircase,
    # that path is: (18,52)->(18,54)->(14,56)->(12,56)->(8,58)
    # But (18,54) to (14,56) requires going LEFT at y=54 from x=18 to x=14
    # Actually the staircase goes: x=18 at y=52, down to x=18 at y=54 (qgdz-efzv-5 x=14..19),
    # then left to x=14 still at y=54, then down to y=56 at x=14 (qgdz-efzv-4 x=12..17),
    # then left to x=12, then down to y=58 at x=12 (qgdz-efzv-2 x=8..13),
    # then left to x=8

    # Now at (8,58). If we f-click 2 more times (total 6 = back to 0):
    # qgdz at y=54 goes back to variant 1 (x=6..11)
    # qgdz at y=56 goes back to variants 2+6 (x=8..13 and x=16..21)
    # qgdz at y=58 goes back to variant 4 (x=12..17) -- we're at x=8, NO support!

    # But what about standing on the kbqq-efzv-1 at (32,56)?
    # It covers (32..37, 56..61). Can we walk there from the staircase? No, max x is 20.

    # New idea: what if we use TELEPORT to change player position?
    # After walking down to (8,58), we're far from any itki.
    # The itkis are at (18,48), (32,52), (4,4), (34,58).
    # (34,58) is an itki! But is it reachable from (8,58)?
    # After 4 f-clicks, reach at y=58 = x=8,10,12. Not x=34.

    # What about (18,48)? It's at y=48, we're at y=58. Can't reach y=48 from y=58
    # because there's no floor connecting them at these x values.

    # Let me check: after c-click (bridge walkable) + 4 f-clicks + walk to (8,58),
    # what's the full reach?
    restore_game_state(game, state0)
    env.step(GameAction.ACTION6, data=ct['jpug-bjuk'])
    for i in range(4):
        env.step(GameAction.ACTION6, data=ct['sprite-6'])
    parents = player_reachable_cells(game)
    if (8, 58) in parents:
        moves = reconstruct_moves(parents, (8, 58))
        for m in moves:
            env.step(m)
        print(f"\nAt (8,58) after c+4f: steps={game.step_counter_ui.current_steps}")
        parents2 = player_reachable_cells(game)
        print(f"Reach from (8,58): {len(parents2)} cells")
        ys = sorted(set(y for _, y in parents2.keys()))
        for y in ys:
            xs = sorted(x for x, yy in parents2.keys() if yy == y)
            print(f"  y={y}: x={xs}")

    # Final check: what happens if we walk to (10,58) then do 1 f-click?
    # After 4f: y=58 has variant 2 (x=8..13). After 5f: variant 3 (x=10..15)
    # At (10,58), still on variant 3. Then walk right to (14,58)? No, y=58 max x=14..15.
    # Not helpful.

    # OK let me try a COMPLETELY different approach.
    # What if the f-cycle creates a RIGHT-going staircase if we START from a different position?
    # After teleporting to (32,52), the reach is (32..36, 48..52).
    # Can any f-cycle create support at (32,54)?
    # qgdz is at x=6..21 only. x=32 is NOT covered by any qgdz variant.

    # So the RIGHT side gap at (32,54-55) is NEVER covered by qgdz.
    # The only way to bridge it is the hhxv bridge or some other mechanism.

    # But we need the crane to move the hhxv bridge, which needs plates, which needs
    # crossing the gap. TRUE CIRCULAR DEPENDENCY.

    # UNLESS: the hhxv bridge can be moved to cover that gap WITHOUT pressure plates.
    # How? The crane is at (0,0) initially at pixel (22,30).
    # The grab button is REMOVED until zbhi-g is picked up.
    # The direction buttons are INVISIBLE until plates are stepped on.
    # INVISIBLE sprites can't be clicked.

    # WAIT. Let me re-check the click handler. ozbgzuaoya at line 2386:
    # It skips REMOVED and INVISIBLE sprites. So INVISIBLE can't be clicked.
    # But sys_click sprites in line 2751 use ozbgzuaoya too.
    # Actually no - line 2751: sprite = self.ozbgzuaoya(game_x, game_y, "sys_click")
    # This finds visible, non-removed sys_click sprites at the click position.
    # INVISIBLE sprites are skipped.

    # So direction buttons truly can't be used without pressure plates.

    # LAST IDEA: What about the kbqq-efzv-1 at (32,56)?
    # It's 6x6 covering (32..37, 56..61). The kbqq-efzv-1 at (32,48) covers (32..37, 48..53).
    # Gap: y=54-55 at x=32-37.
    # The merged-sprite-1 at (40,-2) is 2x66 TANGIBLE. It blocks at x=40-41.
    # qeqe-bg-1 at (40,0) is 32x66 TANGIBLE. Blocks x=40-71.

    # So the right panel (x>=40) is blocked. The game board is 0-39.

    # Is there ANYTHING at y=54-55 and x=22-37?
    print("\n\n=== ALL SPRITES AT Y=52..57, X=22..37 ===")
    restore_game_state(game, state0)
    for s in game.current_level.get_sprites():
        if s.y + s.height > 52 and s.y < 57 and s.x + s.width > 22 and s.x < 38:
            print(f"  {s.name:35s} ({s.x:3d},{s.y:3d}) {s.width}x{s.height} int={s.interaction.name} tags={s.tags}")

if __name__ == "__main__":
    main()
