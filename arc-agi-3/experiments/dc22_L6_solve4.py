#!/usr/bin/env python3
"""dc22 L6 - trace teleport mechanics to find path to (4,4)."""
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

def dump_itkis(game, label=""):
    print(f"\n  itkis ({label}):")
    for s in game.current_level.get_sprites():
        if 'itki' in s.tags and s.interaction != InteractionMode.REMOVED:
            on_player = (s.x == game.fdvakicpimr.x and s.y == game.fdvakicpimr.y)
            color_cycle = 'itki-color-cycle' in s.tags
            print(f"    {s.name:15s} ({s.x:2d},{s.y:2d}) cc={color_cycle} {'<-- PLAYER' if on_player else ''}")

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    replay_to_L6(env)
    game = env._game
    ct = find_click_targets(game)

    print(f"Player: ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    print(f"Click targets: {list(ct.keys())}")
    dump_itkis(game, "INITIAL")

    state0 = save_game_state(game)

    # The key: gkrr-jpug has itki-color-jpug tag AND 'd' tag.
    # It's REMOVED initially. To activate it: walk over zbhi-jpug-adnw-1 (tag 'd') at (6,18).
    # Then gkrr-jpug becomes INTANGIBLE (clickable).
    # Clicking gkrr-jpug calls fnhzudfjhd which cycles itki-color-cycle sprites.

    # The itki at (18,48) has itki-color-cycle. Its name follows bscfvnwogt:
    # ["itkiupry", "itkibgeg", "itkizfrq", "itkijbyz"]
    # Initially: itkiupry1. After 1 cycle: itkibgeg1.
    # After 2 cycles: itkizfrq1. After 3 cycles: itkijbyz1.

    # The teleport (npqswumrzz) from itkibgeg1 looks for itkibgeg2.
    # But there's no itkibgeg2 initially. What IS there?
    # Initially: itkiupry1(18,48), itkiupry2(32,52), itkizfrq2(4,4), itkijbyz2(34,58)
    # After c-click (variant swap): itkiupry2(18,48), itkiupry1(32,52), itkizfrq1(4,4), itkijbyz1(34,58)

    # So: if we cycle the color of itkiupry1(18,48) to itkizfrq1, then
    # when we click c while on it, the variant swap makes all itkis cycle:
    # itkizfrq1(18,48) -> itkizfrq2 exists at (4,4)!
    # But the c-click does variant swap first THEN looks for teleport target.

    # Actually, let me re-read the code more carefully.
    # Line 2724-2742: wqsthrbpfk(letter, clicked_sprite) returns all sprites
    # with the letter tag that are visible and not REMOVED. Then:
    # - If itki-color-jpug in clicked_sprite: calls fnhzudfjhd(azifrpswxp)
    #   which cycles color-cycle sprites. Returns remaining sprites.
    # - Else: finds itki the player is on, then calls npqswumrzz(itki, sprites)
    #   to find teleport target. Also cycles variants via uqbvwhliqb.

    # For the regular c-click (jpug-bjuk): NOT itki-color-jpug. So:
    # 1. Find all sprites with tag 'c' that are visible, not REMOVED, not the clicked one
    # 2. Check if player is on an itki in that list
    # 3. If yes: npqswumrzz finds the NEXT variant. uqbvwhliqb("itkiupry1") = "itkiupry2"
    #    So it looks for itkiupry2 in the list. If found, teleports there.
    # 4. Then cycle variants for ALL sprites in the list

    # For gkrr-jpug click (itki-color-jpug = True):
    # 1. Find all sprites with tag 'd' that are visible, not REMOVED
    # 2. fnhzudfjhd cycles their NAMES (itkiupry1 -> itkibgeg1)
    # 3. No teleport happens!

    # So the color cycle (via gkrr-jpug) changes the NAME of the itki at (18,48).
    # After that, when we click c (jpug-bjuk), the system looks for the NEXT
    # variant of the current name. If itkibgeg1 is at (18,48), it looks for
    # itkibgeg2. But itkibgeg2 doesn't exist. Second fallback: looks for
    # same name. So it would find... nothing, and teleport fails.

    # UNLESS: clicking c also cycles all c-group sprites. The itkis at
    # (4,4) and (34,58) cycle too (from itkizfrq2->itkizfrq1, etc).
    # So after one c-click: itkizfrq2(4,4) -> itkizfrq1(4,4).
    # After two c-clicks: itkizfrq1(4,4) -> itkijbyz2(4,4).

    # Let me check: if we do gkrr-jpug click TWICE to change itkiupry1(18,48)
    # to itkizfrq1(18,48), then when we click c (variant swap):
    # - itkizfrq1(18,48) cycles to itkizfrq2(18,48) (all c-group swap)
    # - itkizfrq2(4,4) cycles to itkizfrq1(4,4)
    # - itkiupry2(32,52) cycles to itkiupry1(32,52)
    # - itkijbyz2(34,58) cycles to itkijbyz1(34,58)
    # Player on itkizfrq2(18,48), teleport target: itkizfrq1(4,4). SUCCESS!

    # Wait but the variant swap happens AFTER the teleport check, right?
    # Let me re-read lines 2728-2742:
    # 1. Find tuulhnbosq = itki player is on (in azifrpswxp)
    # 2. If found: srlcdwxhhe = npqswumrzz(tuulhnbosq, azifrpswxp) -> teleport
    # 3. Then for each sprite in azifrpswxp: uqbvwhliqb -> swap variants

    # So TELEPORT happens FIRST, then variant swap.
    # At the time of teleport, the names haven't been swapped yet.

    # So if itkizfrq1 is at (18,48) (after 2 gkrr clicks):
    # npqswumrzz(itkizfrq1, ...) looks for uqbvwhliqb("itkizfrq1") = "itkizfrq2"
    # In the c-group: itkizfrq2 IS at (4,4)!
    # Teleport to (4,4)!

    # But wait, we need gkrr-jpug to be active first, which requires zbhi-d at (6,18).
    # And (6,18) is only reachable from the top-left kbqq cluster.
    # Which requires the hhxv bridge (variant 2) to be walkable.
    # Which requires clicking c once.

    # But after clicking c, the variant names swap. So the itki at (18,48)
    # would now be itkiupry2 (from variant swap). Its itki-color-cycle tag persists.

    # Let me trace the EXACT sequence:
    # 1. Initial: itkiupry1(18,48)[cc], itkiupry2(32,52), itkizfrq2(4,4), itkijbyz2(34,58)
    #    hhxv = variant 1 (not walkable)
    # 2. Click c (not on itki): variants swap ->
    #    itkiupry2(18,48)[cc], itkiupry1(32,52), itkizfrq1(4,4), itkijbyz1(34,58)
    #    hhxv = variant 2 (WALKABLE!)
    # 3. Walk to itkiupry2(18,48), click c: teleport to itkiupry1(32,52)
    #    [teleport first, then swap: itkiupry1->itkiupry2 etc]
    #    After swap: itkiupry1(18,48)[cc], itkiupry2(32,52), itkizfrq2(4,4), itkijbyz2(34,58)
    #    hhxv = variant 1 (NOT walkable) <-- PROBLEM! Bridge disappears!

    # Hmm, step 3 switches hhxv back to variant 1. So the bridge is no longer walkable
    # after teleporting. But the teleport puts us at (32,52) from where we can reach
    # zbhi-g at (34,48).

    # OK so steps 2-3 are: click c (bridge becomes walkable), walk to itki, click c
    # (teleport but bridge goes back). The bridge being walkable was only needed for
    # one brief period.

    # But wait, we need to reach (6,18) zbhi-d which requires walking through the bridge.
    # After step 2, bridge is walkable. We could walk to (6,18) before teleporting.
    # But (6,18) is on the LEFT side and (18,48) is where we teleport from.
    # Can we reach (6,18) from (18,48)?

    # After step 2 (hhxv variant 2 walkable):
    # - kbqq(18,48) x=18..23, y=48..53
    # - No path from x=18 to x=12 at y=48..53 (gap at x=12..17)
    # - Bridge at (0,24) variant 2 walkable at (6..12, 24..42)
    # - But to get ON the bridge from (18,48), we need floor between x=12..17 at
    #   some y level. There isn't any.

    # So even with the bridge walkable, we can't reach the left side from (18,48)!
    # We need to teleport to (4,4) to get to the left side.

    # And to teleport to (4,4), we need to cycle the itki color FIRST (via gkrr-jpug).
    # But gkrr-jpug requires zbhi-d at (6,18) which is on the left side.
    # STILL circular!

    # Unless there's another way. Let me check: what if we click c multiple times?
    # After odd c-clicks: hhxv variant 2 (walkable), even c-clicks: variant 1

    # What if we click c once (bridge walkable), then walk to (18,48),
    # then click c again (teleport, bridge goes to variant 1)?
    # After teleport: player at (32,52). OK that's the same as before.

    # What if we DON'T teleport? Just click c without being on itki:
    # Click c once: bridge walkable (variant 2)
    # But reach doesn't include left side.

    # Wait - what if itkiupry1 at (18,48) is the support for the player?
    # The kbqq-efzv-1 at (18,48) is 6x6 (x=18..23, y=48..53).
    # itkiupry1 at (18,48) is 2x2.
    # After c-click, itkiupry2 replaces itkiupry1. Both have gigzqgcfncq -> INTANGIBLE.
    # Their pixel values determine support.

    # Let me check: do itkiupry1 and itkiupry2 have different pixel patterns?
    # And similarly for itkizfrq, itkijbyz, itkibgeg?

    print("\n=== ITKI PIXEL ANALYSIS ===")
    # Check all itki sprite definitions
    for name in ["itkiupry1", "itkiupry2", "itkibgeg1", "itkibgeg2",
                  "itkizfrq1", "itkizfrq2", "itkijbyz1", "itkijbyz2"]:
        found = False
        for s in game.current_level.get_sprites():
            if s.name == name:
                pixels = s.render()
                print(f"  {name:15s} ({s.x:2d},{s.y:2d}) pixels={[list(row) for row in pixels]} int={s.interaction.name}")
                found = True
                break
        if not found:
            print(f"  {name:15s} NOT in level sprites")

    # BIG IDEA: What if the teleport mechanism allows teleporting to (34,58)?
    # itkijbyz2 is at (34,58). And (34,58) is ON a pressure plate!
    # If we could teleport to (34,58), we'd be on the pressure plate area!

    # Let me check: after one c-click, itkijbyz2(34,58) becomes itkijbyz1(34,58).
    # If we stand on itkijbyz1 at (34,58), click c:
    # - npqswumrzz: uqbvwhliqb("itkijbyz1") = "itkijbyz2"
    # - Looks for itkijbyz2 in c-group. Is there one?
    # After the variant swap, all will swap. But teleport happens FIRST.
    # At the time of teleport, the current names are the "1" variants (after odd c-clicks).
    # So if itkijbyz1 is at (34,58), it looks for itkijbyz2.
    # itkijbyz2 would be... where? Initially itkijbyz2 is at (34,58).
    # After one c-click, it cycles to itkijbyz1.
    # So after one c-click: no itkijbyz2 anywhere. Teleport fails.

    # UNLESS we can CREATE an itkijbyz2 somewhere via color cycling!

    # OK I think I need to try a different approach. Let me just use env.step
    # to manually execute sequences and see what happens.

    print("\n\n=== SYSTEMATIC SEQUENCE EXPLORATION ===")

    # Sequence 1: Click c once (bridge walkable), try full reach
    restore_game_state(game, state0)
    env.step(GameAction.ACTION6, data=ct['jpug-bjuk'])
    print(f"After 1 c-click: player ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    parents = player_reachable_cells(game)
    print(f"Reach: {len(parents)} cells")
    # Check if (4,4) is reachable (it shouldn't be)
    print(f"  (4,4) reachable? {(4,4) in parents}")
    print(f"  (6,18) reachable? {(6,18) in parents}")
    print(f"  (34,58) reachable? {(34,58) in parents}")

    # OK so c-click doesn't change reach because we're not on the left side.
    # The bridge being walkable doesn't help us if we can't reach it.

    # What if we walk to (18,48) itki, click c to teleport to (32,52),
    # then walk to zbhi-g at (34,48), then try to reach the plates?
    # From the earlier experiment, after teleport + zbhi pickup, reach is
    # (32..36, 48..52) - only 9 cells. Gap at y=54-55.

    # The ONLY way to cross that gap is to either:
    # A) Move the hhxv bridge there (but need crane which needs plates - circular)
    # B) Find hidden floor tiles we haven't seen
    # C) Use a mechanism we haven't tried

    # Let me check option B: are there any REMOVED sprites at (32-36, 54-55)?
    restore_game_state(game, state0)
    print("\n\nAll sprites at y=54-55:")
    for s in game.current_level.get_sprites():
        if s.y >= 53 and s.y <= 56:
            print(f"  {s.name:35s} ({s.x:3d},{s.y:3d}) {s.width}x{s.height} int={s.interaction.name} tags={s.tags}")

    # What if the itkijbyz at (34,58) can be used as a stepping stone?
    # It's at (34,58), right on a pressure plate. If we could teleport there directly...
    # Let me check: from (32,52) after zbhi-g, can we reach (34,58)?
    # No, there's no floor at y=54-55 between (34,52) and (34,56).

    # What about falling? If we walk off the edge at (34,52) going DOWN,
    # we'd try to move to (34,54). No support there -> fall.
    # Fall restores last click checkpoint and costs 20 steps.
    # Not helpful.

    # NEW IDEA: What about the qgdz on the LEFT side?
    # qgdz at (6,54) has 16x2 size. After f-cycling, the active pixels shift.
    # qgdz-efzv-6 has nonblank x=16..21.
    # If we f-cycle to get qgdz-6 at y=54 (f-click 5x), then the floor extends
    # from kbqq(6,48) down through qgdz(16..21, 54) to kbqq(6,56)... wait,
    # kbqq at (6,48) covers y=48..53. qgdz at y=54 with pixels at x=16..21
    # means floor at (16..21, 54). But there's no kbqq at (16,48).
    # And we can't reach x=16 from x=18 initial position because of the gap.

    # WAIT. What about x=18? kbqq(18,48) covers x=18..23, y=48..53.
    # qgdz at (6,54) with variant 6 has nonblank at x=16..21.
    # That's x=16+6=22 max. But the sprite is at x=6, so pixel col 10..15
    # maps to x=16..21. Player at x=18 can step onto qgdz at (18,54)!
    # Then from y=54, maybe walk further right/down to the qgdz at y=56, y=58...

    # Let me check this carefully!
    print("\n\n=== F-CYCLE STAIRCASE FROM x=18 ===")
    # After 5 f-clicks, qgdz-efzv-6 is at y=54 with nonblank x=16..21
    # After 4 f-clicks, qgdz-efzv-5 is at y=54 with nonblank x=14..19
    # After 3 f-clicks, qgdz-efzv-4 is at y=54 with nonblank x=12..17
    # etc.

    # We want to create a staircase from x=18 area to x=32 area.
    # qgdz sprites are at y=54, y=56, y=58. Each 16x2.
    # Variants have 6-pixel-wide blocks at different x positions.
    # Can we choose f-cycle count to create a continuous path from
    # x=18 down through y=54->56->58 and then to kbqq(32,56)?

    # After N f-clicks:
    # y=54 gets variant: cycle[0] -> 1,2,3,4,5,6,1,2...
    # y=56 gets variant: cycle[1] -> 2,1,3,5,4,6...
    # y=58 gets variant: cycle[2] -> 4,5,6,1,2,3...
    # (based on earlier experiment output)

    # Actually let me just test each f-click count and check reach

    for n_f in range(7):
        restore_game_state(game, state0)
        for i in range(n_f):
            env.step(GameAction.ACTION6, data=ct['sprite-6'])
        parents = player_reachable_cells(game)

        # Check specific y levels
        has_54 = any(y == 54 for _, y in parents.keys())
        has_56 = any(y == 56 for _, y in parents.keys())
        has_58 = any(y == 58 for _, y in parents.keys())
        max_y = max(y for _, y in parents.keys()) if parents else 0
        max_x = max(x for x, y in parents.keys()) if parents else 0

        # Check qgdz active sprites
        qgdz_54 = qgdz_56 = qgdz_58 = None
        for s in game.current_level.get_sprites():
            if 'qgdz' in s.name and s.interaction != InteractionMode.REMOVED:
                if s.y == 54: qgdz_54 = s.name
                elif s.y == 56: qgdz_56 = s.name
                elif s.y == 58: qgdz_58 = s.name

        print(f"  f-clicks={n_f}: reach={len(parents)}, has_y54={has_54}, y56={has_56}, y58={has_58}, "
              f"max_y={max_y}, qgdz54={qgdz_54}, qgdz56={qgdz_56}, qgdz58={qgdz_58}")

        if has_54 or has_56 or has_58:
            ys_ext = sorted(set(y for _, y in parents.keys() if y >= 52))
            for y in ys_ext:
                xs = sorted(x for x, yy in parents.keys() if yy == y)
                print(f"    y={y}: x={xs}")

    # Now try ALSO clicking c (bridge variant 2) + f-clicks
    print("\n\n=== C-CLICK + F-CYCLE COMBINATIONS ===")
    for n_c in [1]:  # c-click odd = bridge walkable
        for n_f in range(7):
            restore_game_state(game, state0)
            for i in range(n_c):
                env.step(GameAction.ACTION6, data=ct['jpug-bjuk'])
            for i in range(n_f):
                env.step(GameAction.ACTION6, data=ct['sprite-6'])
            parents = player_reachable_cells(game)

            has_54 = any(y == 54 for _, y in parents.keys())
            print(f"  c={n_c} f={n_f}: reach={len(parents)}, has_y54={has_54}")
            if len(parents) > 18:
                ys = sorted(set(y for _, y in parents.keys()))
                for y in ys:
                    xs = sorted(x for x, yy in parents.keys() if yy == y)
                    print(f"    y={y}: x={xs}")

if __name__ == "__main__":
    main()
