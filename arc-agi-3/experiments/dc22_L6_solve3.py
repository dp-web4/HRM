#!/usr/bin/env python3
"""dc22 L6 - understand hhxv bridge and figure out how to bridge the gap."""
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
    targets = []
    cam_h = game.camera._height
    y_offset = (64 - cam_h) // 2
    seen_coords = set()
    for s in game.current_level.get_sprites():
        if 'jpug' in s.tags and 'sys_click' in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2 + y_offset
            if (cx, cy) not in seen_coords:
                seen_coords.add((cx, cy))
                targets.append(('jpug', s.name, cx, cy))
        elif 'sys_click' in s.tags and 'jpug' not in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2 + y_offset
            if (cx, cy) not in seen_coords:
                seen_coords.add((cx, cy))
                targets.append(('sys_click', s.name, cx, cy))
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

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    replay_to_L6(env)
    game = env._game

    print("=== L6: UNDERSTANDING THE BRIDGE SUPPORT ===")

    # The hhxv bridge has -2 pixels. But bwxffbswqz checks for -2 specifically:
    # if mmpptbeooz == -2: returns the sprite if not REMOVED
    # But uxwpppoljm checks if pixel < 0 -> continue (skip). So -2 IS skipped.
    # This means the hhxv bridge does NOT provide support!

    # But wait - maybe when the bridge is in variant 2, the pixels change?
    # Let me check hhxv-dmxj2 pixels

    print("\nhhxv-dmxj2 pixels:")
    for s in game.current_level.get_sprites():
        if s.name == 'hhxv-dmxj2':
            pixels = s.render()
            print(f"  {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name}")
            unique_vals = set()
            for row in range(pixels.shape[0]):
                for col in range(pixels.shape[1]):
                    unique_vals.add(int(pixels[row][col]))
            print(f"  Unique pixel values: {sorted(unique_vals)}")
            # Show a few rows
            for row in [0, 1, 9, 10, 18, 19]:
                if row < pixels.shape[0]:
                    print(f"  row {row}: {list(pixels[row])}")

    # Check the djvhljplfg function - it determines if a sprite is grabbable as hhxv
    # Requirements: name ends in digit, prefix starts with "hhxv", qmxenejanqe has 2 variants, has "wbze" tag
    print("\nGrabbable hhxv check:")
    for s in game.current_level.get_sprites():
        if 'hhxv' in s.name and s.interaction != InteractionMode.REMOVED:
            is_grabbable = game.djvhljplfg(s)
            print(f"  {s.name}: grabbable={is_grabbable} tags={s.tags}")

    # The hhxv bridge has 'wbze' tag and 'c' tag. It's part of the c-group.
    # When c is clicked, hhxv-dmxj1 cycles to hhxv-dmxj2.
    # Both variants have gigzqgcfncq -> INTANGIBLE (not collidable = no blocking).
    # Both have -2 pixels -> no support either.
    # So what's the point of the bridge?!

    # OH WAIT. The bridge IS a bridge piece for the CRANE. The crane can GRAB it
    # and MOVE it. When grabbed and moved, the bridge creates walkable terrain
    # at the new location!

    # Actually no - the bridge still has -2 pixels wherever you put it.
    # Unless... the act of attaching it to the crane changes something?

    # Let me look at how the crane hhxv attachment works more carefully.
    # From the source (line 2831-2869): when evlrmeiyoy (crane moved):
    # - If hhxv attached: compute target position, check collision, then:
    #   - lreqqfwsxn: moves all sprites with matching prefix from old to new position
    #   - dldhnotovw: re-finds the attached sprite reference
    # This moves ALL hhxv-dmxj sprites (variant 1 AND 2) together.

    # The key is that hhxv-dmxj with -2 pixels acts as a COLLISION bridge.
    # Other sprites can collide with it (via bwxffbswqz which detects -2 pixels).
    # But it doesn't provide support (uxwpppoljm skips < 0).

    # So the hhxv bridge is NOT the floor. It's something else.
    # Maybe it's used to block falls? No - INTANGIBLE means not collidable.

    # Let me re-examine. INTANGIBLE = interaction mode.
    # is_collidable property: for INTANGIBLE, returns False.
    # But bwxffbswqz checks: if pixel == -2: returns sprite regardless of collidable
    # And uxwpppoljm checks: if sprite._interaction == INTANGIBLE and pixel >= 0: returns sprite

    # So for hhxv with -2 pixels and INTANGIBLE:
    # - bwxffbswqz: MATCHES (returns sprite for -2 regardless)
    # - uxwpppoljm: SKIPS (pixel < 0, continue)
    # - collides_with: SKIPS (not collidable)

    # The hhxv bridge is detected by bwxffbswqz (used for zbhi key detection).
    # It doesn't block or support the player. It's basically invisible to movement.

    # BUT WAIT: the crane collision check (qioieiaucx) uses tzwaxkrztd which
    # checks pixel overlap between bridges. This prevents bridges from overlapping.
    # So the hhxv bridge occupies space for OTHER bridges but doesn't affect the player.

    # Actually, I think I'm overthinking this. The bridge's PURPOSE might be
    # revealed when we look at what happens when it's MOVED.
    # When the crane grabs and moves it, the NEW position might overlap with
    # floor tiles, effectively extending the walkable area.

    # Actually, the simplest interpretation: the hhxv bridge IS the walkable floor
    # for connecting otherwise disconnected areas. But its -2 pixels mean it
    # ISN'T walkable as-is. The -2 just means it's collidable-only (blocks other
    # sprites from overlapping but doesn't support player).

    # Hmm. Let me look at this from a different angle.
    # What if the c-click cycling is the key? When we click c, the hhxv-dmxj
    # cycles between variant 1 and 2. But both have -2 pixels.
    # Unless one variant has different pixel values?

    # I already see that hhxv-dmxj1 and hhxv-dmxj2 are separate sprites.
    # Let me check hhxv-dmxj2 pixel values more carefully.

    # Also - could the itki-color-cycle on itkiupry1(18,48) do something special?
    # When c is clicked with itki-color-jpug=False, the regular variant swap happens.
    # The itkiupry1 at (18,48) has itki-color-cycle tag but jpug-bjuk does NOT have
    # itki-color-jpug tag. So fnhzudfjhd is NOT called. Instead, the regular
    # variant swap + teleport happens.

    # BUT: the itki-color-cycle tag was specifically added to the sprite at (18,48)
    # in the L6 on_set_level code (line 2026-2029). AND it also adds tag 'd'.
    # This means itkiupry1 at (18,48) is part of the 'd' group too!

    # When we pick up zbhi-jpug-adnw-1 (tag 'd') at (6,18), all jpug sprites
    # with tag 'd' become INTANGIBLE. The itki at (18,48) has tag 'd' but it's
    # already INTANGIBLE. And the grab button? No, that has tag 'g'.

    # Wait - zbhi mechanism: walk over zbhi with single-letter tag -> all jpug sprites
    # with that letter tag become INTANGIBLE and the zbhi is REMOVED.

    # Actually, the zbhi code at line 2913-2921:
    # if ijdmcivcnf: (letter tag)
    #   dkbscduxgr = get_sprites_by_tag("jpug")
    #   for wldbhnwqbn in dkbscduxgr:
    #     if ijdmcivcnf in wldbhnwqbn.tags:
    #       wldbhnwqbn._interaction = INTANGIBLE
    #       self.current_level.remove_sprite(uejgczquds)

    # This sets jpug sprites with the matching letter to INTANGIBLE.
    # For zbhi-jpug-bgeg-1 (tag 'g'): makes jpug+'g' sprites INTANGIBLE.
    # That includes nxhz-bynyvtuepbt-1 (tags: ['jpug', 'g', 'bynyvtuepbt', 'sys_click'])

    # For zbhi-jpug-adnw-1 (tag 'd'): makes jpug+'d' sprites INTANGIBLE.
    # But what has jpug+'d'? Let me check.

    print("\n\nSprites with 'd' tag:")
    for s in game.current_level.get_sprites():
        if 'd' in s.tags:
            print(f"  {s.name} tags={s.tags} int={s.interaction.name}")

    print("\nSprites with 'jpug' AND 'd' tags:")
    for s in game.current_level.get_sprites():
        if 'jpug' in s.tags and 'd' in s.tags:
            print(f"  {s.name} tags={s.tags} int={s.interaction.name}")

    # So the d-zbhi might not unlock anything directly.
    # Let me check the gkrr-jpug sprite mentioned in the mechanics

    print("\ngkrr-jpug sprite:")
    for s in game.current_level.get_sprites():
        if 'gkrr' in s.name:
            print(f"  {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name} tags={s.tags}")

    # OK let me take a completely different approach.
    # Let me look at the actual screenshot and check what things look like visually.
    save_frame(env.observation_space.frame, f"{VIS}/L6_initial_clean.png")

    # The fundamental problem is: how do we get from y=52 to y=56?
    # The gap at y=54-55 at x=32-37 has NO floor tiles.
    # Options:
    # 1. Move the hhxv bridge to cover it -> but -2 pixels don't support
    # 2. Use qgdz to extend floor -> qgdz only covers x=6..21
    # 3. Use the itki-color-cycle somehow -> it just swaps itki names
    # 4. Something with the 'd' tag and itki-color-cycle

    # Actually, wait. Let me re-read line 2026-2029:
    # if self.level_index == 5:
    #     for rynhunhbyw in self.current_level.get_sprites_by_tag("itki"):
    #         if rynhunhbyw.x == 18 and rynhunhbyw.y == 48:
    #             rynhunhbyw.tags.append(vgiqmhpyxb)  # 'itki-color-cycle'
    #             rynhunhbyw.tags.append("d")
    #             break

    # This adds itki-color-cycle AND 'd' tags to the itki at (18,48).
    # But what does itki-color-cycle DO when jpug-bjuk does NOT have itki-color-jpug?
    # Looking at the click handler (line 2724-2742):
    # if bpnwmawiuv in wldbhnwqbn.tags: (itki-color-jpug)
    #     self.fnhzudfjhd(azifrpswxp)
    # else:
    #     ... (regular variant swap + teleport)

    # jpug-bjuk does NOT have itki-color-jpug, so the regular path is taken.
    # The itki-color-cycle tag on the sprite has no effect during the regular path.
    # UNLESS it's used elsewhere...

    # Let me check if itki-color-cycle is referenced anywhere else
    # vgiqmhpyxb = "itki-color-cycle" is used in fnhzudfjhd
    # fnhzudfjhd: finds sprites with "itki" + vgiqmhpyxb + valid itki name + not REMOVED
    # Then cycles their names through bscfvnwogt

    # So itki-color-cycle sprites get their NAME changed when fnhzudfjhd is called.
    # But fnhzudfjhd is only called when the clicked jpug has itki-color-jpug tag.
    # And jpug-bjuk doesn't have that tag.
    # Is there another jpug that has itki-color-jpug? Let me check.

    print("\n\nSprites with itki-color-jpug tag:")
    for s in game.current_level.get_sprites():
        if 'itki-color-jpug' in s.tags:
            print(f"  {s.name} tags={s.tags} int={s.interaction.name}")

    # Maybe the gkrr-jpug has it?

    print("\ngkrr-jpug details:")
    for s in game.current_level.get_sprites():
        if s.name == 'gkrr-jpug':
            print(f"  {s.name} ({s.x},{s.y}) tags={s.tags} int={s.interaction.name}")

    # Wait, the mechanics doc says gkrr-jpug is a "Color group display" with
    # 4-quadrant tile (colors 8,11,10,7). Shows color cycling state.
    # And there's a mention of itki-color-jpug enabling click-to-cycle on gate buttons.

    # Maybe clicking gkrr-jpug triggers color cycling? Let me check if it has
    # sys_click tag.

    state0 = save_game_state(game)

    # Let me also look at the d-zbhi more carefully.
    # zbhi-jpug-adnw-1 is at (6,18). It's on the LEFT side of the map.
    # To reach it, we'd need to get to the top-left kbqq cluster.
    # From (18,48), we can't reach (6,18) directly.

    # What if we use the hhxv bridge to create a path?
    # The bridge at (0,24) 20x20 connects the left kbqq cluster to kbqq(6,44)/(8,44).
    # But we can't walk on the bridge (pixels are -2).

    # WAIT. Let me re-check. The kbqq at (4,16) (8,16) (4,20) (8,20) are at x=4..11, y=16..23
    # The kbqq at (6,44) (8,44) are at x=6..11, y=44..47
    # The kbqq at (6,48) is at x=6..11, y=48..53
    # So: (4..11, 16..23) then gap at y=24..43 then (6..11, 44..53)

    # The hhxv bridge at (0,24) covers (0..19, 24..43). But with -2 pixels it's
    # not walkable. So these two kbqq clusters are NOT connected.

    # Unless... the player can walk through INTANGIBLE sprites (-2 pixels are
    # not blocking since sprite is INTANGIBLE), and then fall? No, that costs 20 steps
    # and restores checkpoint.

    # Hmm. Let me think about this differently.
    # The hhxv bridge can be GRABBED by the crane and MOVED.
    # But to grab, we need the grab button active (need zbhi-g at (34,48)).
    # And to move crane, we need pressure plate buttons.
    # And to reach pressure plates, we need to bridge the y=54-55 gap.
    # This IS circular.

    # UNLESS there's another way to activate the crane buttons.
    # The pressure plate code (mbszrqqnqm) checks if player overlaps pressure plates.
    # When overlapping, it makes the matching-letter jpug sprites visible and INTANGIBLE.
    # The direction buttons are: a=right, b=down, e=left, h=up

    # Can we reach the pressure plates via the LEFT side?
    # From (6,48) kbqq-efzv-1, the qgdz at (6,54) extends to x=6..11 at y=54-55
    # Then qgdz at (6,56) extends to x=8..13 at y=56-57
    # Then qgdz at (6,58) extends to x=12..17 at y=58-59
    # These provide a STAIRCASE going right as you go down!
    # But they only reach x=17 max. Plates are at x=32-36.

    # With f-cycling, the qgdz positions shift. qgdz-efzv-6 has nonblank x=16..21.
    # But that's still not enough to reach x=32.

    # What if multiple f-clicks create overlapping paths?
    # Each f-click cycles ALL three qgdz sprites simultaneously.

    # Actually, let me carefully check: can we physically walk from the initial
    # position (28,52) all the way left to (6,48) via kbqq-efzv-1 tiles?

    # Initial reach: x=18..28, y=48..52
    # kbqq-efzv-1 at (18,48): covers x=18..23, y=48..53
    # kbqq-efzv-1 at (24,48): covers x=24..29, y=48..53
    # kbqq-efzv-1 at (6,48): covers x=6..11, y=48..53
    # Gap at x=12..17, y=48..53 - no floor!

    # kbqq-efzv-1 at (18,48) has width 6, so covers x=18..23
    # kbqq-efzv-1 at (6,48) has width 6, so covers x=6..11
    # x=12..17 has NO kbqq, NO qgdz, nothing.

    # So we CANNOT walk from the initial area to the left kbqq cluster.
    # Unless the hhxv bridge is moved to bridge that gap.
    # But moving the bridge requires crane, which requires plates, which requires...

    # OK so the itki at (18,48) has tag 'd'. And the d-zbhi is at (6,18).
    # If we could get to (6,18) and pick up the d-zbhi, all jpug+d sprites
    # become INTANGIBLE. The itki at (18,48) would become... already INTANGIBLE.
    # What else has jpug+d?

    # Let me check if gkrr-jpug has tag 'd'
    for s in game.current_level.get_sprites():
        if s.name == 'gkrr-jpug':
            print(f"\ngkrr-jpug tags: {s.tags}")
            has_d = 'd' in s.tags
            print(f"  Has 'd' tag: {has_d}")

    # What if the gkrr-jpug button, when unlocked, triggers the itki-color-cycle?
    # Let me check its tags for itki-color-jpug

    # Actually, maybe I need to look at ALL d-tagged jpug sprites
    print("\nAll jpug sprites with single-letter tags:")
    for s in game.current_level.get_sprites():
        if 'jpug' in s.tags:
            letter = next((t for t in s.tags if len(t) == 1), None)
            if letter:
                print(f"  {s.name:35s} letter={letter} int={s.interaction.name} vis={s.is_visible} tags={s.tags}")

    # Maybe the solution involves a COMPLETELY different approach.
    # What if we can click the crane direction buttons DIRECTLY (they're on the
    # right panel at fixed positions) without needing pressure plates?

    # The buttons are INVISIBLE initially. But the click handler (ozbgzuaoya) checks:
    # - interaction != REMOVED and != INVISIBLE
    # - is_visible = True
    # So INVISIBLE sprites CAN'T be clicked. But INTANGIBLE can be.

    # The pressure plates toggle them between INVISIBLE and INTANGIBLE.
    # There's no other way to make them visible.

    # OK, ultimate question: what creates floor at the gap positions?
    # Let me look for ANY sprite that could provide support at (30,54) or (32,54)

    print("\n\n=== EXHAUSTIVE SUPPORT CHECK: y=54-55 ===")
    for y in [54, 55]:
        for x in range(0, 50, 2):
            supp = game.uxwpppoljm(x, y, game.fdvakicpimr)
            if supp:
                print(f"  ({x},{y}): supported by {supp.name} ({supp.x},{supp.y}) {supp.width}x{supp.height}")

    # And x=12..17 at y=48..53
    print("\n=== EXHAUSTIVE SUPPORT CHECK: x=12-17, y=48-53 ===")
    for y in range(48, 54, 2):
        for x in range(12, 18, 2):
            supp = game.uxwpppoljm(x, y, game.fdvakicpimr)
            if supp:
                print(f"  ({x},{y}): supported by {supp.name} ({supp.x},{supp.y}) {supp.width}x{supp.height}")

    # Now let's try something radical: what if we click c FIRST (not from an itki,
    # just click it), which cycles variants. Then check if anything changed.

    print("\n\n=== EXPERIMENT: Click c WITHOUT being on itki ===")
    c_data = None
    for k, n, cx, cy in find_click_targets(game):
        if n == 'jpug-bjuk':
            c_data = {'x': cx, 'y': cy}
            break

    fd = env.step(GameAction.ACTION6, data=c_data)
    print(f"After c-click (not on itki): player at ({game.fdvakicpimr.x},{game.fdvakicpimr.y}), state={fd.state.name}")

    # Check what changed
    print("\nItki states after c-click (no teleport):")
    for s in game.current_level.get_sprites():
        if 'itki' in s.tags and s.interaction != InteractionMode.REMOVED:
            print(f"  {s.name} ({s.x},{s.y}) int={s.interaction.name} tags={s.tags}")

    print("\nhhxv after c-click:")
    for s in game.current_level.get_sprites():
        if 'hhxv-dmxj' in s.name and s.interaction != InteractionMode.REMOVED:
            print(f"  {s.name} ({s.x},{s.y}) int={s.interaction.name}")

    # Check if itkiupry1 moved positions (the one with itki-color-cycle)
    for s in game.current_level.get_sprites():
        if 'itki-color-cycle' in s.tags and s.interaction != InteractionMode.REMOVED:
            print(f"\nitki-color-cycle sprite: {s.name} ({s.x},{s.y})")

    # Check reach after this click
    parents_after_c = player_reachable_cells(game)
    print(f"\nReach after c-click (no teleport): {len(parents_after_c)} cells")
    ys = sorted(set(y for _, y in parents_after_c.keys()))
    for y in ys:
        xs = sorted(x for x, yy in parents_after_c.keys() if yy == y)
        print(f"  y={y}: x={xs}")

    # What if the c-click when not on itki doesn't trigger teleport but still cycles?
    # That would change the hhxv variant, which might change the pixel pattern

    # Let me check support at the gap after hhxv-dmxj2 is active
    print("\nSupport at gap after hhxv-dmxj2:")
    for y in range(24, 44, 2):
        for x in range(0, 20, 2):
            supp = game.uxwpppoljm(x, y, game.fdvakicpimr)
            if supp and 'hhxv' in supp.name:
                print(f"  ({x},{y}): supported by {supp.name}")

    # Check hhxv-dmxj2 pixels
    for s in game.current_level.get_sprites():
        if s.name == 'hhxv-dmxj2' and s.interaction != InteractionMode.REMOVED:
            pixels = s.render()
            unique = set()
            for row in range(pixels.shape[0]):
                for col in range(pixels.shape[1]):
                    unique.add(int(pixels[row][col]))
            print(f"\nhhxv-dmxj2 pixel values: {sorted(unique)}")
            # Show all rows
            for row in range(pixels.shape[0]):
                print(f"  row {row}: {list(pixels[row])}")

    restore_game_state(game, state0)

    # Try clicking gkrr-jpug
    print("\n\n=== EXPERIMENT: Click gkrr-jpug ===")
    gkrr_data = None
    for k, n, cx, cy in find_click_targets(game):
        if n == 'gkrr-jpug':
            gkrr_data = {'x': cx, 'y': cy}
            print(f"  gkrr-jpug click at ({cx},{cy})")
            break

    if gkrr_data:
        fd = env.step(GameAction.ACTION6, data=gkrr_data)
        print(f"  After gkrr-jpug click: state={fd.state.name}")

        # Check what changed
        for s in game.current_level.get_sprites():
            if 'itki' in s.tags and s.interaction != InteractionMode.REMOVED:
                if 'itki-color-cycle' in s.tags:
                    print(f"  itki-color-cycle: {s.name} ({s.x},{s.y})")
    else:
        print("  gkrr-jpug NOT in click targets")

    restore_game_state(game, state0)

if __name__ == "__main__":
    main()
