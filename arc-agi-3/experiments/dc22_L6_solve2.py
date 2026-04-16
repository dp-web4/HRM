#!/usr/bin/env python3
"""dc22 L6 solver - deeper investigation of the gap and bridge mechanics."""
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

    print("=== L6 DEEP INVESTIGATION ===")
    p = game.fdvakicpimr
    print(f"Player: ({p.x},{p.y})")

    # Check what sprites support player at specific y positions in the (32,48) area
    # kbqq-efzv-1 at (32,48) is 6x6, so it covers (32..37, 48..53)
    # But the player can only reach y=48,50,52 (from the reach output)
    # Let's check support at y=54 to understand the gap
    print("\n=== SUPPORT CHECK: x=32..37, y=48..62 ===")
    for y in range(48, 63, 2):
        for x in range(30, 40, 2):
            supp = game.uxwpppoljm(x, y, game.fdvakicpimr)
            if supp:
                print(f"  ({x},{y}): supported by {supp.name} ({supp.x},{supp.y}) {supp.width}x{supp.height} int={supp.interaction.name}")
            else:
                # Check why not supported - look at pixel values
                for s in game.current_level.get_sprites():
                    if s.interaction == InteractionMode.REMOVED or s.interaction != InteractionMode.INTANGIBLE:
                        continue
                    if 'ignore' in s.tags or 'nxhz' in s.tags or 'path' in s.tags:
                        continue
                    if x >= s.x and y >= s.y and x < s.x + s.width and y < s.y + s.height:
                        pixels = s.render()
                        pv = pixels[y - s.y][x - s.x]
                        if pv != -1:
                            print(f"  ({x},{y}): {s.name} ({s.x},{s.y}) pixel={pv} SHOULD support?")

    # Check hhxv-dmxj1 bridge at (0,24) 20x20 - covers (0..19, 24..43)
    print("\n=== HHXV-DMXJ1 BRIDGE ===")
    for s in game.current_level.get_sprites():
        if 'hhxv' in s.name and s.interaction != InteractionMode.REMOVED:
            print(f"  {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name}")
            # Check pixel pattern at edges
            pixels = s.render()
            print(f"  Pixel array shape: {pixels.shape}")
            # Show a few rows
            for row in range(min(3, pixels.shape[0])):
                print(f"    row {row}: {list(pixels[row][:10])}...")
            for row in range(max(0, pixels.shape[0]-3), pixels.shape[0]):
                print(f"    row {row}: {list(pixels[row][:10])}...")

    # Check: the hhxv bridge at (0,24) covers x=0..19, y=24..43
    # The kbqq tiles in that area: (4,4) (8,4) etc up top, (4,16) (8,16) (4,20) (8,20) in mid
    # Also kbqq at (6,44) (8,44) at bottom
    # So the hhxv bridge creates a path from (0..19, 24..43) - connecting mid kbqq to bottom kbqq

    # Key question: can we move the hhxv bridge to bridge the gap at (32..37, 54..55)?
    # That would connect kbqq(32,48) to kbqq(32,56)

    # But first we need to get to the pressure plates to activate crane buttons.
    # From the teleport position (32,52), reach is (32-36, 48-52).
    # We need to reach (32-36, 56-60) where the pressure plates are.
    # The gap is y=54-55.

    # What about the itki color cycling? The itkiupry1 at (18,48) has itki-color-cycle tag.
    # Clicking c while on it should cycle all itki-color-cycle sprites.
    # But wait - the c button (jpug-bjuk) does TWO things:
    # 1. Color cycling (fnhzudfjhd) if tagged itki-color-jpug
    # 2. Teleport + variant swap
    # Which one does jpug-bjuk trigger?

    # Let's check the jpug-bjuk tags
    print("\n=== JPUG-BJUK ANALYSIS ===")
    for s in game.current_level.get_sprites():
        if s.name == 'jpug-bjuk':
            print(f"  jpug-bjuk: tags={s.tags}")
            # Check if it has itki-color-jpug tag
            print(f"  Has itki-color-jpug? {'itki-color-jpug' in s.tags}")

    # The c letter group includes itkis AND the hhxv bridge!
    # hhxv-dmxj1 has tag 'c'!
    # When we click jpug-bjuk (letter c), it cycles all sprites with tag 'c'
    # That includes:
    # - itkizfrq2 at (4,4) - itki color cycle
    # - itkiupry1 at (18,48) - itki with color-cycle tag
    # - itkiupry2 at (32,52) - itki
    # - itkijbyz2 at (34,58) - itki
    # - hhxv-dmxj1 at (0,24) - the bridge!

    # When hhxv-dmxj1 cycles, what happens?
    # It's a wbze with name ending in '1', so uqbvwhliqb gives "hhxv-dmxj2"
    # The cycle swaps hhxv-dmxj1 -> hhxv-dmxj2
    # hhxv-dmxj2 would be at the same position (0,24) but with different pixels!

    print("\n=== HHXV VARIANT ANALYSIS ===")
    # Check all hhxv-dmxj variants in sprites dict
    for s in game.current_level.get_sprites():
        if 'hhxv-dmxj' in s.name:
            print(f"  {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name} tags={s.tags}")

    # Let's actually try clicking c and see what happens to the bridge
    state0 = save_game_state(game)

    print("\n=== EXPERIMENT: Click c button ===")
    # Find c target
    c_data = None
    for k, n, cx, cy in find_click_targets(game):
        if n == 'jpug-bjuk':
            c_data = {'x': cx, 'y': cy}
            break

    print(f"  Before c-click:")
    for s in game.current_level.get_sprites():
        if 'hhxv-dmxj' in s.name and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name}")

    fd = env.step(GameAction.ACTION6, data=c_data)
    print(f"  After c-click (state={fd.state.name}):")
    for s in game.current_level.get_sprites():
        if 'hhxv-dmxj' in s.name and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name}")

    # Check itki changes
    print(f"  Itki states:")
    for s in game.current_level.get_sprites():
        if 'itki' in s.tags and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) int={s.interaction.name} tags={s.tags}")

    # Click c again
    fd = env.step(GameAction.ACTION6, data=c_data)
    print(f"\n  After 2nd c-click (state={fd.state.name}):")
    for s in game.current_level.get_sprites():
        if 'hhxv-dmxj' in s.name and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name}")
    for s in game.current_level.get_sprites():
        if 'itki' in s.tags and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) int={s.interaction.name}")

    # Click c third time
    fd = env.step(GameAction.ACTION6, data=c_data)
    print(f"\n  After 3rd c-click (state={fd.state.name}):")
    for s in game.current_level.get_sprites():
        if 'hhxv-dmxj' in s.name and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name}")

    # Click c fourth time (full cycle)
    fd = env.step(GameAction.ACTION6, data=c_data)
    print(f"\n  After 4th c-click (state={fd.state.name}):")
    for s in game.current_level.get_sprites():
        if 'hhxv-dmxj' in s.name and s.interaction != InteractionMode.REMOVED:
            print(f"    {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name}")

    # Restore
    restore_game_state(game, state0)

    # Now the REAL experiment: the itki-color-cycle mechanism
    # When jpug-bjuk has itki-color-jpug tag (bpnwmawiuv), fnhzudfjhd is called
    # otherwise the regular variant swap happens
    # Let's check: does jpug-bjuk have itki-color-jpug tag?

    print("\n=== CHECK COLOR-CYCLE vs TELEPORT ===")
    # From source line 2725: if bpnwmawiuv in wldbhnwqbn.tags: self.fnhzudfjhd(azifrpswxp)
    # bpnwmawiuv = "itki-color-jpug"
    # wldbhnwqbn = the clicked sprite
    for s in game.current_level.get_sprites():
        if s.name == 'jpug-bjuk':
            has_color_jpug = 'itki-color-jpug' in s.tags
            print(f"  jpug-bjuk tags: {s.tags}")
            print(f"  Has itki-color-jpug: {has_color_jpug}")

    # Now try the REAL sequence: walk to itki, click c to teleport,
    # then check what the f-cycle does
    print("\n\n=== F-CYCLE EXPERIMENT ===")
    # sprite-6 is the f button at (53,5)
    f_data = None
    for k, n, cx, cy in find_click_targets(game):
        if n == 'sprite-6':
            f_data = {'x': cx, 'y': cy}
            break

    print(f"  qgdz before f-clicks:")
    for s in game.current_level.get_sprites():
        if 'qgdz' in s.name and s.interaction != InteractionMode.REMOVED:
            pixels = s.render()
            print(f"    {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name} pixels[0]={list(pixels[0][:8])}")

    for i in range(1, 7):
        fd = env.step(GameAction.ACTION6, data=f_data)
        print(f"\n  After f-click #{i}:")
        for s in game.current_level.get_sprites():
            if 'qgdz' in s.name and s.interaction != InteractionMode.REMOVED:
                print(f"    {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name}")

    restore_game_state(game, state0)

    # Now let's check the actual pixel support at critical positions when hhxv bridge
    # is moved. The hhxv-dmxj1 is at (0,24) 20x20. Can the crane grab it?
    print("\n\n=== CRANE GRAB ANALYSIS ===")
    print(f"Crane at ({game.nxhz_x},{game.nxhz_y}) pixel ({game.nxhz.x},{game.nxhz.y})")
    print(f"Crane size: {game.nxhz.width}x{game.nxhz.height}")
    print(f"hhxv bridge center: ({0+10},{24+10}) = (10,34)")
    anchor_x, anchor_y = game.venypzwjkd()
    print(f"Crane anchor: ({anchor_x},{anchor_y})")

    # For hhxv mode, anchor = nxhz.x + nxhz.width//2, nxhz.y + nxhz.height//2
    # crane at (22,30) with 8x8 -> anchor at (26,34)
    # hhxv bridge at (0,24) 20x20 -> center at (10,34)
    # So to grab, crane anchor must = bridge center. Need crane anchor at (10,34)
    # That means nxhz pixel at (10 - 4, 34 - 4) = (6, 30)
    # nxhz pixel = nxhz_start[0] + nxhz_x*4, nxhz_start[1] - nxhz_y*4
    # = 22 + nxhz_x*4, 30 - nxhz_y*4
    # Need: 22 + nxhz_x*4 = 6 -> nxhz_x = -4
    #        30 - nxhz_y*4 = 30 -> nxhz_y = 0
    # nxhz_x = -4, nxhz_y = 0
    # Is (-4, 0) a valid crane position?
    print(f"\nIs crane pos (-4, 0) valid? {game.ogwbggfvor(-4, 0)}")
    print(f"Is crane pos (-3, 0) valid? {game.ogwbggfvor(-3, 0)}")

    # At (-3, 0): pixel (10, 30), anchor at (14, 34). Bridge center at (10, 34). No match.
    # Need to find a crane position where anchor matches bridge center

    # Actually, for hhxv grab, the function fnonlfqqca checks anchor vs bridge center:
    # anchor_x = nxhz.x + nxhz.width//2 = pixel_x + 4
    # anchor_y = nxhz.y + nxhz.height//2 = pixel_y + 4
    # bridge center = bridge.x + bridge.width//2, bridge.y + bridge.height//2
    # = 0 + 10, 24 + 10 = (10, 34)
    # Need anchor at (10, 34): pixel at (6, 30)
    # nxhz_x such that 22 + nxhz_x*4 = 6 -> nxhz_x = -4
    # Not valid! Max negative is -3.

    # So we CAN'T grab the bridge from initial position. We need to move crane
    # to a position where its anchor aligns with bridge center.
    # But we can't move crane because the buttons are gated by pressure plates!

    # Wait... let's look at this differently. Can the grab button be activated
    # WITHOUT a pressure plate? The grab button is nxhz-bynyvtuepbt-1 which has tag 'g'.
    # It's initially INVISIBLE (hidden by pressure plate logic? No, it's activated by zbhi-jpug-bgeg-1)

    # Actually, zbhi-jpug-bgeg-1 at (34,48) has tag 'g'. When player walks over it,
    # all jpug sprites with tag 'g' become INTANGIBLE. The grab button nxhz-bynyvtuepbt-1
    # has... wait, let me check its tags.

    print("\n=== GRAB BUTTON ANALYSIS ===")
    for s in game.current_level.get_sprites():
        if 'bynyvtuepbt' in s.name:
            print(f"  {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name} vis={s.is_visible} tags={s.tags}")

    # The zbhi mechanism: walking over zbhi with tag 'g' unlocks jpug sprites with tag 'g'
    # The grab button (nxhz-bynyvtuepbt-1) has 'sys_click' tag but does it have 'jpug' and 'g'?

    # The crane direction buttons have both 'jpug' and single-letter tags plus 'sys_click'
    # They're gated by pressure plates (mbszrqqnqm), not zbhi
    # But the grab button... let me check

    # From the sprite listing:
    # nxhz-bynyvtuepbt-1 at (47,17) is listed as INVISIBLE at start
    # But after zbhi-g pickup, it becomes INTANGIBLE (from the experiment output above)
    # So zbhi-g DOES unlock the grab button!

    # The crane direction buttons are different - they're gated by pressure plates
    # They have letter tags a, b, e, h corresponding to the plates

    # So the sequence would be:
    # 1. Get to pressure plates to activate crane buttons
    # 2. Move crane to align with bridge
    # 3. Grab bridge
    # 4. Move bridge to create path

    # But how do we get to pressure plates? They're at (32-36, 56-60)
    # and we can't reach them from (32-36, 48-52) due to the gap at y=54-55

    # Wait - what about the f-cycle? The qgdz sprites are at (6, 54-58)
    # They don't cover x=32. But if we cycle them, maybe a different variant
    # extends further right?

    # Let's check qgdz sprite variants more carefully
    print("\n=== QGDZ VARIANT ANALYSIS ===")
    for s in game.current_level.get_sprites():
        if 'qgdz' in s.name:
            pixels = s.render()
            # Check width coverage
            nonblank_cols = set()
            for row in range(pixels.shape[0]):
                for col in range(pixels.shape[1]):
                    if pixels[row][col] >= 0:
                        nonblank_cols.add(col + s.x)
            if nonblank_cols:
                print(f"  {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name} "
                      f"nonblank_x: {min(nonblank_cols)}..{max(nonblank_cols)}")
            else:
                print(f"  {s.name} ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name} ALL BLANK")

    # Check if any qgdz variant reaches x=32
    # qgdz at (6, 54) with width 16 covers x=6..21 max
    # That's not enough to reach x=32

    # So the gap at y=54-55, x=32 can't be bridged by qgdz

    # What about the hhxv bridge itself? If we could move it to (32, 54) area...
    # But we need crane access first, which needs pressure plates, which needs the bridge
    # Chicken-and-egg problem!

    # UNLESS: we can reach the plates from the LEFT side
    # The kbqq at (6,44) (8,44) + kbqq-efzv-1 at (6,48) cover bottom-left
    # The hhxv bridge at (0,24) 20x20 covers (0..19, 24..43)
    # These connect: kbqq(8,20) -> hhxv(0,24)...(19,43) -> kbqq(6,44)(8,44) -> kbqq(6,48)
    # From (6,48) we have qgdz at (6,54)...
    # But that only goes to x=21 at most. Still far from plates at x=32

    # Wait, let me check the initial reach more carefully
    # Initial reach was only y=48..52 at x=18..28
    # But the left side (x=4..8, y=4..48) isn't reachable initially either
    # because there's a gap between x=18 and x=8 at y=48

    # kbqq-efzv-1 at (18,48) is 6x6, covers (18..23, 48..53)
    # kbqq-efzv-1 at (6,48) is 6x6, covers (6..11, 48..53)
    # Gap at x=12..17 at y=48..53 - no floor!

    # But the hhxv bridge at (0,24) covers (0..19, 24..43)
    # That includes x=12..17 at y=24..43 but NOT at y=48

    # So the left side IS reachable from the initial position?
    # Let's check: from (18,48), can player walk to (12,48)?
    # kbqq(18,48) covers x=18..23. kbqq(6,48) covers x=6..11. Gap at x=12..17.
    # No floor at x=12, y=48. Not reachable.

    # Unless we can use the itki teleport to get to (4,4)?
    # itkizfrq2 at (4,4) has tag 'c'. After c-click, player teleports between itkis.
    # But the c-click teleport mechanism: player must be ON an itki, and click c.
    # The npqswumrzz function finds the NEXT itki in the group.
    # What's the cycle order?

    print("\n\n=== ITKI TELEPORT CYCLE ===")
    # Let's trace what happens when we click c while on different itkis
    # After the c-click from (18,48), we landed at (32,52)
    # Now what if we c-click from (32,52)?

    state1 = save_game_state(game)

    # Walk to itkiupry2 at first (it's initially at 32,52)
    # But wait, which itki is where after a c-click?
    # Initially: itkiupry1 at (18,48), itkiupry2 at (32,52)
    # After c-click: names get swapped via variant cycling

    # Let me just trace the teleport chain
    print("  Tracing c-click teleport chain:")

    # Start at (18,48)
    parents0 = player_reachable_cells(game)
    if (18, 48) in parents0:
        moves = reconstruct_moves(parents0, (18, 48))
        for m in moves:
            env.step(m)

    for i in range(6):
        pos_before = (game.fdvakicpimr.x, game.fdvakicpimr.y)
        fd = env.step(GameAction.ACTION6, data=c_data)
        pos_after = (game.fdvakicpimr.x, game.fdvakicpimr.y)
        print(f"  c-click #{i+1}: {pos_before} -> {pos_after} (state={fd.state.name})")
        if fd.state.name == 'LOSE':
            print("  LOST! Player fell.")
            break
        # Check what itkis exist now
        for s in game.current_level.get_sprites():
            if 'itki' in s.tags and s.interaction != InteractionMode.REMOVED:
                on_player = (s.x == game.fdvakicpimr.x and s.y == game.fdvakicpimr.y)
                print(f"    {s.name} ({s.x},{s.y}) {'<-- PLAYER' if on_player else ''}")

    # Restore
    restore_game_state(game, state1)

    # The key question: does any teleport destination put us at (4,4)?
    # itkizfrq2 is at (4,4). If we teleport there, we'd be on the top-left kbqq cluster.
    # From there: kbqq(4..11, 4..23) -> hhxv bridge (0..19, 24..43) -> kbqq(6..11, 44..53)
    # Then we can reach qgdz at (6, 54-58)
    # But still can't reach plates at x=32-36

    # Actually wait - from (6,48) kbqq-efzv-1, the reach extends to (6..11, 48..53)
    # And from there, (6, 54) has qgdz-efzv-1 which is INTANGIBLE
    # Can we step onto it? INTANGIBLE means not collidable but IS walkable (support check)
    # Yes! INTANGIBLE sprites are checked by uxwpppoljm for support.
    # So qgdz at y=54-58 should be walkable if pixels are non-negative

    # BUT qgdz has gigzqgcfncq tag -> INTANGIBLE. And it has ordebgeg tag.
    # The pixel pattern determines walkability.
    # Let's check qgdz pixel values at key positions

    print("\n\n=== QGDZ PIXEL ANALYSIS ===")
    for s in game.current_level.get_sprites():
        if 'qgdz' in s.name and s.interaction != InteractionMode.REMOVED:
            pixels = s.render()
            print(f"\n{s.name} ({s.x},{s.y}) {s.width}x{s.height}:")
            for row in range(pixels.shape[0]):
                row_data = [int(pixels[row][col]) for col in range(pixels.shape[1])]
                print(f"  row {row}: {row_data}")

if __name__ == "__main__":
    main()
