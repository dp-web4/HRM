"""Explore game r11l-aa269680 - understand mechanics, sprites, and actions."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade, OperationMode
from arcengine import GameAction
import numpy as np

arcade = Arcade(operation_mode=OperationMode.OFFLINE)
env = arcade.make("r11l-aa269680")

print("=" * 80)
print("GAME: r11l-aa269680")
print("=" * 80)

game = env._game
print(f"\nAvailable actions: {game._available_actions}  (6 = ACTION6 = click)")
print(f"Max actions per level: {game._max_actions}")
print(f"Number of levels: {len(game._levels)}")

fd = env.reset()
game = env._game
frame = np.array(fd.frame)
print(f"Frame: {frame.shape} (1,H,W) palette-indexed")

# ===== LEVEL SURVEY =====
for level_idx in range(len(game._levels)):
    print(f"\n{'='*80}")
    print(f"LEVEL {level_idx + 1}")
    print(f"{'='*80}")

    env.reset()
    game = env._game
    if level_idx > 0:
        game.set_level(level_idx)
        game._action_count = 0

    level = game.current_level
    print(f"Grid size: {level.grid_size}")

    all_sprites = level.get_sprites()
    print(f"Total sprites: {len(all_sprites)}")

    # Categorize
    cats = {}
    for s in all_sprites:
        colors = sorted(set(int(c) for c in np.unique(s.pixels) if c >= 0))
        tags = getattr(s, '_tags', [])
        # Determine category
        for prefix in ['bdkazLeg', 'bdkaz', 'kzeze', 'bvzgd', 'qtwnv', 'xigcb', 'txxvz', 'mpjrv']:
            if s.name.startswith(prefix):
                cat = prefix
                break
        else:
            cat = 'other'
        if cat not in cats:
            cats[cat] = []
        cats[cat].append((s.name, s.x, s.y, s.width, s.height, tags, colors))

    for cat in ['bdkaz', 'bdkazLeg', 'kzeze', 'bvzgd', 'qtwnv', 'xigcb', 'txxvz', 'other']:
        if cat not in cats:
            continue
        items = cats[cat]
        labels = {'bdkaz': 'body', 'bdkazLeg': 'leg/click', 'kzeze': 'target', 'bvzgd': 'terrain',
                  'qtwnv': 'obstacle', 'xigcb': 'paint', 'txxvz': 'decor'}
        print(f"\n  {cat} ({labels.get(cat,'?')}) [{len(items)}]:")
        for name, x, y, w, h, tags, colors in items:
            print(f"    {name}: ({x},{y}) {w}x{h} colors={colors}")

    # Creature groups
    print(f"\n  Creature Groups:")
    for key, data in game.brdck.items():
        body = data['kignw']
        legs = data['mdpcc']
        target = data['xwdrv']
        b = f"({body.x},{body.y})" if body else "None"
        t = f"({target.x},{target.y})" if target else "None"
        print(f"    {key}: body={b} target={t} legs={len(legs)}")
        if body and target:
            bcx = body.x + body.width // 2
            bcy = body.y + body.height // 2
            tcx = target.x + target.width // 2
            tcy = target.y + target.height // 2
            print(f"      body_ctr=({bcx},{bcy}) target_ctr=({tcx},{tcy}) delta=({tcx-bcx},{tcy-bcy})")

    if game.nxahg:
        print(f"  Paint sprites to absorb: {[f'{s.name}@({s.x},{s.y})' for s in game.nxahg]}")
    if game.fwqwj:
        print(f"  Yukft paint register: {game.fwqwj}")

# ===== ACTION TESTS =====
print(f"\n{'='*80}")
print("ACTION TESTS")
print(f"{'='*80}")

# Test on Level 1
env.reset()
game = env._game

print(f"\nLevel 1 Initial:")
sel = game.mjdkn
print(f"  Selected leg: {sel.name}@({sel.x},{sel.y})")
print(f"  All clickable legs: {[f'{l.name}@({l.x},{l.y})' for l in game.ftmaz]}")

# Test 1: Click on each leg
print(f"\n--- Test: Selecting legs ---")
for leg in game.ftmaz:
    cx, cy = leg.x + leg.width//2, leg.y + leg.height//2
    old = game.mjdkn.name
    env.step(GameAction.ACTION6, data={"x": cx, "y": cy})
    new = game.mjdkn.name
    print(f"  Click ({cx},{cy}) on '{leg.name}': {old} -> {new}")

# Test 2: Move a leg
print(f"\n--- Test: Moving a leg ---")
env.reset()
game = env._game

sel = game.mjdkn
print(f"  Selected: {sel.name}@({sel.x},{sel.y})")

# Find its group
for key, data in game.brdck.items():
    if sel in data['mdpcc']:
        body = data['kignw']
        target = data['xwdrv']
        all_legs = data['mdpcc']
        break

tcx = target.x + target.width//2
tcy = target.y + target.height//2
bcx = body.x + body.width//2
bcy = body.y + body.height//2
dx, dy = tcx - bcx, tcy - bcy
print(f"  Delta to target: ({dx},{dy})")

# Move selected leg
new_x = sel.x + dx + sel.width//2
new_y = sel.y + dy + sel.height//2
print(f"  Moving to ({new_x},{new_y})...")
env.step(GameAction.ACTION6, data={"x": new_x, "y": new_y})

# Animate
steps = 0
while game.bmtib and steps < 30:
    env.step(GameAction.ACTION6, data={"x": 0, "y": 0})
    steps += 1
print(f"  Animation: {steps} steps, leg now @({sel.x},{sel.y})")
print(f"  Body now @({body.x},{body.y}) center=({body.x+body.width//2},{body.y+body.height//2})")

# ===== SOLVE LEVEL 1 =====
print(f"\n{'='*80}")
print("SOLVE LEVEL 1")
print(f"{'='*80}")

env.reset()
game = env._game

for key, data in game.brdck.items():
    body = data['kignw']
    legs = data['mdpcc']
    target = data['xwdrv']
    if not (body and target):
        continue

    tcx = target.x + target.width//2
    tcy = target.y + target.height//2
    bcx = body.x + body.width//2
    bcy = body.y + body.height//2
    dx, dy = tcx - bcx, tcy - bcy
    print(f"\n{key}: body@({bcx},{bcy}) -> target@({tcx},{tcy}), delta=({dx},{dy})")

    for i, leg in enumerate(legs):
        # Select
        cx, cy = leg.x + leg.width//2, leg.y + leg.height//2
        env.step(GameAction.ACTION6, data={"x": cx, "y": cy})

        # Move
        new_x = leg.x + dx + leg.width//2
        new_y = leg.y + dy + leg.height//2
        env.step(GameAction.ACTION6, data={"x": new_x, "y": new_y})

        # Animate
        steps = 0
        while game.bmtib and steps < 30:
            fd = env.step(GameAction.ACTION6, data={"x": 0, "y": 0})
            steps += 1
        print(f"  leg {i}: moved to ({leg.x},{leg.y}), {steps} anim steps")

    print(f"  Body: ({body.x},{body.y}) center=({body.x+body.width//2},{body.y+body.height//2})")
    print(f"  Target: ({target.x},{target.y}) center=({tcx},{tcy})")
    print(f"  Colliding: {body.collides_with(target)}")

print(f"\nGame state: level={game._current_level_index}, actions={game._action_count}, winning={game.winning}")

# Check if we won
fd = env.step(GameAction.ACTION6, data={"x": 0, "y": 0})
print(f"State after step: {fd.state}, levels_completed={fd.levels_completed}")

# Step a few more for win animation
for _ in range(30):
    fd = env.step(GameAction.ACTION6, data={"x": 0, "y": 0})
    if fd.state != "NOT_FINISHED" or fd.levels_completed > 0:
        print(f"  State: {fd.state}, levels_completed={fd.levels_completed}")
        break

print(f"\nFinal: level={game._current_level_index}, actions={game._action_count}")

# ===== SUMMARY =====
print(f"\n{'='*80}")
print("MECHANICS SUMMARY")
print(f"{'='*80}")
print("""
GAME TYPE: Click-based creature movement puzzle
  - Only ACTION6 (click at x,y) is available
  - 64x64 grid, palette-indexed frames

SPRITE TYPES:
  bdkaz-*      Body - 5x5 colored diamond, auto-positioned at centroid of legs
  bdkazLeg-*   Leg  - 5x5 diamond, clickable (sys_click tag), selectable/movable
  kzeze-*      Target - where the body must end up
  bvzgd-*      Terrain - walkable area boundary
  qtwnv-*      Obstacle - body collision penalty (5 = game over)
  xigcb-*      Paint - color sprites absorbed by black "yukft" creatures
  txxvz-*      Decoration

INTERACTION:
  1. Click ON a leg -> selects it (highlight changes)
  2. Click EMPTY walkable space -> moves selected leg there (instant, gfwuu=1)
  3. Body auto-repositions to centroid of its legs after each move
  4. Can't move leg to position that collides with terrain boundary
  5. If body ends up on obstacle after move -> penalty counter + flash

WIN: All bodies overlap their targets (+ color match for yukft)
LOSE: Action count >= 60 OR 5 obstacle collisions

LEVELS:         Creatures  Legs  Special              Baseline
  1             1          2     simple               7
  2             2          5     obstacle              28
  3             2          6     obstacle              30
  4             3          7     obstacle+decor         20
  5             2(yukft)   5     paint absorption       37
  6             2(yukft)   5     many paints            45

SOLVING STRATEGY:
  Simple levels (1-4):
    - Compute delta = target_center - body_center
    - Move each leg by that delta
    - Avoid obstacles (route around qtwnv)

  Yukft levels (5-6):
    - Black bodies must absorb paint sprites to match target colors
    - Route body through correct paint sprites
    - Then position on target
    - Paint absorption happens when body overlaps xigcb sprite
""")
