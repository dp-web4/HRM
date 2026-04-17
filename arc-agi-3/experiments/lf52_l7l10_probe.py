#!/usr/bin/env python3
"""
Probe L7 and L10: map initial states, verify jump conditions.
"""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

def dump_grid(eq, label, rows=12, cols=25):
    """Dump grid contents for a level."""
    grid = eq.hncnfaqaddg
    print(f"\n=== {label} ===")
    print(f"  Level: {eq.whtqurkphir}, Offset: {grid.cdpcbbnfdp}")

    pieces = {}
    blocks = set()
    walls = set()
    pegs = set()
    walkable = set()

    for y in range(rows):
        for x in range(cols):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            if not names:
                continue
            for n in names:
                if n == 'fozwvlovdui':
                    pieces[(x,y)] = 'N'
                elif n == 'fozwvlovdui_red':
                    pieces[(x,y)] = 'R'
                elif n == 'fozwvlovdui_blue':
                    pieces[(x,y)] = 'B'
                elif n == 'hupkpseyuim2':
                    blocks.add((x,y))
                elif 'kraubslpehi' in n:
                    walls.add((x,y))
                elif n == 'dgxfozncuiz':
                    pegs.add((x,y))
                elif n == 'hupkpseyuim':
                    walkable.add((x,y))

    print(f"  Pieces: {pieces}")
    print(f"  Blocks: {sorted(blocks)}")
    print(f"  Walls ({len(walls)}): {sorted(walls)[:20]}{'...' if len(walls) > 20 else ''}")
    print(f"  Pegs: {sorted(pegs)}")
    print(f"  Walkable ({len(walkable)})")

    # Check key cells
    print("\n  Key cell contents:")
    for y in range(rows):
        for x in range(cols):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            if len(names) >= 2 or any(n in ['fozwvlovdui', 'fozwvlovdui_red', 'fozwvlovdui_blue'] for n in names):
                print(f"    ({x},{y}): {names}")

    return {'pieces': pieces, 'blocks': blocks, 'walls': walls, 'pegs': pegs, 'walkable': walkable}

def check_jump(eq, src, direction):
    """Check if a jump is valid using engine's qikmikecdf."""
    valid = eq.qikmikecdf(src, direction)
    dx, dy = direction
    mx, my = src[0]+dx, src[1]+dy
    lx, ly = src[0]+2*dx, src[1]+2*dy
    grid = eq.hncnfaqaddg
    mid_objs = [o.name for o in grid.ijpoqzvnjt(mx, my)]
    land_objs = [o.name for o in grid.ijpoqzvnjt(lx, ly)]
    print(f"  Jump {src} dir {direction}: valid={valid}")
    print(f"    middle ({mx},{my}): {mid_objs}")
    print(f"    landing ({lx},{ly}): {land_objs}")
    return valid

# Setup
arc = Arcade()
env = arc.make('lf52')
fd = env.reset()
game = env._game

# === L7 ===
print("=" * 60)
print("L7 PROBE")
print("=" * 60)
game.set_level(6)  # 0-indexed, L7 = internal 6
eq = game.ikhhdzfmarl
s7 = dump_grid(eq, "L7 initial", rows=10, cols=25)

# Check red piece jump validity
print("\nRed@(6,1) jump checks:")
for d in [(0,-1), (0,1), (-1,0), (1,0)]:
    check_jump(eq, (6,1), d)

# Check (6,3) contents in detail
print("\nCell (6,3) full detail:")
objs = eq.hncnfaqaddg.ijpoqzvnjt(6, 3)
for o in objs:
    print(f"  name={o.name}")

# After pushes, check if (6,3) becomes a valid landing
# Try the push sequence from the investigation: L, L, U, U, R, R, R
print("\n--- Trying push sequence to move block to (6,3) ---")
push_map = {
    'L': GameAction.ACTION3, 'R': GameAction.ACTION4,
    'U': GameAction.ACTION1, 'D': GameAction.ACTION2
}
pushes = ['L', 'L', 'U', 'U', 'R', 'R', 'R']
for p in pushes:
    env.step(push_map[p])
    # Check block positions
    blocks_now = set()
    for y in range(10):
        for x in range(25):
            objs = eq.hncnfaqaddg.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            if 'hupkpseyuim2' in names:
                blocks_now.add((x,y))
    print(f"  After PUSH {p}: blocks={sorted(blocks_now)}")

# Now check (6,3) contents
print("\nAfter pushes, cell (6,3):")
objs = eq.hncnfaqaddg.ijpoqzvnjt(6, 3)
for o in objs:
    print(f"  name={o.name}")
print(f"  obj count: {len(objs)}")

# Check if red can now jump down
print("\nRed@(6,1) jump after pushes:")
check_jump(eq, (6,1), (0,1))

# Check movable count and win condition
print(f"\nWin count (ddaguepwkt): {eq.ddaguepwkt}")
print(f"Level: {eq.whtqurkphir}")
print(f"Step count: {eq.asqvqzpfdi}")


# === L10 ===
print("\n" + "=" * 60)
print("L10 PROBE")
print("=" * 60)
# Reset and go to L10
env2 = arc.make('lf52')
fd2 = env2.reset()
game2 = env2._game
game2.set_level(9)  # L10 = internal 9
eq2 = game2.ikhhdzfmarl
s10 = dump_grid(eq2, "L10 initial", rows=14, cols=10)

# Check N pieces
print("\nN@(4,0) jump checks (should all be invalid initially):")
for d in [(0,-1), (0,1), (-1,0), (1,0)]:
    check_jump(eq2, (4,0), d)

print("\nN@(6,9) jump checks:")
for d in [(0,-1), (0,1), (-1,0), (1,0)]:
    check_jump(eq2, (6,9), d)

# Check the 7-blocks at x=8
print("\nx=8 column contents:")
for y in range(14):
    objs = eq2.hncnfaqaddg.ijpoqzvnjt(8, y)
    names = [o.name for o in objs]
    if names:
        print(f"  (8,{y}): {names}")

# Check row 6 walls
print("\nRow 6 wall structure:")
for x in range(9):
    objs = eq2.hncnfaqaddg.ijpoqzvnjt(x, 6)
    names = [o.name for o in objs]
    if names:
        print(f"  ({x},6): {names}")

# Row 1 wall structure
print("\nRow 1 wall structure:")
for x in range(9):
    objs = eq2.hncnfaqaddg.ijpoqzvnjt(x, 1)
    names = [o.name for o in objs]
    if names:
        print(f"  ({x},1): {names}")

# Try pushing UP to move 7-blocks
print("\n--- Trying UP pushes ---")
for i in range(8):
    env2.step(GameAction.ACTION1)  # UP
    blocks_now = []
    for y in range(14):
        for x in range(10):
            objs = eq2.hncnfaqaddg.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            if 'hupkpseyuim2' in names:
                blues = [n for n in names if 'fozwvlovdui_blue' in n]
                blocks_now.append(((x,y), 'B' if blues else ''))
    print(f"  After UP #{i+1}: blocks={blocks_now}")

# Now try LEFT pushes to move blocks in row 1
print("\n--- Trying LEFT pushes ---")
for i in range(8):
    env2.step(GameAction.ACTION3)  # LEFT
    blocks_row1 = []
    for x in range(10):
        objs = eq2.hncnfaqaddg.ijpoqzvnjt(x, 1)
        names = [o.name for o in objs]
        if 'hupkpseyuim2' in names:
            blues = [n for n in names if 'fozwvlovdui_blue' in n]
            blocks_row1.append(((x,1), 'B' if blues else ''))
    print(f"  After LEFT #{i+1}: row1 blocks={blocks_row1}")

print("\nDone.")
