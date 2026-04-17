#!/usr/bin/env python3
"""
L10: Build solution trace. Map the grid and verify the edge routing path.

From the investigation:
- 2 N, 10 B. Win = reduce N from 2 to 1.
- N@(4,0), N@(6,9)
- 5 "7" blocks (blue+block+wall) at x=8, rows 9-13
- Strategy: transport 7-blocks UP to row 1, LEFT across wall channel,
  position blue near N@(4,0), N jumps through blue stepping stones

Let me probe this empirically.
"""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}

arc = Arcade()
env = arc.make('lf52')
fd = env.reset()
game = env._game
game.set_level(9)  # L10
eq = game.ikhhdzfmarl

def cell(x, y):
    objs = eq.hncnfaqaddg.ijpoqzvnjt(x, y)
    return [o.name for o in objs]

def get_pieces():
    pieces = {}
    for y in range(15):
        for x in range(10):
            objs = eq.hncnfaqaddg.ijpoqzvnjt(x, y)
            for o in objs:
                if o.name == 'fozwvlovdui': pieces[(x,y)] = 'N'
                elif o.name == 'fozwvlovdui_blue': pieces[(x,y)] = 'B'
    return pieces

def get_blocks():
    blocks = set()
    for y in range(15):
        for x in range(10):
            objs = eq.hncnfaqaddg.ijpoqzvnjt(x, y)
            if any(o.name == 'hupkpseyuim2' for o in objs):
                blocks.add((x,y))
    return blocks

# Full grid map
print("=== L10 Full Grid Map ===")
for y in range(15):
    row_str = f"y={y:2d}: "
    for x in range(10):
        objs = eq.hncnfaqaddg.ijpoqzvnjt(x, y)
        names = [o.name for o in objs]
        if not names: ch = ' '
        elif any('fozwvlovdui' == n for n in names): ch = 'N'
        elif any('fozwvlovdui_blue' == n for n in names):
            if any('hupkpseyuim2' in n for n in names): ch = '7'
            else: ch = 'B'
        elif any('hupkpseyuim2' in n for n in names):
            if any('kraubslpehi' in n for n in names): ch = ';'
            else: ch = 'b'
        elif any('kraubslpehi' in n for n in names): ch = '#'
        elif any('dgxfozncuiz' in n for n in names): ch = 'o'
        elif 'hupkpseyuim' in names: ch = '.'
        else: ch = '?'
        row_str += ch
    print(row_str)

# Pieces and blocks
print(f"\nPieces: {get_pieces()}")
print(f"Blocks: {sorted(get_blocks())}")
print(f"Offset: {eq.hncnfaqaddg.cdpcbbnfdp}")

# All valid jumps
print(f"\n=== All valid jumps (initial state) ===")
for y in range(15):
    for x in range(10):
        for d in DIRS:
            if eq.qikmikecdf((x,y), d):
                dx, dy = d
                print(f"  ({x},{y}) {DIR_NAMES[d]} -> ({x+2*dx},{y+2*dy})")

# Check wall structure for edge routing
print(f"\n=== x=0 column ===")
for y in range(15):
    c = cell(0, y)
    if c: print(f"  (0,{y}): {c}")

print(f"\n=== x=8 column ===")
for y in range(15):
    c = cell(8, y)
    if c: print(f"  (8,{y}): {c}")

print(f"\n=== Row 1 wall channel ===")
for x in range(10):
    c = cell(x, 1)
    if c: print(f"  ({x},1): {c}")

# Try UP pushes to move 7-blocks
print(f"\n=== Trying UP pushes ===")
for i in range(9):
    env.step(GameAction.ACTION1)
    blocks = get_blocks()
    pieces = get_pieces()
    # Show blue positions
    blues = [(pos, n) for pos, n in pieces.items() if n == 'B']
    # Show blocks at x=8
    col8_blocks = sorted([(x,y) for x,y in blocks if x == 8])
    # Show blocks at row 1
    row1_blocks = sorted([(x,y) for x,y in blocks if y == 1])
    print(f"  UP #{i+1}: x8_blocks={col8_blocks}, row1_blocks={row1_blocks}, step={eq.asqvqzpfdi}")

# Now try LEFT pushes
print(f"\n=== Trying LEFT pushes ===")
for i in range(8):
    env.step(GameAction.ACTION3)
    blocks = get_blocks()
    row1_blocks = sorted([(x,y) for x,y in blocks if y == 1])
    pieces = get_pieces()
    blues_r1 = [(pos, n) for pos, n in pieces.items() if n == 'B' and pos[1] == 1]
    print(f"  LEFT #{i+1}: row1_blocks={row1_blocks}, blues_row1={blues_r1}")

# Check where N@(4,0) can jump now
print(f"\n=== N@(4,0) jump check after pushes ===")
for d in DIRS:
    v = eq.qikmikecdf((4,0), d)
    dx, dy = d
    mx, my = 4+dx, 0+dy
    lx, ly = 4+2*dx, 0+2*dy
    if v:
        print(f"  {DIR_NAMES[d]}: VALID (mid=({mx},{my})={cell(mx,my)}, land=({lx},{ly})={cell(lx,ly)})")
    else:
        mid = cell(mx, my) if 0 <= mx < 10 and 0 <= my < 15 else 'OOB'
        land = cell(lx, ly) if 0 <= lx < 10 and 0 <= ly < 15 else 'OOB'
        print(f"  {DIR_NAMES[d]}: invalid (mid=({mx},{my})={mid}, land=({lx},{ly})={land})")

# Show current state
print(f"\nPieces: {get_pieces()}")
print(f"Blocks: {sorted(get_blocks())}")
print(f"Steps: {eq.asqvqzpfdi}")

# Check all valid jumps after pushes
print(f"\n=== All valid jumps after pushes ===")
for y in range(15):
    for x in range(10):
        for d in DIRS:
            if eq.qikmikecdf((x,y), d):
                dx, dy = d
                print(f"  ({x},{y}) {DIR_NAMES[d]} -> ({x+2*dx},{y+2*dy})")

print("\nDone.")
