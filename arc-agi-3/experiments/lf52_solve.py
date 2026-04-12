#!/usr/bin/env python3
"""
lf52 World-Model Solver — Peg solitaire with colored pieces.

WORLD MODEL (decoded from obfuscated lf52.py):
==============================================
Grid: cells of 6x6 pixels, grid offset varies per level
Scene: 64x64 pixels (same as display)
Actions: 1=up, 2=down, 3=left, 4=right, 6=click, 7=undo

OBJECTS:
- fozwvlovdui: game pieces (pegs) — can be neutral, red, or blue
- hupkpseyuim: walkable ground tiles
- hupkpseyuim2: pushable objects
- kraubslpehi: walls
- lgbyiaitpdi: arrow indicators (shown after selecting a piece)
- cwyrzsciwms: completion button
- csrvckunbev: selection cursor

MECHANICS — Peg Solitaire:
- Click a fozwvlovdui piece to select it
- Arrow indicators appear showing valid jumps
- Click an arrow to execute the jump:
  - Piece jumps over adjacent piece to landing 2 cells away
  - Jumped-over piece is removed
  - Same-colored pieces can jump over same-color or neutral
  - Different colored pieces cannot jump over each other
- Win when piece count reaches target (1 for most levels, 2 for levels 6-7)
- Some levels have camera scrolling (grid moves)
- Step limit: L1=64, L2-5=320 (64*5), L6+=640 (64*10)

STRATEGY:
DFS to solve peg solitaire, then execute jumps via click interactions.
"""

import os, sys, time
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from PIL import Image

VISUAL_DIR = "shared-context/arc-agi-3/visual-memory/lf52"
os.makedirs(VISUAL_DIR, exist_ok=True)

PALETTE = {
    0:(255,255,255), 1:(220,220,220), 2:(255,0,0), 3:(128,128,128),
    4:(255,255,0), 5:(100,100,100), 6:(255,0,255), 7:(255,192,203),
    8:(200,0,0), 9:(128,0,0), 10:(0,0,255), 11:(135,206,250),
    12:(0,0,200), 13:(255,165,0), 14:(0,255,0), 15:(128,0,128),
}

def save_frame(frame_data, path):
    frame = np.array(frame_data[0])
    h, w = frame.shape
    s = 8
    img = Image.new('RGB', (w*s, h*s))
    pix = img.load()
    for y in range(h):
        for x in range(w):
            c = PALETTE.get(int(frame[y,x]), (0,0,0))
            for dy in range(s):
                for dx in range(s):
                    pix[x*s+dx, y*s+dy] = c
    img.save(path)


def extract_grid(eq):
    """Extract grid state."""
    grid = eq.hncnfaqaddg
    cells = set()
    pieces = {}  # (x,y) -> 'neutral'|'red'|'blue'

    for y in range(30):
        for x in range(30):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            if 'hupkpseyuim' in names or 'hupkpseyuim2' in names:
                cells.add((x, y))
            for name in names:
                if name == 'fozwvlovdui':
                    pieces[(x, y)] = 'neutral'
                elif name == 'fozwvlovdui_red':
                    pieces[(x, y)] = 'red'
                elif name == 'fozwvlovdui_blue':
                    pieces[(x, y)] = 'blue'

    return cells, pieces


def solve_solitaire(cells, pieces, target_count=1):
    """DFS solver for peg solitaire with colored pieces."""
    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    # State: frozenset of (x, y, color)
    init_state = frozenset((x, y, c) for (x, y), c in pieces.items())

    if len(init_state) <= target_count:
        return []

    def can_jump(jumper_color, middle_color):
        """Check if jumper can jump over middle piece."""
        if jumper_color == middle_color:
            return True
        if middle_color == 'neutral' or jumper_color == 'neutral':
            return True
        return False  # red can't jump blue, blue can't jump red

    def dfs(state, moves, count):
        if count <= target_count:
            return moves

        state_dict = {(x, y): c for x, y, c in state}
        pos_set = set((x, y) for x, y, c in state)

        for x, y, color in sorted(state):
            for dx, dy in directions:
                mx, my = x + dx, y + dy  # middle
                lx, ly = x + 2*dx, y + 2*dy  # landing

                if (mx, my) not in pos_set:
                    continue
                if (lx, ly) in pos_set:
                    continue
                if (lx, ly) not in cells:
                    continue

                mid_color = state_dict[(mx, my)]
                if not can_jump(color, mid_color):
                    continue

                # Valid jump
                new_state = (state - {(x, y, color), (mx, my, mid_color)}) | {(lx, ly, color)}
                result = dfs(frozenset(new_state), moves + [((x,y), (lx,ly))], count - 1)
                if result is not None:
                    return result

        return None

    return dfs(init_state, [], len(pieces))


def execute_jumps(env, game, jumps, level_idx):
    """Execute jump sequence on the game engine."""
    eq = game.ikhhdzfmarl
    grid = eq.hncnfaqaddg
    all_moves = []

    for i, (src, dst) in enumerate(jumps):
        sx, sy = src
        dx, dy = dst

        off = grid.cdpcbbnfdp

        # Click on source piece
        px = sx * 6 + off[0] + 3
        py = sy * 6 + off[1] + 3

        fd = env.step(GameAction.ACTION6, data={'x': px, 'y': py})
        all_moves.append((GameAction.ACTION6, {'x': px, 'y': py}))

        if fd.levels_completed > level_idx or fd.state.name == 'WIN':
            return fd, all_moves

        # Click on arrow indicator
        half_dx = (dx - sx) // 2
        half_dy = (dy - sy) // 2
        ax = sx * 6 + off[0] + half_dx * 12 + 1
        ay = sy * 6 + off[1] + half_dy * 12 + 1

        fd = env.step(GameAction.ACTION6, data={'x': ax, 'y': ay})
        all_moves.append((GameAction.ACTION6, {'x': ax, 'y': ay}))

        if fd.levels_completed > level_idx or fd.state.name == 'WIN':
            return fd, all_moves

        # Check if completion button appeared
        if eq.zvcnglshzcx:
            # Click completion button (bottom-left, x<16 y>48)
            fd = env.step(GameAction.ACTION6, data={'x': 8, 'y': 56})
            all_moves.append((GameAction.ACTION6, {'x': 8, 'y': 56}))
            if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                return fd, all_moves

    # Advance animation
    for _ in range(50):
        fd = env.step(GameAction.ACTION1)
        all_moves.append(GameAction.ACTION1)
        if fd.levels_completed > level_idx or fd.state.name != 'NOT_FINISHED':
            break

    return fd, all_moves


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    obs = env.reset()
    game = env._game

    print(f"LF52: {obs.win_levels} levels")
    save_frame(obs.frame, f"{VISUAL_DIR}/initial.png")

    levels_solved = 0

    for level in range(obs.win_levels):
        eq = game.ikhhdzfmarl
        print(f"\n=== LF52 Level {level+1} (internal: {eq.whtqurkphir}) ===")

        cells, pieces = extract_grid(eq)

        # Print grid
        colors = {'neutral': 'O', 'red': 'R', 'blue': 'B'}
        max_x = max(x for x, y in cells) if cells else 0
        max_y = max(y for x, y in cells) if cells else 0
        print(f"  Grid ({max_x+1}x{max_y+1}):")
        for y in range(max_y+1):
            row = '    '
            for x in range(max_x+1):
                if (x, y) in pieces:
                    row += colors[pieces[(x,y)]]
                elif (x, y) in cells:
                    row += '.'
                else:
                    row += ' '
            print(row)

        print(f"  Pieces: {len(pieces)}")
        print(f"  Grid offset: {eq.hncnfaqaddg.cdpcbbnfdp}")

        # Target count
        target = 2 if eq.whtqurkphir in [6, 7] else 1
        print(f"  Target: {target} piece(s) remaining")

        save_frame(env.observation_space.frame, f"{VISUAL_DIR}/L{level+1}_start.png")

        t0 = time.time()
        jumps = solve_solitaire(cells, pieces, target)
        elapsed = time.time() - t0

        if jumps is None:
            print(f"  No solution found ({elapsed:.1f}s)")
            print(f"  FAILED on level {level+1}")
            break

        print(f"  Solution: {len(jumps)} jumps ({elapsed:.1f}s)")
        for j in jumps:
            print(f"    {j[0]} -> {j[1]}")

        fd, moves = execute_jumps(env, game, jumps, level)

        if fd.levels_completed > level:
            levels_solved = fd.levels_completed
            print(f"  Level {level+1} SOLVED! completed={fd.levels_completed}")
            save_frame(fd.frame, f"{VISUAL_DIR}/L{level+1}_solved.png")
        else:
            print(f"  Level {level+1} execution failed. state={fd.state.name}, completed={fd.levels_completed}")
            print(f"  Won={eq.iajuzrgttrv}, evxflhofing={eq.evxflhofing}")
            save_frame(fd.frame, f"{VISUAL_DIR}/L{level+1}_failed.png")
            break

        if fd.state.name == 'WIN':
            print("  GAME WON!")
            break

    print(f"\nFinal: {levels_solved}/{obs.win_levels} levels solved")


if __name__ == "__main__":
    main()
