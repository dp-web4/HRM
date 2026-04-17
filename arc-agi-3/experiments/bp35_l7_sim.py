#!/usr/bin/env python3
"""Custom simulator for bp35 L7. Much faster than engine replay.

Cell types:
  '#' = wall (xcjjwqfzjfe) — blocks move, blocks fall
  'g' = gravity flip (lrpkmzabbfa) — clickable (anywhere), flips gravity + consumed + player falls
  '+' = gem (fjlzdjxhant) — step-onto wins, fall-onto wins
  'u' = spike u (hzusueifitk) — move-into bounces (no move), fall-into kills at prev cell
  'v' = spike v (ubhhgljbnpu) — same as u
  'x' = spike qclfkhjnaac — clickable to destroy; else like u/v (but L7 has none)
  '1' = yuuqpmlxorv — move-into bounces, fall blocks; clickable remote converts to '2'
  '2' = oonshderxef — move-into OK (pass), fall passes through; clickable remote converts to '1'
  '.' = empty
  'y' = etlsaqqtjvn — spreads on click (L7 has none)

Fall: starting from cell (x, y0) going dy direction:
  iterate cell = (x, y0 + dy), y0 + 2*dy, ...
  continue while cell is empty or '2' or (aknlbboysnc bg — ignore)
  on gem: land on gem, WIN
  on u/v spike: STOP at prev cell (udsicoryza), LOSE
  on anything else (wall, 1, g, x): stop at prev cell, no consequence

Move LEFT/RIGHT:
  target = (px +/- 1, py)
  if target has gem: WIN
  if target empty or '2': do fall from target (gravity direction)
  else: no move (bounce)

Click:
  if click target is 'g': flip gravity, REMOVE g tile, perform pbsitubcfd(click_target, override=True) — player falls from (px, py + dy) in new grav direction (with override check: reject if cell is wall/g/1/x/y/spike)
  if click target is '1': CONVERT to '2'. Then non-override pbsitubcfd: if click target == (px, py + dy), fall from (px, py + dy).
  if click target is '2': CONVERT to '1'. Then same as above.
  Otherwise: no-op.
"""
import sys, os, time
from collections import deque
from copy import deepcopy

# L7 grid: 11 cols x 28 rows (y=4..31).
# Build from the source grid7 reversed.
# Player spawn = (3, 19), gravity UP.

# Read from source and materialize
GRID_STR = [
    # source rows 0 (y=31) to 31 (y=0), but we only care about y=4..31
    "ooooooooooo",   # y=31 src_row=0
    "ooooooooooo",   # y=30 src_row=1
    "ooooooooooo",   # y=29 src_row=2
    "ooooooooooo",   # y=28 src_row=3
    "go   oooooo",   # y=27 src_row=4
    "go   oo   o",   # y=26 src_row=5
    "go + oo o o",   # y=25 src_row=6  (GEM at col 3)
    "go      u o",   # y=24 src_row=7  (u at col 8)
    "go        o",   # y=23 src_row=8
    "gooooooo  o",   # y=22 src_row=9
    "go    2o  o",   # y=21 src_row=10
    "go        o",   # y=20 src_row=11
    "go n  2o  o",   # y=19 src_row=12  (player spawn at col 3)
    "gooooooo  o",   # y=18 src_row=13
    "go u222u  o",   # y=17 src_row=14
    "go  222   o",   # y=16 src_row=15
    "go  222   o",   # y=15 src_row=16
    "go o222 o o",   # y=14 src_row=17
    "go o222 o o",   # y=13 src_row=18
    "go oooooo o",   # y=12 src_row=19
    "go 2 u uo o",   # y=11 src_row=20
    "go 2 1 2o o",   # y=10 src_row=21
    "go 22 12o o",   # y=9 src_row=22
    "go 2v v2o o",   # y=8 src_row=23
    "goooooo2  o",   # y=7 src_row=24
    "goooooo2  o",   # y=6 src_row=25
    "goooooo2o o",   # y=5 src_row=26
    "ooooooo2ovo",   # y=4 src_row=27
    "ooooooooooo",   # y=3 src_row=28 (above gameplay)
    "ooooooooooo",   # y=2
    "ooooooooooo",   # y=1
    "ooooooooooo",   # y=0
]

# Char mapping
CHAR2CELL = {
    ' ': '.',  # empty
    'o': '#',  # wall
    'g': 'g',
    'n': '.',  # player spawn; cell empty
    '+': '+',
    'x': 'x',
    'v': 'v',
    'u': 'u',
    '1': '1',
    '2': '2',
    'y': 'y',
    'm': '.',  # bg decorations — treat as empty
    'w': '.',
}


def build_grid():
    # y=31 is src_row=0; y=4 is src_row=27; but we need y=0..31 access.
    # Indexing: grid[y][x]
    grid = [['#']*11 for _ in range(32)]
    for src_idx, s in enumerate(GRID_STR):
        y = 31 - src_idx
        if y < 0 or y >= 32: continue
        for x, ch in enumerate(s):
            grid[y][x] = CHAR2CELL.get(ch, '?')
    return grid


# Gravity: True = UP (dy=-1), False = DOWN (dy=1)
def fsvnqdbzrp(grid, start_x, start_y, grav_up):
    """Fall simulation. Returns (landed_on_gem, landed_on_spike, final_x, final_y)."""
    dy = -1 if grav_up else 1
    udsicoryza = (start_x, start_y)
    nx, ny = start_x, start_y + dy
    while 0 <= ny < 32:
        cell = grid[ny][nx]
        if cell == '.' or cell == '2':
            udsicoryza = (nx, ny)
            ny += dy
            continue
        if cell == '+':
            return (True, False, nx, ny)
        if cell == 'u' or cell == 'v':
            return (False, True, udsicoryza[0], udsicoryza[1])
        # wall, g, 1, x, y: stop at udsicoryza
        return (False, False, udsicoryza[0], udsicoryza[1])
    return (False, False, udsicoryza[0], udsicoryza[1])


def pbsitubcfd(grid, px, py, grav_up, override):
    """Player falls from current pos in grav direction. Returns (new_x, new_y, won, lost)."""
    dy = -1 if grav_up else 1
    tx, ty = px, py + dy
    if ty < 0 or ty >= 32: return (px, py, False, False)
    cell = grid[ty][tx]
    # Check initial cell
    if cell == '#':
        return (px, py, False, False)  # can't move into wall
    if override and cell in ('g', '1', 'x', 'y', 'u', 'v'):
        return (px, py, False, False)  # override reject
    # Now check gem, spike, etc.
    if cell == '+':
        # immediately at gem? No, fall enters eylagpkfjn first. Actually this is where we step into.
        # Per engine, if first cell is gem, proceed to fsvnqdbzrp which handles it.
        pass
    # Proceed to fsvnqdbzrp
    won, lost, fx, fy = fsvnqdbzrp(grid, tx, ty, grav_up)
    # But wait — fsvnqdbzrp starts from AFTER eylagpkfjn. We need to handle cell check at eylagpkfjn first.
    # Actually: if cell == '+' (gem) at eylagpkfjn, player arrives there and wins.
    # If cell == '1', override rejected (done). Non-override: proceed to fsvnqdbzrp.
    # If cell == '2' or '.', proceed.
    # If cell == 'u'/'v', lose (spike direct)? Actually pbsitubcfd enters fsvnqdbzrp regardless.
    if cell == 'u' or cell == 'v':
        # spike at step location (non-override). Move to spike cell and die.
        # Actually based on engine: this case doesn't hit override branch because override excludes u/v.
        # Non-override: allowed to proceed to fsvnqdbzrp. fsvnqdbzrp checks (tx, ty+dy).
        # But wait, fsvnqdbzrp starts at (tx, ty+dy) with udsicoryza=(tx, ty). If we're stepping into spike (tx, ty),
        # then udsicoryza=(tx, ty) which is the spike cell... but fsvnqdbzrp doesn't check udsicoryza for spike.
        # Looking at engine: the spike check only happens via the fsvnqdbzrp loop terminator. udsicoryza is just
        # "cell before terminator". So stepping INTO spike directly doesn't use spike logic.
        # Actually — but pywlvyklps explicitly rejects move-into-spike via else branch (not in move list).
        # Here in pbsitubcfd, we don't have that gate. So pbsitubcfd would land player at spike without lose().
        # But that's probably unreachable through clicks.
        # For our simulator: just do fsvnqdbzrp.
        pass
    # Return fsvnqdbzrp result
    if won:
        return (fx, fy, True, False)
    if lost:
        return (fx, fy, False, True)
    # Else: check if the terminator of fall was '+'
    return (fx, fy, False, False)


def step_move(grid, px, py, grav_up, dx):
    """Horizontal step + fall. Returns (new_x, new_y, won, lost, new_grav_up)."""
    tx, ty = px + dx, py
    if tx < 0 or tx >= 11:
        return (px, py, False, False, grav_up)  # out of bounds
    cell = grid[ty][tx]
    if cell == '+':
        return (tx, ty, True, False, grav_up)
    if cell in ('.', '2'):
        # Move/fall
        won, lost, fx, fy = fsvnqdbzrp(grid, tx, ty, grav_up)
        if won:
            return (fx, fy, True, False, grav_up)
        if lost:
            return (fx, fy, False, True, grav_up)
        # Determine new position: if fall_count > 0, use fallen; else use target
        # For simplicity: the fsvnqdbzrp result fx, fy represents landing (udsicoryza).
        return (fx, fy, False, False, grav_up)
    # wall, u, v, 1, g, x, y: no move (bounce)
    return (px, py, False, False, grav_up)


def step_click(grid, px, py, grav_up, cx, cy):
    """Click at (cx, cy). Returns (new_grid, new_px, new_py, new_grav, won, lost)."""
    if not (0 <= cx < 11 and 0 <= cy < 32):
        return (grid, px, py, grav_up, False, False)
    cell = grid[cy][cx]
    new_grid = grid
    if cell == 'g':
        # Flip gravity, consume g, pbsitubcfd with override
        new_grid = [row[:] for row in grid]
        new_grid[cy][cx] = '.'
        new_grav = not grav_up
        nx, ny, won, lost = pbsitubcfd(new_grid, px, py, new_grav, override=True)
        return (new_grid, nx, ny, new_grav, won, lost)
    if cell == '1':
        # Convert to 2, then pbsitubcfd non-override
        new_grid = [row[:] for row in grid]
        new_grid[cy][cx] = '2'
        # pbsitubcfd requires click target == (px, py+dy)
        dy = -1 if grav_up else 1
        if (cx, cy) == (px, py + dy):
            # At time of pbsitubcfd evaluation, cell was still '1' originally. But pbsitubcfd checks cell
            # contents NOW. After conversion: cell is '2'. Hmm, but the check is BEFORE conversion in engine.
            # Let me trust: pbsitubcfd checks cell at call time. Conversion is scheduled in animation queue.
            # So pbsitubcfd uses PRE-conversion state.
            # For simplicity in sim: pbsitubcfd on '2' behavior.
            # Actually per engine code, pbsitubcfd is called with kojxiszwpx = click target; but it recomputes
            # eylagpkfjn = (px, py+dy). And reads names at eylagpkfjn which is STILL '1' at call time.
            # Then checks `xcjjwqfzjfe in list or (override and ...)`. Non-override, '1' not wall → proceed.
            # Then fsvnqdbzrp((cx, cy)=(px, py+dy)). This is fall FROM (cx, cy).
            # udsicoryza = (cx, cy), eylagpkfjn = (cx, cy+dy). continue loop based on next cell.
            # So fall path starts AT (cx, cy). The '1' cell itself is passed through.
            won, lost, fx, fy = fsvnqdbzrp(grid, cx, cy, grav_up)  # use ORIGINAL grid (pre-conversion)
            return (new_grid, fx, fy, grav_up, won, lost)
        # Not adjacent → no player move, just conversion
        return (new_grid, px, py, grav_up, False, False)
    if cell == '2':
        # Convert to 1, then pbsitubcfd non-override
        new_grid = [row[:] for row in grid]
        new_grid[cy][cx] = '1'
        dy = -1 if grav_up else 1
        if (cx, cy) == (px, py + dy):
            won, lost, fx, fy = fsvnqdbzrp(grid, cx, cy, grav_up)
            return (new_grid, fx, fy, grav_up, won, lost)
        return (new_grid, px, py, grav_up, False, False)
    if cell == 'x':
        # Destroy x spike, then pbsitubcfd non-override
        new_grid = [row[:] for row in grid]
        new_grid[cy][cx] = '.'
        dy = -1 if grav_up else 1
        if (cx, cy) == (px, py + dy):
            won, lost, fx, fy = fsvnqdbzrp(new_grid, cx, cy, grav_up)  # use NEW grid (x destroyed)
            return (new_grid, fx, fy, grav_up, won, lost)
        return (new_grid, px, py, grav_up, False, False)
    # Other cells: no-op
    return (grid, px, py, grav_up, False, False)


def state_sig(grid, px, py, grav_up):
    g_grid = tuple(tuple(row) for row in grid)
    return (px, py, grav_up, g_grid)


def bfs(start_grid, start_px, start_py, start_grav_up, max_states=100000, max_time=60):
    sig0 = state_sig(start_grid, start_px, start_py, start_grav_up)
    visited = {sig0: None}
    queue = deque([(start_grid, start_px, start_py, start_grav_up, [])])
    start_t = time.time()
    expansions = 0
    while queue:
        grid, px, py, grav_up, path = queue.popleft()
        expansions += 1
        if expansions % 500 == 0:
            print(f'  exp={expansions}, queue={len(queue)}, visited={len(visited)}, t={time.time()-start_t:.1f}s')
        if time.time() - start_t > max_time:
            print('TIMEOUT')
            return None
        if len(visited) > max_states:
            print('CAP')
            return None
        # Actions: LEFT, RIGHT, one CLICK on any g (all equivalent for flip), click on 1/2 tiles
        # Focus remote clicks on tiles near relevant paths
        actions = [('LEFT', None), ('RIGHT', None)]
        # One g click (they all flip gravity the same way)
        for y in range(32):
            found = False
            for x in range(11):
                if grid[y][x] == 'g':
                    actions.append(('CLICK', (x, y)))
                    found = True
                    break
            if found: break
        # Adjacent 1/2 (direct interaction)
        dy = -1 if grav_up else 1
        ay = py + dy
        if 0 <= ay < 32:
            for ax in range(11):
                if grid[ay][ax] in ('1', '2') and ax == px:
                    actions.append(('CLICK', (px, ay)))
        # Remote 1/2 clicks on cells that could be in a fall path
        # Key: tiles that could block fall in col 6 at row 21, and in cols 4-8 row 13-21
        relevant = set()
        # Tiles that could stop a fall from above at row 19-26
        for tx in range(2, 10):
            for ty in range(13, 27):
                if grid[ty][tx] in ('1', '2'):
                    relevant.add((tx, ty))
        for xy in relevant:
            actions.append(('CLICK', xy))
        for act, param in actions:
            if act == 'LEFT':
                nx, ny, won, lost, ngrav = step_move(grid, px, py, grav_up, -1)
                ngrid = grid
            elif act == 'RIGHT':
                nx, ny, won, lost, ngrav = step_move(grid, px, py, grav_up, 1)
                ngrid = grid
            else:
                ngrid, nx, ny, ngrav, won, lost = step_click(grid, px, py, grav_up, param[0], param[1])
            if lost:
                continue
            if won:
                print(f'  WIN! path_len={len(path)+1}')
                return path + [(act, param)]
            new_sig = state_sig(ngrid, nx, ny, ngrav)
            if new_sig in visited:
                continue
            visited[new_sig] = (act, param)
            queue.append((ngrid, nx, ny, ngrav, path + [(act, param)]))
    print('No path found')
    return None


if __name__ == '__main__':
    grid = build_grid()
    # Print rows of interest
    print('L7 grid (y=4..31):')
    print('    ' + ''.join(f'{x:2}' for x in range(11)))
    for y in range(4, 32):
        row = f'{y:2}  '
        for x in range(11):
            row += ' ' + grid[y][x]
        print(row)
    print()
    # Verify gem and spawn
    print(f'Gem at: {[(x, y) for y in range(32) for x in range(11) if grid[y][x] == "+"]}')

    path = bfs(grid, 3, 19, True, max_states=200000, max_time=120)
    if path:
        print('\nWINNING PATH:')
        for i, (a, p) in enumerate(path):
            if a == 'CLICK':
                print(f'  {i+1}. CLICK ({p[0]}, {p[1]})')
            else:
                print(f'  {i+1}. {a}')
