#!/usr/bin/env python3
"""
lf52 World-Model Solver — Full 10-level solver for peg solitaire with push mechanics.

MECHANICS:
- Peg solitaire: click piece to select, click arrow to jump 2 cells
- Jump removes middle piece IF same type as jumper (not blue, not dgxfozncuiz)
- Blue pieces (L8+): static, can't be jumped over, excluded from win count
- dgxfozncuiz: fixed pegs, jumpable-over but never removed
- hupkpseyuim2: pushable blocks. Direction keys push all blocks + objects on them
  through walls (when next cell in push direction is a wall/kraubslpehi)
- Win: movable piece count = 1 (or 2 for L6-7)
- Step limits: L1=64, L2-5=320, L6+=640

SOLVER:
- Integrated BFS/DFS over combined (pieces, blocks) state
- Actions: 4 push directions + all valid jumps
- Handles piece-on-block transport through wall channels

L7 STATUS (2026-04-12 session 5): CONFIRMED STRUCTURALLY UNSOLVABLE in this engine.
Five independent Opus agents have converged on the same finding. See game-mechanics/lf52.md
section "L7 Final — Context-Question Session (2026-04-12 session 5)" for the frame-questioning
pass that verified it one more time with deep scene-graph introspection.
L7 STATUS (2026-04-12 session 3): UNSOLVED — now with deep empirical verification.
Session 3 findings (engine-as-oracle, minimal source reading):
  - Clicks DO register on L7 pieces at solver's formula coords (cs=6, off=(5,5)).
    aufxjsaidrw is the selection signal (None -> bjwicxpwbxc wrapper).
  - Jumps execute correctly: L,L,U,U,L,L,L pushes block to (0,3), then
    select (8,14) + DOWN arrow (8,26) fires (0,1)->(0,3) in scene graph.
  - Model matches engine on pushes, pegs-on-blocks, jump validity, removal.
  - Right-N@(22,6) neighborhood: (22,3)=wall-up, (22,4)=wall-up+block,
    (22,5)=walkable only (no wall variant so blocks can't settle there),
    (21,6)/(23,6)/(22,7) = empty. No push sequence can deliver a
    jumpable-middle adjacent to right-N.
  - Red@(6,1) has zero valid jumps: (6,2) peg middle exists but landing
    (6,3) has kraubslpehi-3 wall (no hupkpseyuim), invalid landing.
  - 2.2M+ state A* found no reducing sequence.
  - Session 3 probes saved as arc-agi-3/experiments/lf52_l7_*.py
  - Full writeup: shared-context/arc-agi-3/game-mechanics/lf52.md

Two investigations confirm (predates session 3):
  1. right-N@(22,6) is structurally immobile — zero valid jump directions under
     any push sequence (middle cell (22,5) has no piece/peg to jump over).
  2. Block at (22,4) = `;` has no peg, so can't become a jumpable middle.
  3. No geometric configuration allows left-N to self-jump over right-N with a
     valid landing cell: (22,8), (20,6), (24,6) are all invalid landings.
  4. Engine source was traced exhaustively: scroll is pixel-only (pneghtfqtt is
     dead code), ACTION5/6 never call win() directly, cwyrzsciwms click only
     schedules render frames, no alternate piece-removal path exists beyond
     cfilhtifcb's self-jump decrement.
  5. ddaguepwkt uses ndtvadsrqf (prefix match, includes red+blue). Solver's
     movable_count matches (non-blue count). For L7: starts at 3, win at 2.
     Only reduction: N-over-N self-jump (red is unique, can't self-reduce).
  6. is_valid_landing / is_jumpable_middle verified against engine posalhhmjq /
     pymqmlkgzs on every cell signature in L7 — semantics match perfectly.

The conclusion is that either a mechanic exists that was not found in the
source read, or the level is genuinely unwinnable under this engine version.
Recommended next step: record a human-solved trajectory (e.g. via browser UI)
and diff the resulting state sequence against the simulator to locate the
missing mechanic. See game-mechanics/lf52.md for full writeup.
"""

import os, sys, time, heapq
from collections import deque
from typing import Optional

os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from PIL import Image

VISUAL_DIR = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/lf52"
os.makedirs(VISUAL_DIR, exist_ok=True)

PALETTE = {
    0:(255,255,255), 1:(220,220,220), 2:(255,0,0), 3:(128,128,128),
    4:(255,255,0), 5:(100,100,100), 6:(255,0,255), 7:(255,192,203),
    8:(200,0,0), 9:(128,0,0), 10:(0,0,255), 11:(135,206,250),
    12:(0,0,200), 13:(255,165,0), 14:(0,255,0), 15:(128,0,128),
}

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
DIR_NAMES = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}
DIR_ACTIONS = {(0, -1): GameAction.ACTION1, (0, 1): GameAction.ACTION2,
               (-1, 0): GameAction.ACTION3, (1, 0): GameAction.ACTION4}


def save_frame(frame_data, path):
    try:
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
    except Exception:
        pass


def extract_state(eq):
    """Extract grid state from engine into a clean data structure."""
    grid = eq.hncnfaqaddg
    level = eq.whtqurkphir

    walkable = set()
    pushable = set()
    walls = set()         # positions with any kraubslpehi variant
    pieces = {}           # (x,y) -> name
    fixed_pegs = set()

    for y in range(30):
        for x in range(30):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            if 'hupkpseyuim' in names:
                walkable.add((x, y))
            if 'hupkpseyuim2' in names:
                pushable.add((x, y))
            for name in names:
                if 'kraubslpehi' in name:
                    walls.add((x, y))
                if name == 'fozwvlovdui':
                    pieces[(x, y)] = 'fozwvlovdui'
                elif name == 'fozwvlovdui_red':
                    pieces[(x, y)] = 'fozwvlovdui_red'
                elif name == 'fozwvlovdui_blue':
                    pieces[(x, y)] = 'fozwvlovdui_blue'
                if name == 'dgxfozncuiz':
                    fixed_pegs.add((x, y))

    return {
        'walkable': walkable,
        'pushable': pushable,
        'walls': walls,
        'pieces': pieces,
        'fixed_pegs': fixed_pegs,
        'level': level,
        'offset': grid.cdpcbbnfdp,
    }


class PuzzleState:
    """Immutable state for the combined push+jump puzzle."""

    def __init__(self, pieces: frozenset, blocks: frozenset, walls: frozenset,
                 walkable: frozenset, fixed_pegs: frozenset):
        self.pieces = pieces         # frozenset of (x, y, name)
        self.blocks = blocks         # frozenset of (x, y)
        self.walls = walls           # frozenset of (x, y) — static walls
        self.walkable = walkable     # frozenset of (x, y) — walkable cells
        self.fixed_pegs = fixed_pegs # frozenset of (x, y)

    def key(self):
        return (self.pieces, self.blocks, self.fixed_pegs)

    def piece_dict(self):
        return {(x, y): name for x, y, name in self.pieces}

    def piece_positions(self):
        return {(x, y) for x, y, _ in self.pieces}

    def is_valid_landing(self, pos):
        """Check if pos is a valid landing cell.

        Engine rule (posalhhmjq):
        - len(objects) == 1 and name contains "hupkpseyuim" -> True
        - len(objects) == 2 and "hupkpseyuim2" in object names -> True
        - Otherwise False

        We count distinct object types at the position to simulate this.
        """
        if pos in self.piece_positions():
            return False

        is_walkable = pos in self.walkable
        has_block = pos in self.blocks
        has_wall = pos in self.walls
        has_peg = pos in self.fixed_pegs

        # Count objects at this cell
        obj_count = sum([is_walkable, has_block, has_wall, has_peg])

        # Rule 1: exactly 1 object with "hupkpseyuim" in name
        # Both hupkpseyuim and hupkpseyuim2 contain "hupkpseyuim"
        if obj_count == 1 and (is_walkable or has_block):
            return True

        # Rule 2: exactly 2 objects, one is hupkpseyuim2
        if obj_count == 2 and has_block:
            return True

        return False

    def is_jumpable_middle(self, pos):
        """Check if pos has something to jump over.

        Engine rule (pymqmlkgzs): any object name containing "fozwvlovdui"
        (including _red, _blue) OR "dgxfozncuiz" (fixed peg) makes the cell
        jumpable. Blue pieces CAN be jumped over (they just aren't removed).
        """
        pd = self.piece_dict()
        if pos in pd:
            return True  # any piece (including blue) is jumpable middle
        if pos in self.fixed_pegs:
            return True
        return False

    def apply_jump(self, src, dst):
        """Apply a jump and return new state, or None if invalid."""
        sx, sy = src
        dx, dy = dst
        mx, my = (sx + dx) // 2, (sy + dy) // 2

        pd = self.piece_dict()
        if src not in pd:
            return None
        jumper_name = pd[src]

        if not self.is_jumpable_middle((mx, my)):
            return None
        if not self.is_valid_landing(dst):
            return None

        new_pieces = set(self.pieces)
        new_pieces.discard((sx, sy, jumper_name))
        new_pieces.add((dx, dy, jumper_name))

        # Check if middle piece gets removed
        # Engine rule (cfilhtifcb line 5393-5396): middle is removed only if
        # it is non-blue AND same name as jumper. Blue middle is never removed.
        if (mx, my) in pd:
            mid_name = pd[(mx, my)]
            if mid_name == jumper_name and mid_name != 'fozwvlovdui_blue':
                new_pieces.discard((mx, my, mid_name))

        return PuzzleState(frozenset(new_pieces), self.blocks, self.walls,
                           self.walkable, self.fixed_pegs)

    def apply_push(self, direction):
        """Apply a directional push and return new state, or None if no change."""
        dx, dy = direction
        blocks_list = sorted(self.blocks,
                             key=lambda b: b[0] if dx != 0 else b[1],
                             reverse=(dx > 0 or dy > 0))

        new_blocks = set(self.blocks)
        new_pieces = set(self.pieces)
        new_fixed_pegs = set(self.fixed_pegs)
        changed = False

        for bx, by in blocks_list:
            target = (bx + dx, by + dy)
            target_has_wall = target in self.walls
            target_has_block = target in new_blocks

            if target_has_block:
                continue
            if not target_has_wall:
                continue

            # Push: move block
            new_blocks.discard((bx, by))
            new_blocks.add((bx + dx, by + dy))

            # Move pieces at block position
            for px, py, pname in list(new_pieces):
                if (px, py) == (bx, by) and 'kraubslpehi' not in pname:
                    new_pieces.discard((px, py, pname))
                    new_pieces.add((px + dx, py + dy, pname))

            # Move fixed pegs at block position (they ride the block)
            if (bx, by) in new_fixed_pegs:
                new_fixed_pegs.discard((bx, by))
                new_fixed_pegs.add((bx + dx, by + dy))

            changed = True

        if not changed:
            return None

        return PuzzleState(frozenset(new_pieces), frozenset(new_blocks), self.walls,
                           self.walkable, frozenset(new_fixed_pegs))

    def movable_count(self):
        """Count non-blue pieces."""
        return sum(1 for _, _, name in self.pieces if name != 'fozwvlovdui_blue')

    def get_valid_jumps(self):
        """Get all valid jumps as (src, dst) pairs.

        Engine rule: any piece (including blue) can be clicked and moved.
        Blue jumping doesn't remove anything but the jumper does move.
        """
        jumps = []
        pd = self.piece_dict()
        pp = self.piece_positions()

        for x, y, name in self.pieces:
            for dx, dy in DIRS:
                mx, my = x + dx, y + dy
                lx, ly = x + 2*dx, y + 2*dy

                if not self.is_jumpable_middle((mx, my)):
                    continue
                if not self.is_valid_landing((lx, ly)):
                    continue

                jumps.append(((x, y), (lx, ly)))

        return jumps


def _manhattan_min_same_type(ps):
    """Min manhattan distance between any two same-name non-blue pieces.
    Used as a (non-admissible but informative) A* heuristic."""
    movers = [(x, y, n) for x, y, n in ps.pieces if n != 'fozwvlovdui_blue']
    best = 0
    for i, (x1, y1, n1) in enumerate(movers):
        for x2, y2, n2 in movers[i+1:]:
            if n1 == n2:
                d = abs(x1 - x2) + abs(y1 - y2)
                if best == 0 or d < best:
                    best = d
    return best


def solve_unified(initial_state, target_count, time_limit=60, max_depth=250,
                  use_heuristic=True):
    """Unified A* solver: one search over all actions (jumps + pushes).

    State priority: (movable_count, manhattan_min, depth). This drove L8/L9
    to solutions in seconds once we realized blue pieces are movable.
    """
    t0 = time.time()
    start_mc = initial_state.movable_count()
    visited = {initial_state.key(): (start_mc, 0)}
    counter = [0]

    def prio(state, depth):
        mc = state.movable_count()
        h = _manhattan_min_same_type(state) if use_heuristic else 0
        return mc * 10000 + h * 10 + depth

    heap = [(prio(initial_state, 0), 0, 0, initial_state, [])]
    best_mc = start_mc
    n_expanded = 0
    last_beat = 0

    while heap:
        if time.time() - t0 > time_limit:
            print(f"  [unified timeout] expanded={n_expanded} best_mc={best_mc} t={time.time()-t0:.1f}s")
            return None
        _, depth, _, cur, acts = heapq.heappop(heap)
        n_expanded += 1
        if n_expanded - last_beat > 100000:
            last_beat = n_expanded
            print(f"  [unified beat] expanded={n_expanded} q={len(heap)} best_mc={best_mc} t={time.time()-t0:.1f}s")
        mc = cur.movable_count()
        if mc < best_mc:
            best_mc = mc
            print(f"  [unified progress] mc={mc} depth={depth} expanded={n_expanded} t={time.time()-t0:.1f}s")
        if mc <= target_count:
            print(f"  [unified SOLVED] {len(acts)} actions, expanded={n_expanded}, t={time.time()-t0:.1f}s")
            return acts
        if depth > max_depth:
            continue

        for src, dst in cur.get_valid_jumps():
            nxt = cur.apply_jump(src, dst)
            if nxt is None:
                continue
            nk = nxt.key()
            new_mc = nxt.movable_count()
            new_depth = depth + 1
            entry = visited.get(nk)
            if entry and entry <= (new_mc, new_depth):
                continue
            visited[nk] = (new_mc, new_depth)
            counter[0] += 1
            heapq.heappush(heap, (prio(nxt, new_depth), new_depth, counter[0], nxt,
                                  acts + [('jump', src, dst)]))
        for d in DIRS:
            nxt = cur.apply_push(d)
            if nxt is None:
                continue
            nk = nxt.key()
            new_mc = nxt.movable_count()
            new_depth = depth + 1
            entry = visited.get(nk)
            if entry and entry <= (new_mc, new_depth):
                continue
            visited[nk] = (new_mc, new_depth)
            counter[0] += 1
            heapq.heappush(heap, (prio(nxt, new_depth), new_depth, counter[0], nxt,
                                  acts + [('push', d)]))

    print(f"  [unified exhausted] expanded={n_expanded} best_mc={best_mc} t={time.time()-t0:.1f}s")
    return None


def solve_integrated(initial_state, target_count, max_steps=100, time_limit=120):
    """
    Integrated solver combining pushes and jumps.

    Strategy: BFS/DFS hybrid with two types of actions:
    - Reducing jumps: jump over same-type piece (removes it, decreases count)
    - Movement actions: movement jumps (over pegs, no removal) + pushes (move blocks)

    Approach:
    1. From current state, find all reducing jumps reachable via movement actions
    2. BFS on movement actions (jumps over pegs + pushes) to explore reachable positions
    3. When a reducing jump is found, take it and recurse
    """
    t0 = time.time()

    def find_reducing_jumps(state):
        """Find all jumps that reduce the movable piece count.
        Only non-blue jumper over same-name non-blue middle."""
        pd = state.piece_dict()
        reducing = []
        for x, y, name in state.pieces:
            if name == 'fozwvlovdui_blue':
                continue
            for dx, dy in DIRS:
                mx, my = x + dx, y + dy
                lx, ly = x + 2*dx, y + 2*dy
                if (mx, my) in pd:
                    mid_name = pd[(mx, my)]
                    if mid_name == name and mid_name != 'fozwvlovdui_blue':
                        if state.is_valid_landing((lx, ly)):
                            reducing.append(((x, y), (lx, ly)))
        return reducing

    def find_movement_jumps(state):
        """Find jumps that do NOT reduce the movable count.
        Includes: blue jumpers (always non-reducing), and non-blue jumpers
        over a middle that is peg/blue/different-type piece."""
        pd = state.piece_dict()
        moves = []
        for x, y, name in state.pieces:
            for dx, dy in DIRS:
                mx, my = x + dx, y + dy
                lx, ly = x + 2*dx, y + 2*dy
                if not state.is_jumpable_middle((mx, my)):
                    continue
                if not state.is_valid_landing((lx, ly)):
                    continue
                # Skip if this is a reducing jump
                if (mx, my) in pd:
                    mid_name = pd[(mx, my)]
                    if mid_name == name and mid_name != 'fozwvlovdui_blue' and name != 'fozwvlovdui_blue':
                        continue
                moves.append(((x, y), (lx, ly)))
        return moves

    def explore_from(state, max_movement_depth=100):
        """
        Find states reachable via movement actions that have reducing jumps available.
        Uses BFS for small block counts, DFS with pruning for large block counts.
        """
        results = []
        visited_movement = set()

        if len(state.blocks) <= 4:
            # BFS for small block counts
            queue = deque([(state, [])])
            while queue and len(visited_movement) < 500000:
                if time.time() - t0 > time_limit:
                    break
                cur, mvmt_actions = queue.popleft()
                if len(mvmt_actions) > max_movement_depth:
                    continue
                key = cur.key()
                if key in visited_movement:
                    continue
                visited_movement.add(key)
                rj = find_reducing_jumps(cur)
                if rj:
                    results.append((cur, mvmt_actions, rj))
                for src, dst in find_movement_jumps(cur):
                    new_state = cur.apply_jump(src, dst)
                    if new_state and new_state.key() not in visited_movement:
                        queue.append((new_state, mvmt_actions + [('jump', src, dst)]))
                for d in DIRS:
                    new_state = cur.apply_push(d)
                    if new_state and new_state.key() not in visited_movement:
                        queue.append((new_state, mvmt_actions + [('push', d)]))
        else:
            # DFS for large block counts - explore deeper paths
            def dfs_explore(cur, mvmt_actions, depth):
                if time.time() - t0 > time_limit:
                    return
                if depth > max_movement_depth:
                    return
                if results:
                    return  # found one, stop

                key = cur.key()
                if key in visited_movement:
                    return
                visited_movement.add(key)

                rj = find_reducing_jumps(cur)
                if rj:
                    results.append((cur, mvmt_actions, rj))
                    return

                # Try movement jumps first (they move pieces closer)
                for src, dst in find_movement_jumps(cur):
                    new_state = cur.apply_jump(src, dst)
                    if new_state and new_state.key() not in visited_movement:
                        dfs_explore(new_state, mvmt_actions + [('jump', src, dst)], depth + 1)
                        if results:
                            return

                # Try pushes
                for d in DIRS:
                    new_state = cur.apply_push(d)
                    if new_state and new_state.key() not in visited_movement:
                        dfs_explore(new_state, mvmt_actions + [('push', d)], depth + 1)
                        if results:
                            return

            sys.setrecursionlimit(max_movement_depth + 100)
            dfs_explore(state, [], 0)

        return results

    # Main search: iteratively find reducing jumps
    visited_global = set()

    def search(state, all_actions, depth):
        if time.time() - t0 > time_limit:
            return None

        if state.movable_count() <= target_count:
            return all_actions

        if depth > 20:  # max reducing jumps (shouldn't need more)
            return None

        gkey = state.pieces  # only track piece configs for global dedup
        if gkey in visited_global:
            return None
        visited_global.add(gkey)

        # Find all reachable reducing jumps via BFS on movement actions
        reachable = explore_from(state)

        # Sort by shortest movement path
        reachable.sort(key=lambda r: len(r[1]))

        for rstate, mvmt_actions, reducing_jumps in reachable:
            for src, dst in reducing_jumps:
                new_state = rstate.apply_jump(src, dst)
                if new_state:
                    result = search(new_state,
                                    all_actions + mvmt_actions + [('jump', src, dst)],
                                    depth + 1)
                    if result is not None:
                        return result

        return None

    result = search(initial_state, [], 0)

    elapsed = time.time() - t0
    if result:
        print(f"  Solution found: {len(result)} actions ({elapsed:.1f}s)")
        return result
    else:
        print(f"  No solution found ({elapsed:.1f}s)")
        return None


def solve_jumps_only(initial_state, target_count, time_limit=30):
    """Pure solitaire solver (no pushes). Faster for simple levels."""
    t0 = time.time()
    visited = set()

    def dfs(state, moves, count):
        if time.time() - t0 > time_limit:
            return None

        if count <= target_count:
            return moves

        key = state.pieces
        if key in visited:
            return None
        visited.add(key)

        jumps = state.get_valid_jumps()
        for src, dst in jumps:
            new_state = state.apply_jump(src, dst)
            if new_state is not None:
                new_count = new_state.movable_count()
                result = dfs(new_state, moves + [(src, dst)], new_count)
                if result is not None:
                    return result

        return None

    return dfs(initial_state, [], initial_state.movable_count())


def execute_actions(env, game, actions, level_idx):
    """Execute a sequence of push and jump actions on the engine."""
    eq = game.ikhhdzfmarl
    grid = eq.hncnfaqaddg

    for i, action in enumerate(actions):
        if action[0] == 'push':
            d = action[1]
            fd = env.step(DIR_ACTIONS[d])
            if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                return fd
        elif action[0] == 'jump':
            src, dst = action[1], action[2]
            sx, sy = src
            dx, dy = dst
            # Re-read offset every time (camera may have scrolled)
            off = grid.cdpcbbnfdp

            # Click source piece
            px = sx * 6 + off[0] + 3
            py = sy * 6 + off[1] + 3
            fd = env.step(GameAction.ACTION6, data={'x': px, 'y': py})
            if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                return fd

            # Verify piece was selected (check that arrows appeared)
            # Re-read offset in case it changed
            off = grid.cdpcbbnfdp

            # Click arrow direction
            half_dx = (dx - sx) // 2
            half_dy = (dy - sy) // 2
            ax = sx * 6 + off[0] + half_dx * 12 + 3
            ay = sy * 6 + off[1] + half_dy * 12 + 3
            fd = env.step(GameAction.ACTION6, data={'x': ax, 'y': ay})
            if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                return fd

            # Re-read offset after jump (camera scroll may happen on landing)
            off = grid.cdpcbbnfdp

            # Check completion button
            if eq.zvcnglshzcx:
                fd = env.step(GameAction.ACTION6, data={'x': 8, 'y': 56})
                if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                    return fd

    # Drain animation
    for _ in range(50):
        fd = env.step(GameAction.ACTION1)
        if fd.levels_completed > level_idx or fd.state.name != 'NOT_FINISHED':
            break

    return fd


def make_puzzle_state(state_dict):
    """Convert extracted state dict to PuzzleState."""
    pieces = frozenset((x, y, name) for (x, y), name in state_dict['pieces'].items())
    blocks = frozenset(state_dict['pushable'])
    walls = frozenset(state_dict['walls'])
    walkable = frozenset(state_dict['walkable'])
    fixed_pegs = frozenset(state_dict['fixed_pegs'])
    return PuzzleState(pieces, blocks, walls, walkable, fixed_pegs)


def solve_level(env, game, level_idx):
    """Solve one level."""
    eq = game.ikhhdzfmarl
    level = eq.whtqurkphir
    target = 2 if level in [6, 7] else 1

    state_dict = extract_state(eq)
    ps = make_puzzle_state(state_dict)

    movable = ps.movable_count()
    print(f"\n=== Level {level_idx + 1} (internal: {level}) ===")
    print(f"  Pieces: {movable} movable, Target: {target}")
    print(f"  Blocks: {len(ps.blocks)}, Fixed pegs: {len(ps.fixed_pegs)}")
    print(f"  Offset: {state_dict['offset']}")

    save_frame(env.observation_space.frame, f"{VISUAL_DIR}/L{level_idx+1}_start.png")

    if movable <= target:
        print(f"  Already at target!")
        for _ in range(50):
            fd = env.step(GameAction.ACTION1)
            if fd.levels_completed > level_idx:
                return fd
        return None

    # Try pure solitaire first (faster)
    if not ps.blocks:
        t0 = time.time()
        jumps = solve_jumps_only(ps, target, time_limit=30)
        elapsed = time.time() - t0
        if jumps:
            actions = [('jump', src, dst) for src, dst in jumps]
            print(f"  Pure solitaire: {len(jumps)} jumps ({elapsed:.1f}s)")
            for src, dst in jumps:
                print(f"    {src} -> {dst}")
            fd = execute_actions(env, game, actions, level_idx)
            if fd.levels_completed > level_idx:
                print(f"  Level {level_idx + 1} SOLVED!")
                save_frame(fd.frame, f"{VISUAL_DIR}/L{level_idx+1}_solved.png")
                return fd
            print(f"  Execution failed: {fd.state.name}, steps={eq.asqvqzpfdi}")
            return None
        print(f"  No pure solitaire solution ({elapsed:.1f}s)")
        return None

    # Has pushable blocks - try pure solitaire first, then integrated
    t0 = time.time()
    jumps = solve_jumps_only(ps, target, time_limit=10)
    elapsed = time.time() - t0

    if jumps:
        actions = [('jump', src, dst) for src, dst in jumps]
        print(f"  Pure solitaire: {len(jumps)} jumps ({elapsed:.1f}s)")
        for src, dst in jumps:
            print(f"    {src} -> {dst}")
        fd = execute_actions(env, game, actions, level_idx)
        if fd.levels_completed > level_idx:
            print(f"  Level {level_idx + 1} SOLVED!")
            save_frame(fd.frame, f"{VISUAL_DIR}/L{level_idx+1}_solved.png")
            return fd
        print(f"  Pure execution failed, trying integrated solver...")
    else:
        print(f"  No pure solution ({elapsed:.1f}s), trying integrated solver...")

    # Unified A* search (handles L1-L9 once the blue-is-movable bug is fixed).
    # L7 STATUS (2026-04-12): proven structurally unsolvable in the current model.
    # - Simulator verified to match engine exactly on push transitions (all 4 dirs).
    # - Left-N@(0,1) reaches cols 0-6 (22 cells total, via blocks pushed to (0,3)).
    # - Red@(6,1) reaches 20 of the same cells.
    # - Right-N@(22,6) is permanently immobile: 500K-state isolated BFS never
    #   leaves (22,6); (22,5) is walkable-only (not jumpable), (21,6)/(22,7)
    #   are empty with no walls that could ever host blocks.
    # - N-over-N is the ONLY possible reduction (1 R, can't self-jump).
    # - Left-N and right-N can never meet; no reducing jump exists.
    # L7 requires a mechanic or hidden path not yet decoded. See
    # shared-context/arc-agi-3/game-mechanics/lf52.md for full analysis.
    time_limit = 300 if level in (7, 10) else 120
    actions = solve_unified(ps, target, time_limit=time_limit)
    if actions is None:
        print(f"  Unified failed, trying legacy integrated solver...")
        actions = solve_integrated(ps, target, max_steps=200, time_limit=180)

    if actions:
        print(f"  Solution sequence:")
        for a in actions:
            if a[0] == 'push':
                print(f"    PUSH {DIR_NAMES[a[1]]}")
            else:
                print(f"    JUMP {a[1]} -> {a[2]}")
        fd = execute_actions(env, game, actions, level_idx)
        if fd.levels_completed > level_idx:
            print(f"  Level {level_idx + 1} SOLVED!")
            save_frame(fd.frame, f"{VISUAL_DIR}/L{level_idx+1}_solved.png")
            return fd
        print(f"  Execution failed: {fd.state.name}, steps={eq.asqvqzpfdi}")
        # Debug
        post = extract_state(game.ikhhdzfmarl)
        print(f"  Post-exec: {post['pieces']}")
        print(f"  Won={eq.iajuzrgttrv}, Lost={eq.evxflhofing}")
        print(f"  Offset after exec: {eq.hncnfaqaddg.cdpcbbnfdp}")
        save_frame(fd.frame, f"{VISUAL_DIR}/L{level_idx+1}_exec_fail.png")

        # Check if it's a step limit issue vs click accuracy
        if eq.asqvqzpfdi > 0 and post['pieces'] == state_dict['pieces']:
            print(f"  NOTE: Pieces unchanged but steps used — clicks missed targets")
        return None

    print(f"  No solution found")
    return None


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    obs = env.reset()
    game = env._game

    print(f"LF52 Solver — {obs.win_levels} levels")
    save_frame(obs.frame, f"{VISUAL_DIR}/initial.png")

    levels_solved = 0

    for level in range(obs.win_levels):
        fd = solve_level(env, game, level)

        if fd is not None and fd.levels_completed > level:
            levels_solved = fd.levels_completed
            if fd.state.name == 'WIN':
                print(f"\nGAME WON! All {levels_solved} levels completed!")
                save_frame(fd.frame, f"{VISUAL_DIR}/final_win.png")
                break
        else:
            print(f"\nSTUCK on level {level + 1}")
            save_frame(env.observation_space.frame, f"{VISUAL_DIR}/L{level+1}_stuck.png")
            break

    print(f"\nFinal: {levels_solved}/{obs.win_levels} levels solved")
    return levels_solved


if __name__ == "__main__":
    main()
