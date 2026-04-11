#!/usr/bin/env python3
"""
sp80 Gravity Pipes — Accurate Simulator & Solver

Water flow rules (from game source analysis):
  - Water drops at (x,y) with direction (dx,dy), initial = (0,1) = downward
  - Each step, look at next cell (x+dx, y+dy):
    * Empty → create water there, same direction
    * Water → continue through (add to next active, don't create new)
    * Pipe (ksmzdcblcz) → split: create water at perpendicular positions
      from CURRENT pos with SAME direction. Perpendicular = left/right if
      moving vertically, up/down if moving horizontally.
    * Receptacle (xsrqllccpx) → fill check: both perpendicular neighbors
      from current pos must be cells of SAME receptacle. If yes=fill, else splash.
    * L-pipe (hfjpeygkxy) → deflect 90°. Check which side of L-pipe water
      touches: if adj1 is L-pipe and adj2 is empty → new_dir = (dy, -dx).
      If adj2 is L-pipe and adj1 is empty → new_dir = (-dy, dx).
    * Danger (uzunfxpwmd) → failure
  - Pipes act as deflectors: water runs along pipe surface, falling off both ends.
  - Key insight: pipe at (px, py) width w creates exit streams at x=(px-1) and x=(px+w)
"""

import sys
import numpy as np
from itertools import product
from typing import List, Tuple, Optional, Dict, Set, Any

# ---------------------------------------------------------------------------
# Sprite data
# ---------------------------------------------------------------------------

SPRITE_PIXELS = {
    "jgfvrvnkaz": [[8,8,8,8,8]],
    "odioorqnkn": [[8,8,8,8,8,8]],
    "trurgcakbj": [[8,8,8,8]],
    "untfxhpddv": [[8,8,8]],
    "nvzozwqarf": [[8,8,8,8,8,8,8,8]],
    "uihgaxtzkm": [[8,8,8,8,8,8,8]],
    "mdhkebfsmg": [[8],[8],[8],[8]],
    "adbrqflmwi": [[8,8,8,4,8,8,8]],
    "zgsbadjnjn": [[8,8,4,8,8]],
    "qwsmjdrvqj": [[15,-1],[15,15]],
    "vkwijvqdla": [[-1,15],[15,15]],
    "syaipsfndp": [[4]],
    "nkrtlkykwe": [[6]],
    "xsrqllccpx": [[11,-1,11],[11,11,11]],
    "uzunfxpwmd": [[1]*32],
}

SPRITE_TAGS = {
    "jgfvrvnkaz": ["ksmzdcblcz","sys_click"],
    "odioorqnkn": ["ksmzdcblcz","sys_click"],
    "trurgcakbj": ["ksmzdcblcz","sys_click"],
    "untfxhpddv": ["ksmzdcblcz","sys_click"],
    "nvzozwqarf": ["ksmzdcblcz","sys_click"],
    "uihgaxtzkm": ["ksmzdcblcz"],
    "mdhkebfsmg": ["ksmzdcblcz","sys_click"],
    "adbrqflmwi": ["ksmzdcblcz","syaipsfndp"],
    "zgsbadjnjn": ["ksmzdcblcz","syaipsfndp","sys_click"],
    "qwsmjdrvqj": ["hfjpeygkxy","sys_click"],
    "vkwijvqdla": ["hfjpeygkxy","sys_click"],
    "syaipsfndp": ["syaipsfndp"],
    "nkrtlkykwe": ["nkrtlkykwe"],
    "xsrqllccpx": ["xsrqllccpx"],
    "uzunfxpwmd": ["uzunfxpwmd"],
}

# ---------------------------------------------------------------------------
# Level definitions
# ---------------------------------------------------------------------------

LEVELS = [
    {  # Level 1
        "sprites": [
            ("jgfvrvnkaz", 3, 4, 0),
            ("nkrtlkykwe", 9, 1, 0),
            ("syaipsfndp", 9, 0, 0),
            ("uzunfxpwmd", 0, 15, 0),
            ("xsrqllccpx", 4, 13, 0),
            ("xsrqllccpx", 10, 13, 0),
        ],
        "grid": (16, 16), "steps": 30, "rotation": 0,
    },
    {  # Level 2
        "sprites": [
            ("jgfvrvnkaz", 6, 6, 0),
            ("nkrtlkykwe", 5, 1, 0),
            ("syaipsfndp", 5, 0, 0),
            ("untfxhpddv", 6, 9, 0),
            ("untfxhpddv", 11, 11, 0),
            ("uzunfxpwmd", 0, 15, 0),
            ("xsrqllccpx", 2, 13, 0),
            ("xsrqllccpx", 6, 13, 0),
            ("xsrqllccpx", 10, 13, 0),
        ],
        "grid": (16, 16), "steps": 45, "rotation": 180,
    },
    {  # Level 3
        "sprites": [
            ("jgfvrvnkaz", 1, 8, 0),
            ("nkrtlkykwe", 1, 1, 0),
            ("nkrtlkykwe", 14, 1, 0),
            ("nkrtlkykwe", 6, 1, 0),
            ("odioorqnkn", 8, 7, 0),
            ("odioorqnkn", 1, 5, 0),
            ("syaipsfndp", 1, 0, 0),
            ("syaipsfndp", 14, 0, 0),
            ("syaipsfndp", 6, 0, 0),
            ("trurgcakbj", 10, 10, 0),
            ("uzunfxpwmd", 0, 15, 0),
            ("xsrqllccpx", 1, 13, 0),
            ("xsrqllccpx", 12, 13, 0),
            ("xsrqllccpx", 7, 13, 0),
        ],
        "grid": (16, 16), "steps": 100, "rotation": 180,
    },
    {  # Level 4
        "sprites": [
            ("adbrqflmwi", 2, 9, 0),
            ("jgfvrvnkaz", 12, 5, 0),
            ("jgfvrvnkaz", 5, 5, 0),
            ("nkrtlkykwe", 7, 1, 0),
            ("syaipsfndp", 7, 0, 0),
            ("trurgcakbj", 12, 13, 0),
            ("trurgcakbj", 14, 10, 0),
            ("uzunfxpwmd", 0, 19, 0),
            ("xsrqllccpx", 2, 17, 0),
            ("xsrqllccpx", 16, 17, 0),
            ("xsrqllccpx", 8, 17, 0),
            ("xsrqllccpx", 12, 17, 0),
        ],
        "grid": (20, 20), "steps": 120, "rotation": 0,
    },
    {  # Level 5
        "sprites": [
            ("jgfvrvnkaz", 2, 9, 0),
            ("nkrtlkykwe", 5, 1, 0),
            ("nkrtlkykwe", 13, 1, 0),
            ("qwsmjdrvqj", 8, 5, 0),
            ("syaipsfndp", 5, 0, 0),
            ("syaipsfndp", 13, 0, 0),
            ("trurgcakbj", 7, 13, 0),
            ("untfxhpddv", 11, 9, 0),
            ("uzunfxpwmd", 0, 19, 0),
            ("uzunfxpwmd", 19, 0, 90),
            ("uzunfxpwmd", -1, 0, 90),
            ("xsrqllccpx", 17, 6, 270),
            ("xsrqllccpx", 2, 17, 0),
            ("xsrqllccpx", 6, 17, 0),
            ("xsrqllccpx", 12, 17, 0),
        ],
        "grid": (20, 20), "steps": 100, "rotation": 180,
    },
    {  # Level 6
        "sprites": [
            ("mdhkebfsmg", 14, 4, 0),
            ("nkrtlkykwe", 9, 1, 0),
            ("qwsmjdrvqj", 9, 5, 0),
            ("syaipsfndp", 9, 0, 0),
            ("uzunfxpwmd", 0, 19, 0),
            ("uzunfxpwmd", 19, 0, 90),
            ("uzunfxpwmd", 0, 0, 90),
            ("vkwijvqdla", 9, 14, 0),
            ("xsrqllccpx", 17, 9, 270),
            ("xsrqllccpx", 8, 17, 0),
            ("xsrqllccpx", 1, 11, 90),
            ("xsrqllccpx", 1, 6, 90),
            ("zgsbadjnjn", 7, 10, 0),
        ],
        "grid": (20, 20), "steps": 120, "rotation": 0,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_cells(name: str, x: int, y: int, rot: int = 0) -> List[Tuple[int, int]]:
    """Get world cells with pixel >= 0."""
    pixels = SPRITE_PIXELS.get(name)
    if not pixels:
        return []
    arr = np.array(pixels)
    if rot == 90:
        arr = np.rot90(arr, k=-1)
    elif rot == 180:
        arr = np.rot90(arr, k=-2)
    elif rot == 270:
        arr = np.rot90(arr, k=-3)
    cells = []
    for py in range(arr.shape[0]):
        for px in range(arr.shape[1]):
            if arr[py, px] >= 0:
                cells.append((x + px, y + py))
    return cells


def get_color4_cells(name: str, x: int, y: int, rot: int = 0) -> List[Tuple[int, int]]:
    """Get cells with pixel == 4 (drip sources)."""
    pixels = SPRITE_PIXELS.get(name)
    if not pixels:
        return []
    arr = np.array(pixels)
    if rot == 90:
        arr = np.rot90(arr, k=-1)
    elif rot == 180:
        arr = np.rot90(arr, k=-2)
    elif rot == 270:
        arr = np.rot90(arr, k=-3)
    cells = []
    for py in range(arr.shape[0]):
        for px in range(arr.shape[1]):
            if arr[py, px] == 4:
                cells.append((x + px, y + py))
    return cells


def get_dims(name: str, rot: int = 0) -> Tuple[int, int]:
    """Get (width, height) after rotation."""
    pixels = SPRITE_PIXELS.get(name)
    if not pixels:
        return (0, 0)
    arr = np.array(pixels)
    if rot == 90:
        arr = np.rot90(arr, k=-1)
    elif rot == 180:
        arr = np.rot90(arr, k=-2)
    elif rot == 270:
        arr = np.rot90(arr, k=-3)
    return (arr.shape[1], arr.shape[0])


# ---------------------------------------------------------------------------
# Level state
# ---------------------------------------------------------------------------

class LevelState:
    def __init__(self, level_idx: int):
        lv = LEVELS[level_idx]
        self.grid = lv["grid"]
        self.steps = lv["steps"]
        self.rotation = lv["rotation"]
        self.level_idx = level_idx

        self.drip_sources = []
        self.initial_water = []
        self.pipes = []
        self.receptacles = []  # list of (set_of_cells, idx)
        self.danger_cells = set()

        recept_idx = 0
        pipe_idx = 0
        for name, x, y, rot in lv["sprites"]:
            tags = SPRITE_TAGS.get(name, [])

            if "syaipsfndp" in tags:
                for cx, cy in get_color4_cells(name, x, y, rot):
                    self.drip_sources.append((cx, cy))

            if "ksmzdcblcz" in tags or "hfjpeygkxy" in tags:
                ptype = "l-pipe" if "hfjpeygkxy" in tags else "straight"
                clickable = "sys_click" in tags
                w, h = get_dims(name, rot)
                self.pipes.append({
                    "name": name, "x": x, "y": y, "rot": rot,
                    "type": ptype, "clickable": clickable,
                    "w": w, "h": h, "idx": pipe_idx,
                })
                pipe_idx += 1

            if "nkrtlkykwe" in tags:
                self.initial_water.append((x, y))

            if "xsrqllccpx" in tags:
                cells = set(get_cells(name, x, y, rot))
                self.receptacles.append((cells, recept_idx))
                recept_idx += 1

            if "uzunfxpwmd" in tags:
                for cx, cy in get_cells(name, x, y, rot):
                    self.danger_cells.add((cx, cy))


# ---------------------------------------------------------------------------
# Water flow simulation
# ---------------------------------------------------------------------------

def simulate_pour(state: LevelState, pipe_positions: Optional[Dict[int, Tuple[int,int]]] = None) -> Dict:
    """
    Simulate a pour. Returns {filled: set, hit_danger: bool, all_filled: bool}
    pipe_positions: {pipe_index: (new_x, new_y)} overrides
    """
    gw, gh = state.grid

    # Build world map: (x,y) -> (tag, ref)
    # ref = pipe_idx for pipes, recept_idx for receptacles
    world = {}

    for pipe in state.pipes:
        i = pipe["idx"]
        px = pipe["x"]
        py = pipe["y"]
        if pipe_positions and i in pipe_positions:
            px, py = pipe_positions[i]
        cells = get_cells(pipe["name"], px, py, pipe["rot"])
        tag = "l-pipe" if pipe["type"] == "l-pipe" else "pipe"
        for cx, cy in cells:
            world[(cx, cy)] = (tag, i)

    recept_cell_map = {}  # (x,y) -> recept_idx
    for cells, ridx in state.receptacles:
        for cx, cy in cells:
            world[(cx, cy)] = ("receptacle", ridx)
            recept_cell_map[(cx, cy)] = ridx

    for cx, cy in state.danger_cells:
        world[(cx, cy)] = ("danger", None)

    # Initialize water
    water_cells = set()
    active = []

    for wx, wy in state.initial_water:
        water_cells.add((wx, wy))
        active.append((wx, wy, 0, 1))

    # Compute drip sources — update positions if pipes with syaipsfndp tag are moved
    drip_sources = list(state.drip_sources)  # copy original
    for pipe in state.pipes:
        if "syaipsfndp" in SPRITE_TAGS.get(pipe["name"], []):
            if pipe_positions and pipe["idx"] in pipe_positions:
                # Remove original drip positions from this pipe
                orig_drips = set(get_color4_cells(pipe["name"], pipe["x"], pipe["y"], pipe["rot"]))
                drip_sources = [d for d in drip_sources if d not in orig_drips]
                # Add new drip positions at moved location
                nx, ny = pipe_positions[pipe["idx"]]
                for cx, cy in get_color4_cells(pipe["name"], nx, ny, pipe["rot"]):
                    drip_sources.append((cx, cy))

    for sx, sy in drip_sources:
        below = (sx, sy + 1)
        if below not in water_cells and below not in world:
            water_cells.add(below)
            active.append((below[0], below[1], 0, 1))

    filled = set()
    hit_danger = False

    for step in range(500):
        if not active:
            break

        new_active = []
        for wx, wy, dx, dy in active:
            # Perpendicular offsets
            if dy != 0:
                adjx1, adjx2 = -1, 1
                adjy1, adjy2 = 0, 0
            else:
                adjx1, adjx2 = 0, 0
                adjy1, adjy2 = -1, 1
            perp = [(adjx1, adjy1), (adjx2, adjy2)]

            nx, ny = wx + dx, wy + dy

            # Check if next cell is already water
            if (nx, ny) in water_cells and (nx, ny) not in world:
                new_active.append((nx, ny, dx, dy))
                continue

            target = world.get((nx, ny))

            if target is None and (nx, ny) not in water_cells:
                # Empty — flow forward
                if 0 <= nx < gw and 0 <= ny < gh:
                    water_cells.add((nx, ny))
                    new_active.append((nx, ny, dx, dy))
                continue

            if target is None and (nx, ny) in water_cells:
                # Water already there
                new_active.append((nx, ny, dx, dy))
                continue

            tag, ref = target

            if tag == "pipe":
                for ax, ay in perp:
                    npos = (wx + ax, wy + ay)
                    # Game: only create new water if position is completely empty
                    if npos not in water_cells and npos not in world:
                        water_cells.add(npos)
                        new_active.append((npos[0], npos[1], dx, dy))
                continue

            if tag == "receptacle":
                ridx = ref
                rcells = state.receptacles[ridx][0]
                s1 = (wx + adjx1, wy + adjy1)
                s2 = (wx + adjx2, wy + adjy2)
                if s1 in rcells and s2 in rcells:
                    filled.add(ridx)
                else:
                    for ax, ay in perp:
                        npos = (wx + ax, wy + ay)
                        if npos not in water_cells and npos not in world:
                            water_cells.add(npos)
                            new_active.append((npos[0], npos[1], dx, dy))
                continue

            if tag == "l-pipe":
                s1 = (wx + adjx1, wy + adjy1)
                s2 = (wx + adjx2, wy + adjy2)
                s1_target = world.get(s1)
                s2_target = world.get(s2)
                s1_is_lp = s1_target and s1_target[0] == "l-pipe" and s1_target[1] == ref
                s2_is_lp = s2_target and s2_target[0] == "l-pipe" and s2_target[1] == ref
                s1_empty = s1 not in world and s1 not in water_cells
                s2_empty = s2 not in world and s2 not in water_cells

                if s1_is_lp and s2_empty:
                    ndx, ndy = dy, -dx
                    npos = (wx + ndx, wy + ndy)
                    if npos not in water_cells:
                        water_cells.add(npos)
                    new_active.append((npos[0], npos[1], ndx, ndy))

                if s2_is_lp and s1_empty:
                    ndx, ndy = -dy, dx
                    npos = (wx + ndx, wy + ndy)
                    if npos not in water_cells:
                        water_cells.add(npos)
                    new_active.append((npos[0], npos[1], ndx, ndy))

                if not (s1_is_lp or s2_is_lp):
                    # Neither side is l-pipe → splash
                    for ax, ay in perp:
                        npos = (wx + ax, wy + ay)
                        if npos not in water_cells and npos not in world:
                            water_cells.add(npos)
                            new_active.append((npos[0], npos[1], dx, dy))
                continue

            if tag == "danger":
                hit_danger = True
                continue

        # Deduplicate active drops to prevent exponential growth
        seen = set()
        deduped = []
        for item in new_active:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        active = deduped

    all_filled = len(filled) == len(state.receptacles) and not hit_danger
    return {"filled": filled, "hit_danger": hit_danger, "all_filled": all_filled}


# ---------------------------------------------------------------------------
# Analytical helper: pipe exit streams
# ---------------------------------------------------------------------------

def pipe_exits(px: int, w: int) -> Tuple[int, int]:
    """A horizontal pipe at x=px with width w produces exit streams at x=(px-1) and x=(px+w)."""
    return (px - 1, px + w)


def cup_center(recept_x: int, rot: int = 0) -> int:
    """The cup opening x-coordinate for a receptacle at position (rx, ry).
    For default rotation (0): cup opening is at rx+1.
    For rotation 90: cup opening is at... depends on rotation.
    """
    # xsrqllccpx pixels: [[11,-1,11],[11,11,11]]
    # The -1 pixel is at (1, 0) relative to sprite origin
    # After rotation, this changes
    if rot == 0:
        return recept_x + 1  # opening at x+1
    elif rot == 90:
        return recept_x  # rotated: opening position changes
    elif rot == 270:
        return recept_x + 1
    return recept_x + 1


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def compute_moves(pipe, from_x, from_y, to_x, to_y):
    """Compute CLICK + arrow action sequence to move a pipe."""
    cx = from_x + pipe["w"] // 2
    cy = from_y + pipe["h"] // 2
    actions = [f"CLICK({cx},{cy})"]
    ddx = to_x - from_x
    ddy = to_y - from_y
    if ddx > 0:
        actions.extend(["RIGHT"] * ddx)
    elif ddx < 0:
        actions.extend(["LEFT"] * (-ddx))
    if ddy > 0:
        actions.extend(["DOWN"] * ddy)
    elif ddy < 0:
        actions.extend(["UP"] * (-ddy))
    return actions


def solve_level(level_idx: int, verbose: bool = True) -> Optional[Dict]:
    """Solve a level by searching pipe positions."""
    state = LevelState(level_idx)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Level {level_idx + 1} (grid={state.grid}, rot={state.rotation}, steps={state.steps})")
        print(f"  Drip sources: {state.drip_sources}")
        print(f"  Pipes: {len(state.pipes)}")
        for p in state.pipes:
            print(f"    [{p['idx']}] {p['name']} ({p['w']}x{p['h']}) at ({p['x']},{p['y']}) "
                  f"type={p['type']} click={p['clickable']}")
        print(f"  Receptacles: {len(state.receptacles)}")
        for cells, ridx in state.receptacles:
            print(f"    [{ridx}] cells={sorted(cells)[:6]}")
        print(f"  Danger cells: {len(state.danger_cells)}")

    # Test default positions first
    result = simulate_pour(state)
    if verbose:
        print(f"\n  Default: filled={result['filled']}/{len(state.receptacles)}, danger={result['hit_danger']}")
    if result["all_filled"]:
        return {"level": level_idx, "actions": ["SELECT"], "verified": True}

    # Identify movable pipes
    movable = [p for p in state.pipes if p["clickable"]]
    if verbose:
        print(f"  Movable pipes: {len(movable)}")

    gw, gh = state.grid

    # Generate candidate positions — all valid positions within grid
    def gen_candidates(pipe):
        w, h = pipe["w"], pipe["h"]
        cands = []
        for y in range(3, gh - h + 1):
            for x in range(-1, gw - w + 2):
                # Check no overlap with receptacle bounding box (+1 buffer)
                valid = True
                for rcells, _ in state.receptacles:
                    if not rcells:
                        continue
                    rxs = [c[0] for c in rcells]
                    rys = [c[1] for c in rcells]
                    rx1, rx2 = min(rxs), max(rxs)
                    ry1, ry2 = min(rys), max(rys)
                    if (x < rx2 + 2 and x + w > rx1 - 1 and
                        y < ry2 + 2 and y + h > ry1 - 1):
                        valid = False
                        break
                if valid:
                    cands.append((x, y))
        return cands

    # Full search: try all combinations
    if len(movable) == 1:
        p = movable[0]
        for x, y in gen_candidates(p):
            pp = {p["idx"]: (x, y)}
            res = simulate_pour(state, pp)
            if res["all_filled"]:
                moves = compute_moves(p, p["x"], p["y"], x, y)
                if verbose:
                    print(f"  SOLVED! Pipe [{p['idx']}] -> ({x},{y})")
                return {"level": level_idx, "actions": moves + ["SELECT"], "verified": True}

    elif len(movable) >= 2:
        # For multi-pipe: first find which positions produce any fills (ignoring danger)
        # Then try combinations
        pipe_useful = {}
        for p in movable:
            useful = []
            for x, y in gen_candidates(p):
                pp = {p["idx"]: (x, y)}
                res = simulate_pour(state, pp)
                nfilled = len(res["filled"])
                if nfilled > 0:
                    useful.append((x, y, nfilled, res["hit_danger"]))
            useful.sort(key=lambda t: -t[2])
            pipe_useful[p["idx"]] = useful[:100]  # keep top 100
            if verbose:
                print(f"    Pipe [{p['idx']}]: {len(useful)} useful positions")
                if useful:
                    print(f"      Top: {useful[:5]}")

        # Also include positions that DON'T interfere (useful for "get out of the way" pipes)
        # For pipes that block existing streams, moving them clears the path
        for p in movable:
            # Include current position and a few "safe" positions
            if (p["x"], p["y"], 0, False) not in pipe_useful.get(p["idx"], []):
                pipe_useful.setdefault(p["idx"], []).append((p["x"], p["y"], 0, False))

        if len(movable) == 2:
            p0, p1 = movable
            # Also try each pipe at its original position (only moving the other)
            all_combos = []
            for x0, y0, _, _ in pipe_useful[p0["idx"]]:
                for x1, y1, _, _ in pipe_useful[p1["idx"]]:
                    all_combos.append(((x0, y0), (x1, y1)))
            # Also try original positions
            for x1, y1, _, _ in pipe_useful[p1["idx"]]:
                all_combos.append(((p0["x"], p0["y"]), (x1, y1)))
            for x0, y0, _, _ in pipe_useful[p0["idx"]]:
                all_combos.append(((x0, y0), (p1["x"], p1["y"])))

            if verbose:
                print(f"  Trying {len(all_combos)} combinations...")
            for (x0, y0), (x1, y1) in all_combos:
                pp = {p0["idx"]: (x0, y0), p1["idx"]: (x1, y1)}
                res = simulate_pour(state, pp)
                if res["all_filled"]:
                    all_moves = []
                    for p in movable:
                        nx, ny = pp[p["idx"]]
                        if nx != p["x"] or ny != p["y"]:
                            all_moves.extend(compute_moves(p, p["x"], p["y"], nx, ny))
                    if verbose:
                        print(f"  SOLVED! {pp}")
                    return {"level": level_idx, "actions": all_moves + ["SELECT"], "verified": True}

        elif len(movable) == 3:
            p0, p1, p2 = movable
            # Try all three-way combinations of top positions
            cands0 = [(x, y) for x, y, _, _ in pipe_useful[p0["idx"]][:30]]
            cands1 = [(x, y) for x, y, _, _ in pipe_useful[p1["idx"]][:30]]
            cands2 = [(x, y) for x, y, _, _ in pipe_useful[p2["idx"]][:30]]
            # Add original positions
            cands0.append((p0["x"], p0["y"]))
            cands1.append((p1["x"], p1["y"]))
            cands2.append((p2["x"], p2["y"]))

            if verbose:
                print(f"  Trying {len(cands0)}x{len(cands1)}x{len(cands2)} = "
                      f"{len(cands0)*len(cands1)*len(cands2)} combinations...")
            for x0, y0 in cands0:
                for x1, y1 in cands1:
                    for x2, y2 in cands2:
                        pp = {p0["idx"]: (x0, y0), p1["idx"]: (x1, y1), p2["idx"]: (x2, y2)}
                        res = simulate_pour(state, pp)
                        if res["all_filled"]:
                            all_moves = []
                            for p in movable:
                                nx, ny = pp[p["idx"]]
                                if nx != p["x"] or ny != p["y"]:
                                    all_moves.extend(compute_moves(p, p["x"], p["y"], nx, ny))
                            if verbose:
                                print(f"  SOLVED! {pp}")
                            return {"level": level_idx, "actions": all_moves + ["SELECT"], "verified": True}

        elif len(movable) >= 4:
            # Too many pipes for brute force. Try analytical approach:
            # Move each pipe individually to see which fills receptacles
            # Then try pairs with others at default
            if verbose:
                print("  4+ pipes: trying individual + pairs...")

            # Individual pipe search with all others at default
            for p in movable:
                for x, y in gen_candidates(p):
                    pp = {p["idx"]: (x, y)}
                    res = simulate_pour(state, pp)
                    if res["all_filled"]:
                        moves = compute_moves(p, p["x"], p["y"], x, y)
                        if verbose:
                            print(f"  SOLVED with single pipe [{p['idx']}] -> ({x},{y})!")
                        return {"level": level_idx, "actions": moves + ["SELECT"], "verified": True}

            # Pair search: try every pair of movable pipes
            for i in range(len(movable)):
                for j in range(i + 1, len(movable)):
                    pi, pj = movable[i], movable[j]
                    ci = [(x, y) for x, y, _, _ in pipe_useful.get(pi["idx"], [])[:20]]
                    cj = [(x, y) for x, y, _, _ in pipe_useful.get(pj["idx"], [])[:20]]
                    ci.append((pi["x"], pi["y"]))
                    cj.append((pj["x"], pj["y"]))
                    for xi, yi in ci:
                        for xj, yj in cj:
                            pp = {pi["idx"]: (xi, yi), pj["idx"]: (xj, yj)}
                            res = simulate_pour(state, pp)
                            if res["all_filled"]:
                                all_moves = []
                                for p in [pi, pj]:
                                    nx, ny = pp[p["idx"]]
                                    if nx != p["x"] or ny != p["y"]:
                                        all_moves.extend(compute_moves(p, p["x"], p["y"], nx, ny))
                                if verbose:
                                    print(f"  SOLVED with 2 pipes! {pp}")
                                return {"level": level_idx, "actions": all_moves + ["SELECT"], "verified": True}

            # Triple search
            for i in range(len(movable)):
                for j in range(i+1, len(movable)):
                    for k in range(j+1, len(movable)):
                        pi, pj, pk = movable[i], movable[j], movable[k]
                        ci = [(x,y) for x,y,_,_ in pipe_useful.get(pi["idx"],[])[:10]]
                        cj = [(x,y) for x,y,_,_ in pipe_useful.get(pj["idx"],[])[:10]]
                        ck = [(x,y) for x,y,_,_ in pipe_useful.get(pk["idx"],[])[:10]]
                        ci.append((pi["x"], pi["y"]))
                        cj.append((pj["x"], pj["y"]))
                        ck.append((pk["x"], pk["y"]))
                        for xi,yi in ci:
                            for xj,yj in cj:
                                for xk,yk in ck:
                                    pp = {pi["idx"]:(xi,yi), pj["idx"]:(xj,yj), pk["idx"]:(xk,yk)}
                                    res = simulate_pour(state, pp)
                                    if res["all_filled"]:
                                        all_moves = []
                                        for p in [pi, pj, pk]:
                                            nx, ny = pp[p["idx"]]
                                            if nx != p["x"] or ny != p["y"]:
                                                all_moves.extend(compute_moves(p, p["x"], p["y"], nx, ny))
                                        if verbose:
                                            print(f"  SOLVED with 3 pipes! {pp}")
                                        return {"level": level_idx, "actions": all_moves + ["SELECT"], "verified": True}

    if verbose:
        print(f"  Level {level_idx + 1}: NO SOLUTION FOUND")
    return None


# ---------------------------------------------------------------------------
# Display rotation transforms for live play
# ---------------------------------------------------------------------------

def transform_action(action: str, k: int) -> str:
    if k == 0:
        return action
    m = {
        1: {"UP": "LEFT", "DOWN": "RIGHT", "LEFT": "DOWN", "RIGHT": "UP"},
        2: {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"},
        3: {"UP": "RIGHT", "DOWN": "LEFT", "LEFT": "UP", "RIGHT": "DOWN"},
    }
    return m.get(k, {}).get(action, action)


def grid_to_display_coords(gx: int, gy: int, k: int) -> Tuple[int, int]:
    """Inverse of udeubouzyp: grid coords -> display coords."""
    if k == 0:
        return (gx, gy)
    elif k == 1:
        return (gy, 63 - gx)
    elif k == 2:
        return (63 - gx, 63 - gy)
    else:
        return (63 - gy, gx)


def generate_sdk_actions(solution: Dict, state: LevelState) -> List[Dict]:
    """Convert solution actions to SDK format with rotation transforms."""
    k = state.rotation // 90 % 4
    gw, gh = state.grid
    scale = 64 // gw  # pixels per grid cell

    sdk = []
    for action in solution["actions"]:
        if action.startswith("CLICK("):
            coords = action[6:-1].split(",")
            gx, gy = int(coords[0]), int(coords[1])
            # Convert grid to pixel coords (center of cell)
            px = gx * scale + scale // 2
            py = gy * scale + scale // 2
            # Apply rotation inverse
            dx, dy = grid_to_display_coords(px, py, k)
            sdk.append({"action": "CLICK", "x": dx, "y": dy})
        elif action == "SELECT":
            sdk.append({"action": "SELECT"})
        else:
            display_action = transform_action(action, k)
            sdk.append({"action": display_action})
    return sdk


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("sp80 Gravity Pipes — Simulator & Solver")
    print("=" * 60)

    all_solutions = {}
    for level_idx in range(6):
        sol = solve_level(level_idx, verbose=True)
        if sol:
            all_solutions[level_idx] = sol
            state = LevelState(level_idx)
            sdk = generate_sdk_actions(sol, state)
            print(f"\n  SDK actions ({len(sdk)}):")
            for a in sdk:
                print(f"    {a}")
        else:
            print(f"\n  Level {level_idx + 1}: UNSOLVED")

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(all_solutions)}/6 levels solved")
    for idx, sol in sorted(all_solutions.items()):
        n = len(sol["actions"])
        print(f"  Level {idx+1}: {n} game actions")
        print(f"    {sol['actions']}")

    return all_solutions


if __name__ == "__main__":
    main()
