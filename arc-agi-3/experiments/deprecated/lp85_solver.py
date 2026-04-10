#!/usr/bin/env python3
"""
lp85 Ring Puzzle Solver

Parses the lp85.py game source code and computes optimal button-click sequences
to solve each level. Uses BFS over the joint positions of all goal sprites on
the ring structures.

Key mechanic: clicking a position fires ALL button sprites at that position.
Multiple rings can rotate simultaneously from a single click if their buttons
are stacked at the same (x,y).

Usage:
    python lp85_solver.py [--level N] [--source PATH]

If --level is omitted, all levels are analyzed and solved.
"""

import ast
import math
import re
import sys
import argparse
from collections import deque
from itertools import permutations
from typing import Dict, List, Optional, Tuple, Set, FrozenSet


# ---------------------------------------------------------------------------
# 1. Parse game source
# ---------------------------------------------------------------------------

GAME_SOURCE_DEFAULT = (
    "/Users/dennispalatov/repos/shared-context/environment_files/"
    "lp85/305b61c3/lp85.py"
)

COORD_MULT = 3  # crxpafuiwp


def parse_source(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def extract_sprite_tags(source: str) -> Dict[str, List[str]]:
    """Return {sprite_name: [tags]} from the top-level `sprites = {...}` dict."""
    tag_map: Dict[str, List[str]] = {}
    pattern = re.compile(
        r'"(\w+)":\s*Sprite\([^)]*?tags=\[([^\]]*)\]',
        re.DOTALL,
    )
    for m in pattern.finditer(source):
        name = m.group(1)
        tags_raw = m.group(2)
        tags = [t.strip().strip('"').strip("'") for t in tags_raw.split(",") if t.strip()]
        tag_map[name] = tags
    return tag_map


# --------------- extract levels list ---------------------------------------

class LevelSprite:
    """A sprite placement in a level."""
    def __init__(self, sprite_name: str, x: int, y: int, tags: List[str]):
        self.sprite_name = sprite_name
        self.x = x
        self.y = y
        self.tags = tags

    def __repr__(self):
        return f"LevelSprite({self.sprite_name}, x={self.x}, y={self.y}, tags={self.tags})"


class LevelData:
    """Parsed level."""
    def __init__(self, index: int, level_name: str, grid_size: Tuple[int, int],
                 step_counter: int, sprites: List[LevelSprite]):
        self.index = index
        self.level_name = level_name
        self.grid_w, self.grid_h = grid_size
        self.step_counter = step_counter
        self.sprites = sprites

    def sprites_by_tag(self, tag: str) -> List[LevelSprite]:
        return [s for s in self.sprites if tag in s.tags]


def extract_levels(source: str, tag_map: Dict[str, List[str]]) -> List[LevelData]:
    """Parse the levels = [...] list from the source."""
    levels: List[LevelData] = []
    level_pattern = re.compile(
        r'#\s*Level\s+(\d+)\s*\n\s*Level\(\s*sprites=\[(.*?)\],\s*'
        r'grid_size=\((\d+),\s*(\d+)\),\s*'
        r'data=\{(.*?)\},\s*\)',
        re.DOTALL,
    )

    for m in level_pattern.finditer(source):
        level_num = int(m.group(1))
        sprites_block = m.group(2)
        gw, gh = int(m.group(3)), int(m.group(4))
        data_block = m.group(5)

        level_name = ""
        step_counter = 0
        name_m = re.search(r'"level_name":\s*"(\w+)"', data_block)
        if name_m:
            level_name = name_m.group(1)
        step_m = re.search(r'"StepCounter":\s*(\d+)', data_block)
        if step_m:
            step_counter = int(step_m.group(1))

        sprite_placements: List[LevelSprite] = []
        sp_pattern = re.compile(
            r'sprites\["(\w+)"\]\.clone\(\)\.set_position\((\d+),\s*(\d+)\)'
        )
        for sp_m in sp_pattern.finditer(sprites_block):
            sname = sp_m.group(1)
            sx, sy = int(sp_m.group(2)), int(sp_m.group(3))
            tags = tag_map.get(sname, [])
            sprite_placements.append(LevelSprite(sname, sx, sy, tags))

        levels.append(LevelData(level_num, level_name, (gw, gh), step_counter, sprite_placements))

    levels.sort(key=lambda lv: lv.index)
    return levels


def extract_ring_maps(source: str) -> Dict[str, Dict[str, List[List[int]]]]:
    """Extract the izutyjcpih dict by finding and evaluating it."""
    start_match = re.search(r'^izutyjcpih\s*=\s*\{', source, re.MULTILINE)
    if not start_match:
        raise ValueError("Could not find izutyjcpih in source")

    brace_start = source.index('{', start_match.start())
    depth = 0
    end_pos = brace_start
    for i in range(brace_start, len(source)):
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                end_pos = i + 1
                break

    dict_str = source[brace_start:end_pos]
    ring_maps = ast.literal_eval(dict_str)
    return ring_maps


# ---------------------------------------------------------------------------
# 2. Ring data structures
# ---------------------------------------------------------------------------

class RingPosition:
    """A position number on a ring, with its (row, col) in the ring map."""
    def __init__(self, pos_num: int, row: int, col: int):
        self.pos_num = pos_num
        self.row = row
        self.col = col

    @property
    def sprite_x(self) -> int:
        return self.col * COORD_MULT

    @property
    def sprite_y(self) -> int:
        return self.row * COORD_MULT


class Ring:
    """A ring: ordered list of positions that rotate."""
    def __init__(self, name: str, positions: Dict[int, RingPosition], max_pos: int):
        self.name = name
        self.positions = positions
        self.max_pos = max_pos
        self.size = max_pos

    def next_pos(self, pos_num: int, direction: str) -> int:
        """Return next position number after rotation."""
        if direction == "R":
            # R (kiofvrbmju=True): pos -> pos+1, max -> 1
            return 1 if pos_num == self.max_pos else pos_num + 1
        else:
            # L (kiofvrbmju=False): pos -> pos-1, 1 -> max
            return self.max_pos if pos_num == 1 else pos_num - 1

    def coord_for_pos(self, pos_num: int) -> Tuple[int, int]:
        """Return (sprite_x, sprite_y) for a position number."""
        rp = self.positions[pos_num]
        return (rp.sprite_x, rp.sprite_y)


def build_rings(ring_maps: Dict[str, Dict[str, List[List[int]]]],
                level_name: str) -> Dict[str, Ring]:
    """Build Ring objects for a given level name."""
    if level_name not in ring_maps:
        return {}
    rings: Dict[str, Ring] = {}
    for ring_name, grid in ring_maps[level_name].items():
        positions: Dict[int, RingPosition] = {}
        max_pos = 0
        for row_idx, row in enumerate(grid):
            for col_idx, val in enumerate(row):
                if val != -1:
                    positions[val] = RingPosition(val, row_idx, col_idx)
                    if val > max_pos:
                        max_pos = val
        rings[ring_name] = Ring(ring_name, positions, max_pos)
    return rings


# ---------------------------------------------------------------------------
# 3. Button groups - what rings rotate together per click
# ---------------------------------------------------------------------------

class ButtonGroup:
    """A physical click position that triggers one or more ring rotations."""
    def __init__(self, display_x: int, display_y: int,
                 rotations: List[Tuple[str, str]],
                 sprite_x: int, sprite_y: int):
        self.display_x = display_x
        self.display_y = display_y
        self.rotations = rotations  # [(ring_name, direction), ...]
        self.sprite_x = sprite_x
        self.sprite_y = sprite_y
        # Descriptive label
        if len(rotations) == 1:
            rn, d = rotations[0]
            self.label = f"{rn}_{d}"
        else:
            parts = []
            for rn, d in rotations:
                parts.append(f"{rn}_{d}")
            self.label = "+".join(parts)

    def __repr__(self):
        return f"ButtonGroup({self.label} @ ({self.display_x},{self.display_y}))"


def build_button_groups(level: LevelData, rings: Dict[str, Ring]) -> List[ButtonGroup]:
    """
    Build the list of distinct button click positions and what they do.

    The game removes sys_click from sprites at duplicate positions, but processes
    ALL button sprites at a clicked position. So we group buttons by position.

    Only include buttons for rings that exist in this level's ring maps.
    """
    grid_w, grid_h = level.grid_w, level.grid_h

    # Group button sprites by position
    pos_buttons: Dict[Tuple[int, int], List[Tuple[str, str]]] = {}  # (x,y) -> [(ring, dir)]

    for s in level.sprites:
        for tag in s.tags:
            if tag.startswith("button_"):
                parts = tag.split("_")
                if len(parts) == 3:
                    ring_name = parts[1]
                    direction = parts[2]
                    # Only include if this ring exists in the level's ring map
                    if ring_name in rings:
                        key = (s.x, s.y)
                        if key not in pos_buttons:
                            pos_buttons[key] = []
                        # Avoid duplicates
                        entry = (ring_name, direction)
                        if entry not in pos_buttons[key]:
                            pos_buttons[key].append(entry)

    groups = []
    for (sx, sy), rotations in pos_buttons.items():
        dx, dy = sprite_to_display(sx, sy, grid_w, grid_h)
        groups.append(ButtonGroup(dx, dy, sorted(rotations), sx, sy))

    return groups


# ---------------------------------------------------------------------------
# 4. Coordinate helpers
# ---------------------------------------------------------------------------

def sprite_to_display(sprite_x: int, sprite_y: int,
                      grid_w: int, grid_h: int) -> Tuple[int, int]:
    """Convert sprite grid coordinates to 64x64 display coordinates."""
    offset_x = (64 - grid_w) // 2
    offset_y = (64 - grid_h) // 2
    return (sprite_x + offset_x, sprite_y + offset_y)


def find_intersections(rings: Dict[str, Ring]) -> Dict[Tuple[int, int], List[Tuple[str, int]]]:
    """Find grid positions shared between rings."""
    coord_to_ring: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
    for rname, ring in rings.items():
        for pos_num, rp in ring.positions.items():
            key = (rp.sprite_x, rp.sprite_y)
            if key not in coord_to_ring:
                coord_to_ring[key] = []
            coord_to_ring[key].append((rname, pos_num))
    return {k: v for k, v in coord_to_ring.items() if len(v) >= 2}


def find_ring_position(x: int, y: int, rings: Dict[str, Ring]
                       ) -> Optional[Tuple[str, int]]:
    """Find which ring and position number a coordinate corresponds to."""
    for rname, ring in rings.items():
        for pos_num, rp in ring.positions.items():
            if rp.sprite_x == x and rp.sprite_y == y:
                return (rname, pos_num)
    return None


def build_coord_to_rings(rings: Dict[str, Ring]) -> Dict[Tuple[int, int], List[Tuple[str, int]]]:
    """Map every (sprite_x, sprite_y) to all (ring_name, pos_num) sharing that coord."""
    result: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
    for rname, ring in rings.items():
        for pos_num, rp in ring.positions.items():
            key = (rp.sprite_x, rp.sprite_y)
            if key not in result:
                result[key] = []
            result[key].append((rname, pos_num))
    return result


# ---------------------------------------------------------------------------
# 5. BFS solver with button groups
# ---------------------------------------------------------------------------

class GoalInfo:
    """Info about a single goal piece."""
    def __init__(self, tag: str, start_ring: str, start_pos: int,
                 target_ring: str, target_pos: int):
        self.tag = tag
        self.start_ring = start_ring
        self.start_pos = start_pos
        self.target_ring = target_ring
        self.target_pos = target_pos


# State: tuple of (sprite_x, sprite_y) for each goal
# We use coordinates as state since the game moves sprites by coordinates,
# and ring membership is determined by current coordinates.

def apply_button_group(
    state: Tuple[Tuple[int, int], ...],
    button: ButtonGroup,
    rings: Dict[str, Ring],
    coord_to_rings: Dict[Tuple[int, int], List[Tuple[str, int]]],
) -> Tuple[Tuple[int, int], ...]:
    """
    Apply a button click to the state. The button triggers rotations of
    one or more rings. For each ring being rotated:
    - Find all positions on that ring
    - For each goal whose coordinates match a position on that ring,
      compute the new position
    - Move the goal to the new coordinates

    Important: the game first collects all (sprite, new_dest) pairs, then
    moves them all. So we need to compute all moves before applying.
    """
    # Collect all ring rotations from this button
    ring_rotations = button.rotations  # [(ring_name, direction), ...]

    # For each rotation, find which goals are affected and their destinations
    # A goal at coordinates (x,y) is affected by ring R if (x,y) is a position on R
    moves: Dict[int, Tuple[int, int]] = {}  # goal_index -> new_coords

    for ring_name, direction in ring_rotations:
        ring = rings[ring_name]
        for goal_idx, (gx, gy) in enumerate(state):
            # Check if this goal is on this ring
            ring_entries = coord_to_rings.get((gx, gy), [])
            for rn, pn in ring_entries:
                if rn == ring_name:
                    # This goal is at position pn on this ring
                    new_pn = ring.next_pos(pn, direction)
                    new_coord = ring.coord_for_pos(new_pn)
                    # If already moved by another ring in this group, the last one wins
                    # (but this shouldn't happen in well-designed puzzles)
                    moves[goal_idx] = new_coord
                    break

    # Apply moves
    new_state = list(state)
    for idx, new_coord in moves.items():
        new_state[idx] = new_coord

    return tuple(new_state)


def bfs_solve(
    goals: List[GoalInfo],
    rings: Dict[str, Ring],
    button_groups: List[ButtonGroup],
    max_depth: int = 200,
) -> Optional[List[int]]:
    """
    BFS to find optimal sequence of button group clicks.
    Returns list of button group indices, or None if no solution found.
    """
    coord_to_rings = build_coord_to_rings(rings)

    # Build target coordinates
    target_state = tuple(
        rings[g.target_ring].coord_for_pos(g.target_pos) for g in goals
    )

    # Initial state as coordinates
    init_state = tuple(
        rings[g.start_ring].coord_for_pos(g.start_pos) for g in goals
    )

    if init_state == target_state:
        return []

    visited: Set[Tuple[Tuple[int, int], ...]] = set()
    visited.add(init_state)

    # Queue: (state, path_of_button_indices)
    queue: deque = deque()
    queue.append((init_state, []))

    while queue:
        state, path = queue.popleft()
        if len(path) >= max_depth:
            continue

        for btn_idx, button in enumerate(button_groups):
            new_state = apply_button_group(state, button, rings, coord_to_rings)

            if new_state == target_state:
                return path + [btn_idx]

            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [btn_idx]))

    return None


# ---------------------------------------------------------------------------
# 6. Compress and display solution
# ---------------------------------------------------------------------------

def compress_solution(
    solution: List[int],
    button_groups: List[ButtonGroup],
) -> List[Tuple[ButtonGroup, int]]:
    """Group consecutive identical button clicks. Returns [(ButtonGroup, count), ...]."""
    if not solution:
        return []
    groups = []
    curr = solution[0]
    count = 1
    for idx in solution[1:]:
        if idx == curr:
            count += 1
        else:
            groups.append((button_groups[curr], count))
            curr = idx
            count = 1
    groups.append((button_groups[curr], count))
    return groups


# ---------------------------------------------------------------------------
# 7. Main analysis and solving
# ---------------------------------------------------------------------------

def analyze_level(level: LevelData, ring_maps: Dict, verbose: bool = True) -> Optional[dict]:
    """Analyze and solve a single level."""
    rings = build_rings(ring_maps, level.level_name)
    if not rings:
        if verbose:
            print(f"  No ring maps found for level name '{level.level_name}'")
        return None

    intersections = find_intersections(rings)
    button_groups = build_button_groups(level, rings)

    # Print ring info
    ring_info = ", ".join(f"{rn}({r.size})" for rn, r in sorted(rings.items()))
    if verbose:
        print(f"  Rings: {ring_info}")
        if intersections:
            for coord, ring_list in sorted(intersections.items()):
                ring_strs = [f"{rn}:{pn}" for rn, pn in ring_list]
                print(f"  Intersection at sprite({coord[0]},{coord[1]}): {', '.join(ring_strs)}")
        print(f"  Button groups ({len(button_groups)}):")
        for bg in button_groups:
            print(f"    {bg.label} -> click({bg.display_x},{bg.display_y})")

    # Find goal sprites and their targets
    goal_sprites = level.sprites_by_tag("goal")
    check_markers = level.sprites_by_tag("bghvgbtwcb")
    goalo_sprites = level.sprites_by_tag("goal-o")
    check_markers_o = level.sprites_by_tag("fdgmtkfrxl")

    if verbose:
        print(f"  Goals: {len(goal_sprites)} goal, {len(goalo_sprites)} goal-o")
        print(f"  Markers: {len(check_markers)} bghvgbtwcb, {len(check_markers_o)} fdgmtkfrxl")

    # Map goals to ring positions
    goal_positions = []
    for gs in goal_sprites:
        rp = find_ring_position(gs.x, gs.y, rings)
        if rp:
            goal_positions.append((gs, rp[0], rp[1], "goal"))
        elif verbose:
            print(f"  WARNING: goal at ({gs.x},{gs.y}) not on any ring!")

    for gs in goalo_sprites:
        rp = find_ring_position(gs.x, gs.y, rings)
        if rp:
            goal_positions.append((gs, rp[0], rp[1], "goal-o"))
        elif verbose:
            print(f"  WARNING: goal-o at ({gs.x},{gs.y}) not on any ring!")

    # Map targets
    goal_targets = []
    for cm in check_markers:
        tx, ty = cm.x + 1, cm.y + 1
        rp = find_ring_position(tx, ty, rings)
        if rp:
            goal_targets.append((rp[0], rp[1], "goal"))
        elif verbose:
            print(f"  WARNING: goal target at ({tx},{ty}) not on any ring!")

    for cm in check_markers_o:
        tx, ty = cm.x + 1, cm.y + 1
        rp = find_ring_position(tx, ty, rings)
        if rp:
            goal_targets.append((rp[0], rp[1], "goal-o"))
        elif verbose:
            print(f"  WARNING: goal-o target at ({tx},{ty}) not on any ring!")

    # Match goals to targets by type, try permutations
    goal_type_goals = [gp for gp in goal_positions if gp[3] == "goal"]
    goal_type_targets = [gt for gt in goal_targets if gt[2] == "goal"]
    goalo_type_goals = [gp for gp in goal_positions if gp[3] == "goal-o"]
    goalo_type_targets = [gt for gt in goal_targets if gt[2] == "goal-o"]

    if len(goal_type_goals) != len(goal_type_targets):
        if verbose:
            print(f"  WARNING: Mismatch - {len(goal_type_goals)} goals vs {len(goal_type_targets)} targets")
    if len(goalo_type_goals) != len(goalo_type_targets):
        if verbose:
            print(f"  WARNING: Mismatch - {len(goalo_type_goals)} goal-o vs {len(goalo_type_targets)} targets")

    best_solution = None
    best_goals = None
    best_len = float('inf')

    g_perms = list(permutations(goal_type_targets))[:120]  # up to 5!
    o_perms = list(permutations(goalo_type_targets))[:120]
    if not g_perms:
        g_perms = [()]
    if not o_perms:
        o_perms = [()]

    max_d = min(level.step_counter, 200)

    for g_perm in g_perms:
        for o_perm in o_perms:
            g_t = list(g_perm) if g_perm else []
            o_t = list(o_perm) if o_perm else []
            if len(g_t) != len(goal_type_goals) or len(o_t) != len(goalo_type_goals):
                continue

            all_goals = []
            for (gs, grn, gpos, _), (trn, tpos, _) in zip(goal_type_goals, g_t):
                all_goals.append(GoalInfo("goal", grn, gpos, trn, tpos))
            for (gs, grn, gpos, _), (trn, tpos, _) in zip(goalo_type_goals, o_t):
                all_goals.append(GoalInfo("goal-o", grn, gpos, trn, tpos))

            if verbose and best_solution is None:
                for g in all_goals:
                    print(f"    {g.tag}: {g.start_ring}{g.start_pos} -> {g.target_ring}{g.target_pos}")

            solution = bfs_solve(all_goals, rings, button_groups, max_depth=max_d)
            if solution is not None and len(solution) < best_len:
                best_solution = solution
                best_goals = all_goals
                best_len = len(solution)
                if best_len <= 1:
                    break
        if best_solution is not None and best_len <= 1:
            break

    if best_solution is not None and verbose and best_goals is not None:
        # Show final assignment if different from first tried
        print(f"  Best goal assignment:")
        for g in best_goals:
            print(f"    {g.tag}: {g.start_ring}{g.start_pos} -> {g.target_ring}{g.target_pos}")

    if best_solution is None:
        if verbose:
            print("  NO SOLUTION FOUND within step limit!")
        return None

    # Compress and display
    compressed = compress_solution(best_solution, button_groups)
    if verbose:
        print(f"  Solution ({len(best_solution)} clicks):")
        for bg, count in compressed:
            print(f"    {bg.label} x{count} -> click({bg.display_x},{bg.display_y})")

    # Build full click sequence
    click_sequence = []
    for btn_idx in best_solution:
        bg = button_groups[btn_idx]
        click_sequence.append((bg.display_x, bg.display_y))

    if verbose:
        print(f"\n  Full click sequence ({len(click_sequence)} clicks):")
        for i, (cx, cy) in enumerate(click_sequence):
            print(f"    Step {i+1}: click({cx},{cy})")

    return {
        "level": level.index,
        "level_name": level.level_name,
        "solution_indices": best_solution,
        "compressed": compressed,
        "clicks": click_sequence,
        "goals": best_goals,
        "button_groups": button_groups,
    }


def main():
    parser = argparse.ArgumentParser(description="lp85 Ring Puzzle Solver")
    parser.add_argument("--level", type=int, default=None,
                        help="Level number to solve (1-indexed). If omitted, solve all.")
    parser.add_argument("--source", type=str, default=GAME_SOURCE_DEFAULT,
                        help="Path to lp85.py game source")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output")
    args = parser.parse_args()

    print("Parsing game source...")
    source = parse_source(args.source)

    print("Extracting sprite tags...")
    tag_map = extract_sprite_tags(source)
    print(f"  Found {len(tag_map)} sprite definitions")

    print("Extracting levels...")
    levels = extract_levels(source, tag_map)
    print(f"  Found {len(levels)} levels")

    print("Extracting ring maps...")
    ring_maps = extract_ring_maps(source)
    print(f"  Found ring maps for: {', '.join(sorted(ring_maps.keys()))}")

    print()

    if args.level is not None:
        target_levels = [lv for lv in levels if lv.index == args.level]
        if not target_levels:
            print(f"Level {args.level} not found!")
            sys.exit(1)
    else:
        target_levels = levels

    results = []
    for level in target_levels:
        print(f"{'='*60}")
        print(f"Level {level.index} ({level.level_name}):")
        print(f"  Grid: {level.grid_w}x{level.grid_h}, Steps: {level.step_counter}")
        result = analyze_level(level, ring_maps, verbose=not args.quiet)
        if result:
            results.append(result)
        print()

    # Summary
    if results:
        print(f"{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in results:
            compressed = r["compressed"]
            total = len(r["solution_indices"])
            desc_parts = []
            for bg, c in compressed:
                desc_parts.append(f"{bg.label}x{c}")
            print(f"  Level {r['level']} ({r['level_name']}): "
                  f"{total} clicks - {', '.join(desc_parts)}")


if __name__ == "__main__":
    main()
