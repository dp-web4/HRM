#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 sb26 Solver — Sprout

sb26 is a sequence-matching puzzle:
- Top row: destination slots with target colors (left to right = target sequence)
- Middle: frames with empty slots where items go
- Bottom: palette of colored items to arrange
- ACTION6: click to select/place items (click palette item, then click empty slot)
- ACTION5: submit arrangement for evaluation
- ACTION7: undo (free, no energy cost)

Strategy: observe grid → read colors → place items in order → submit.
"""
import sys, os, time, json
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")
import numpy as np
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from arcengine import GameAction
from arc_agi import Arcade


def get_frame(fd):
    grid = np.array(fd.frame)
    return grid[-1] if grid.ndim == 3 else grid


def scan_row_runs(grid, row_idx, exclude_colors):
    """Scan a single row for contiguous color runs.
    Returns list of (center_x, color, run_length) sorted by x."""
    row = grid[row_idx, :]
    runs = []
    c = 0
    while c < row.shape[0]:
        color = int(row[c])
        if color in exclude_colors:
            c += 1
            continue
        start = c
        while c < row.shape[0] and int(row[c]) == color:
            c += 1
        run_len = c - start
        if run_len >= 2:
            runs.append(((start + c - 1) // 2, color, run_len))
    return runs


def read_destinations(grid):
    """Read destination color sequence from row 1.
    Returns list of (center_x, color)."""
    bg = int(np.bincount(grid.flatten()).argmax())
    runs = scan_row_runs(grid, 1, {bg, 0, 5})
    # Only keep runs >= 3 wide (real destination borders)
    return [(x, c) for x, c, w in runs if w >= 3]


def read_palette(grid):
    """Read palette items from bottom area.
    Returns list of (center_x, center_y, color) sorted by x."""
    bg = int(np.bincount(grid.flatten()).argmax())
    exclude = {bg, 0, 2, 5, 8}
    # Try rows 57-60 to find palette items
    for row_idx in range(57, 61):
        runs = scan_row_runs(grid, row_idx, exclude)
        if runs:
            return [(x, row_idx, c) for x, c, w in runs if w >= 3]
    return []


def read_slots(grid):
    """Find empty slots (color 2 markers) in frame area y=15-50.
    Returns list of (center_x, center_y) sorted by x."""
    slots = []
    # Scan for color 2 pixels, group into slot centers
    for r in range(15, 50):
        for c in range(64):
            if int(grid[r, c]) == 2:
                if not any(abs(c - sx) < 4 and abs(r - sy) < 4 for sx, sy in slots):
                    slots.append((c, r))
    # Merge nearby
    merged = []
    for sx, sy in sorted(slots, key=lambda s: s[0]):
        found = False
        for i, (mx, my) in enumerate(merged):
            if abs(sx - mx) < 4 and abs(sy - my) < 4:
                merged[i] = ((mx + sx) // 2, (my + sy) // 2)
                found = True
                break
        if not found:
            merged.append((sx, sy))
    return sorted(merged, key=lambda s: s[0])


def place_and_submit(env, fd, placements, verbose=False):
    """Execute a sequence of (palette_pos, slot_pos) placements, then submit.
    Returns (fd, leveled_up)."""
    prev_levels = fd.levels_completed
    grid = get_frame(fd)

    for i, (pal_pos, slot_pos) in enumerate(placements):
        px, py = pal_pos
        sx, sy = slot_pos

        if verbose:
            color_at = int(grid[py, px]) if 0 <= py < 64 and 0 <= px < 64 else '?'
            print(f"  [{i}] click({px},{py}) c={color_at} → click({sx},{sy})")

        # Click palette item to select
        fd = env.step(GameAction.ACTION6, data={'x': int(px), 'y': int(py)})
        # Click slot to place
        fd = env.step(GameAction.ACTION6, data={'x': int(sx), 'y': int(sy)})
        grid = get_frame(fd)

    # Submit
    fd = env.step(GameAction.ACTION5)

    # Wait for animation
    for _ in range(50):
        if fd.levels_completed > prev_levels or fd.state.name in ("WON", "GAME_OVER"):
            break
        fd = env.step(GameAction.ACTION7)

    return fd, fd.levels_completed > prev_levels


def solve_level_smart(env, fd, verbose=False):
    """Read grid, match destination colors to palette, place in order."""
    grid = get_frame(fd)
    prev_levels = fd.levels_completed

    dests = read_destinations(grid)
    palette = read_palette(grid)
    slots = read_slots(grid)

    if verbose:
        print(f"  Dests: {dests}")
        print(f"  Palette: {palette}")
        print(f"  Slots: {slots}")

    if not dests or not palette or not slots:
        return fd, False

    dest_colors = [c for _, c in dests]

    # Build color → palette position lookup
    pal_by_color = {}
    for px, py, pc in palette:
        if pc not in pal_by_color:
            pal_by_color[pc] = (px, py)

    # Plan placements: for each destination color, find matching palette item
    placements = []
    used_colors = set()
    for i, target in enumerate(dest_colors):
        if i >= len(slots):
            break
        if target in pal_by_color and target not in used_colors:
            placements.append((pal_by_color[target], slots[i]))
            used_colors.add(target)

    if len(placements) != len(dest_colors):
        if verbose:
            print(f"  Could only plan {len(placements)}/{len(dest_colors)} placements")

    if not placements:
        return fd, False

    return place_and_submit(env, fd, placements, verbose)


def solve_level_brute(arcade, game_id, verbose=False):
    """Try all permutations of palette items into slots."""
    env0 = arcade.make(game_id)
    fd0 = env0.reset()
    grid0 = get_frame(fd0)

    palette = read_palette(grid0)
    slots = read_slots(grid0)

    n = min(len(palette), len(slots))
    if n == 0:
        return None, False
    if n > 7:
        if verbose:
            print(f"  {n}! too many permutations")
        return None, False

    pal_positions = [(px, py) for px, py, _ in palette]
    pal_colors = [c for _, _, c in palette]

    total = 1
    for i in range(1, n + 1):
        total *= i
    if verbose:
        print(f"  Brute: {n} items ({pal_colors}), {total} perms, {len(slots)} slots")

    for idx, perm in enumerate(permutations(range(n))):
        # Fresh game instance per attempt
        env = arcade.make(game_id)
        fd = env.reset()
        grid = get_frame(fd)

        # Re-read positions (should be same but defensive)
        pal = read_palette(grid)
        slt = read_slots(grid)
        if len(pal) < n or len(slt) < n:
            continue

        # Build placement plan for this permutation
        placements = []
        for slot_idx, pal_idx in enumerate(perm):
            if slot_idx >= len(slt) or pal_idx >= len(pal):
                break
            placements.append(((pal[pal_idx][0], pal[pal_idx][1]), slt[slot_idx]))

        fd, leveled = place_and_submit(env, fd, placements)

        if leveled:
            actual = [pal_colors[i] for i in perm]
            if verbose:
                print(f"  ★ Perm {idx}/{total}: {actual} → Level {fd.levels_completed}!")
            return fd, True

    return None, False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE ARC-AGI-3 sb26 Solver — Sprout")
    print("=" * 60)

    arcade = Arcade()
    envs = arcade.get_environments()
    sb26 = [e for e in envs if "sb26" in (e.game_id if hasattr(e, "game_id") else str(e))]
    if not sb26:
        print("sb26 not found!")
        return

    game_id = sb26[0].game_id if hasattr(sb26[0], "game_id") else str(sb26[0])
    print(f"Game: {game_id}")

    # Attempt 1: Smart placement
    print("\n--- Smart placement ---")
    env = arcade.make(game_id)
    fd = env.reset()
    print(f"Actions: {[a.value if hasattr(a, 'value') else int(a) for a in fd.available_actions]}, win={fd.win_levels}")

    fd, success = solve_level_smart(env, fd, verbose=True)
    best_levels = fd.levels_completed

    if success:
        print(f"  ★ Smart placement worked! Level {fd.levels_completed}/{fd.win_levels}")
    else:
        print("  Smart placement failed, trying brute force...")

    # Attempt 2: Brute force
    if not success:
        print("\n--- Brute force ---")
        fd2, success2 = solve_level_brute(arcade, game_id, verbose=True)
        if fd2 and success2:
            best_levels = fd2.levels_completed
            fd = fd2

    # If we got level 1, try to keep going on subsequent levels
    if best_levels > 0 and fd.state.name not in ("WON", "GAME_OVER"):
        print(f"\n--- Continuing from level {best_levels} ---")
        for attempt in range(20):
            grid = get_frame(fd)
            dests = read_destinations(grid)
            palette = read_palette(grid)
            slots = read_slots(grid)
            if not dests or not palette or not slots:
                print(f"  Level {fd.levels_completed}: can't read grid")
                break

            fd, leveled = solve_level_smart(env, fd, verbose=args.verbose)
            if leveled:
                best_levels = fd.levels_completed
                print(f"  ★ Level {fd.levels_completed}/{fd.win_levels}!")
            else:
                print(f"  Level {fd.levels_completed}: smart failed, trying brute...")
                # Need fresh instance for brute force but we're mid-game
                # Try brute on a fresh instance instead
                fd3, ok3 = solve_level_brute(arcade, game_id, verbose=args.verbose)
                if fd3 and ok3:
                    best_levels = max(best_levels, fd3.levels_completed)
                    print(f"  ★ Brute: Level {fd3.levels_completed}/{fd3.win_levels}!")
                break

            if fd.state.name in ("WON", "GAME_OVER"):
                break

    print(f"\n{'='*60}")
    print(f"RESULT: {best_levels}/{fd.win_levels if fd else '?'} levels")

    # Store in membot
    try:
        import requests
        requests.post("http://localhost:8000/api/store",
            json={"text": f"ARC-AGI-3 sb26: scored {best_levels} levels. "
                  f"Sequence-matching puzzle: read dest colors at top, match palette items, "
                  f"place into frame slots left-to-right, submit with ACTION5."},
            timeout=2)
    except:
        pass


if __name__ == "__main__":
    main()
