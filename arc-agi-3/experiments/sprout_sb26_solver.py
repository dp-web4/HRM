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

Strategy: observe grid → read colors → place items → submit.
Learned from observation + code category analysis (not runtime code reading).
"""
import sys, os, time, json
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from arcengine import GameAction
from arc_agi import Arcade

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


def get_frame(fd):
    grid = np.array(fd.frame)
    return grid[-1] if grid.ndim == 3 else grid


def find_colored_blocks(grid, y_min, y_max, exclude_colors=None):
    """Find rectangular colored blocks in a horizontal band.
    Returns list of (x_center, y_center, color) sorted by x.
    Much simpler than flood fill — just scan for runs of non-bg color."""
    bg = int(np.bincount(grid.flatten()).argmax())
    exclude = set(exclude_colors or []) | {bg, 0}
    band = grid[y_min:y_max, :]

    # Find horizontal runs of each color
    blocks = []
    for r in range(band.shape[0]):
        c = 0
        while c < band.shape[1]:
            color = int(band[r, c])
            if color in exclude:
                c += 1
                continue
            # Find run length
            start = c
            while c < band.shape[1] and int(band[r, c]) == color:
                c += 1
            run_len = c - start
            if run_len >= 2:
                cx = (start + c - 1) // 2
                cy = r + y_min
                blocks.append((cx, cy, color, run_len))

    # Group into distinct objects: same color, nearby positions
    objects = []
    for cx, cy, color, size in sorted(blocks, key=lambda b: (b[2], b[0])):
        merged = False
        for i, (ox, oy, oc, os) in enumerate(objects):
            if oc == color and abs(cx - ox) < 6 and abs(cy - oy) < 6:
                # Merge — update center
                objects[i] = ((ox + cx) // 2, (oy + cy) // 2, color, os + size)
                merged = True
                break
        if not merged:
            objects.append((cx, cy, color, size))

    return sorted(objects, key=lambda o: o[0])


def analyze_sb26_grid(grid, verbose=False):
    """Analyze the sb26 grid to find destination colors, palette items, and frame slots."""
    h, w = grid.shape
    bg = int(np.bincount(grid.flatten()).argmax())

    if verbose:
        print(f"  Grid: {h}x{w}, bg={bg}")

    # Destinations: colored boxes at top (y=1-6), exclude bg(4), gray(5), black(0)
    dest_objects = find_colored_blocks(grid, 0, 10, exclude_colors={5})

    # Palette: colored items at bottom (y=55-63), exclude divider color 2 and gray 5
    palette_objects = find_colored_blocks(grid, 55, 63, exclude_colors={2, 5, 8})

    # Empty slots: color 2 dots in frame area (y=12-50), NOT the divider row
    slot_objects = find_colored_blocks(grid, 12, 50, exclude_colors={5, 8})

    return {
        "destinations": dest_objects,
        "palette": palette_objects,
        "slots": slot_objects,
    }


def read_destinations(grid, verbose=False):
    """Read destination color sequence from top of grid.
    Destinations are 6x6 colored border boxes in top 10 rows.
    Returns list of colors left-to-right."""
    bg = int(np.bincount(grid.flatten()).argmax())
    structural = {bg, 0, 5}  # bg, black, gray background

    # Scan row 1 (where borders are visible) for non-structural colors
    row = grid[1, :]
    runs = []
    c = 0
    while c < 64:
        color = int(row[c])
        if color in structural:
            c += 1
            continue
        start = c
        while c < 64 and int(row[c]) == color:
            c += 1
        run_len = c - start
        if run_len >= 3:  # Min width for a destination border
            center_x = (start + c - 1) // 2
            runs.append((center_x, color))

    if verbose:
        print(f"  Destinations: {runs}")
    return runs


def read_palette(grid, verbose=False):
    """Read palette items from bottom area (y > 55).
    Returns list of (center_x, center_y, color) left-to-right."""
    bg = int(np.bincount(grid.flatten()).argmax())
    structural = {bg, 0, 2, 5, 8}  # bg, black, slot indicator, gray, frame border

    # Palette items are colored blocks at y ~57-60
    # Scan a representative row (58)
    items = []
    for scan_row in [57, 58, 59]:
        row = grid[scan_row, :]
        c = 0
        while c < 64:
            color = int(row[c])
            if color in structural:
                c += 1
                continue
            start = c
            while c < 64 and int(row[c]) == color:
                c += 1
            run_len = c - start
            if run_len >= 3:
                center_x = (start + c - 1) // 2
                # Check not already found
                if not any(abs(center_x - ix) < 4 for ix, _, _ in items):
                    items.append((center_x, scan_row, color))

        if items:
            break  # Found items on this row

    if verbose:
        print(f"  Palette: {items}")
    return sorted(items, key=lambda i: i[0])


def read_empty_slots(grid, verbose=False):
    """Find empty slots (color 2 clusters) in frame area (y 15-50).
    Returns list of (center_x, center_y) left-to-right."""
    slots = []
    # Color 2 marks empty slots in the frame area
    for r in range(15, 50):
        for c in range(64):
            if int(grid[r, c]) == 2:
                # Check not already near a found slot
                if not any(abs(c - sx) < 4 and abs(r - sy) < 4 for sx, sy in slots):
                    slots.append((c, r))

    # Group nearby color-2 pixels into slot centers
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

    if verbose:
        print(f"  Slots: {merged}")
    return sorted(merged, key=lambda s: s[0])


def try_sb26_placement(env, fd, verbose=False):
    """Try to solve current sb26 level by reading grid and placing items."""
    grid = get_frame(fd)
    prev_levels = fd.levels_completed

    if verbose:
        print(f"\n  === Level {fd.levels_completed} ===")

    # Read game state
    dests = read_destinations(grid, verbose)
    palette = read_palette(grid, verbose)
    slots = read_empty_slots(grid, verbose)

    if not dests or not palette or not slots:
        if verbose:
            print(f"  Missing data: dests={len(dests)} palette={len(palette)} slots={len(slots)}")
        return fd, False

    dest_colors = [c for _, c in dests]
    if verbose:
        print(f"  Target sequence: {dest_colors}")
        print(f"  Palette colors: {[c for _, _, c in palette]}")
        print(f"  Slot count: {len(slots)}")

    # Build palette lookup: color → (x, y)
    pal_by_color = {}
    for px, py, pc in palette:
        if pc not in pal_by_color:
            pal_by_color[pc] = (px, py)

    # Place each destination color into corresponding slot
    placed = 0
    for i, target_color in enumerate(dest_colors):
        if i >= len(slots):
            if verbose:
                print(f"  Ran out of slots at dest {i}")
            break

        if target_color not in pal_by_color:
            if verbose:
                print(f"  Color {target_color} not in palette!")
            continue

        px, py = pal_by_color[target_color]
        sx, sy = slots[i]

        if verbose:
            print(f"  [{i}] color {target_color}: click({px},{py}) → click({sx},{sy})")

        # Click 1: select palette item
        try:
            fd = env.step(GameAction.ACTION6, data={'x': int(px), 'y': int(py)})
        except Exception as e:
            if verbose:
                print(f"    Select failed: {e}")
            continue

        # Click 2: place in slot
        try:
            fd = env.step(GameAction.ACTION6, data={'x': int(sx), 'y': int(sy)})
        except Exception as e:
            if verbose:
                print(f"    Place failed: {e}")
            continue

        placed += 1
        grid = get_frame(fd)

        # Remove used color and re-scan remaining palette/slots
        del pal_by_color[target_color]
        palette = read_palette(grid, verbose=False)
        pal_by_color = {}
        for ppx, ppy, ppc in palette:
            if ppc not in pal_by_color:
                pal_by_color[ppc] = (ppx, ppy)
        slots = read_empty_slots(grid, verbose=False)

    if verbose:
        print(f"  Placed {placed}/{len(dest_colors)} items")

    # Submit with ACTION5
    try:
        fd = env.step(GameAction.ACTION5)
    except Exception as e:
        if verbose:
            print(f"  Submit error: {e}")
        return fd, False

    # Wait for animation (ACTION5 triggers eval animation)
    for tick in range(50):
        grid = get_frame(fd)
        if fd.levels_completed > prev_levels:
            break
        if fd.state.name in ("WON", "GAME_OVER"):
            break
        # Animations process on each step — send dummy actions
        try:
            fd = env.step(GameAction.ACTION7)  # undo is cheapest (free)
        except:
            break

    leveled = fd.levels_completed > prev_levels
    if verbose:
        status = "LEVEL UP!" if leveled else "no level up"
        print(f"  Result: {status} ({fd.levels_completed}/{fd.win_levels})")

    return fd, leveled


def brute_force_permutations(arcade, game_id, verbose=False):
    """Try all permutations of palette items into slots on a fresh game instance."""
    from itertools import permutations

    # Fresh instance to read layout
    env0 = arcade.make(game_id)
    fd0 = env0.reset()
    grid0 = get_frame(fd0)

    palette = read_palette(grid0)
    slots = read_empty_slots(grid0)

    n = min(len(palette), len(slots))
    if n == 0:
        return None, False
    if n > 7:
        if verbose:
            print(f"  {n}! permutations too many, skipping brute force")
        return None, False

    total_perms = 1
    for i in range(1, n + 1):
        total_perms *= i
    if verbose:
        print(f"  Brute force: {n} items, {total_perms} permutations")

    for perm_idx, perm in enumerate(permutations(range(n))):
        env2 = arcade.make(game_id)
        fd2 = env2.reset()
        grid2 = get_frame(fd2)

        # Re-read positions fresh
        pal = read_palette(grid2)
        slt = read_empty_slots(grid2)
        if len(pal) < n or len(slt) < n:
            continue

        ok = True
        for slot_idx, pal_idx in enumerate(perm):
            if pal_idx >= len(pal) or slot_idx >= len(slt):
                ok = False
                break
            px, py, _ = pal[pal_idx]
            sx, sy = slt[slot_idx]

            try:
                fd2 = env2.step(GameAction.ACTION6, data={'x': int(px), 'y': int(py)})
                fd2 = env2.step(GameAction.ACTION6, data={'x': int(sx), 'y': int(sy)})
            except:
                ok = False
                break

            grid2 = get_frame(fd2)
            pal = read_palette(grid2)
            slt = read_empty_slots(grid2)

        if not ok:
            continue

        # Submit
        prev = fd2.levels_completed
        try:
            fd2 = env2.step(GameAction.ACTION5)
        except:
            continue

        # Wait for animation
        for _ in range(50):
            if fd2.levels_completed > prev or fd2.state.name in ("WON", "GAME_OVER"):
                break
            try:
                fd2 = env2.step(GameAction.ACTION7)
            except:
                break

        if fd2.levels_completed > prev:
            if verbose:
                pal_colors = [c for _, _, c in read_palette(get_frame(arcade.make(game_id).reset()))]
                actual = [pal_colors[i] for i in perm]
                print(f"  ★ Perm {perm_idx}: {perm} = {actual} → Level {fd2.levels_completed}!")
            return fd2, True

    return None, False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--brute", action="store_true", help="Try brute force permutations")
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

    # Attempt 1: Smart placement (read colors, match to destinations)
    print("\n--- Smart placement ---")
    env = arcade.make(game_id)
    fd = env.reset()
    grid = get_frame(fd)
    print(f"Grid: {grid.shape}, actions: {[a.value if hasattr(a, 'value') else int(a) for a in fd.available_actions]}, win={fd.win_levels}")

    fd, success = try_sb26_placement(env, fd, verbose=True)
    best_levels = fd.levels_completed if success else 0

    # Attempt 2: Brute force permutations (try all orderings)
    if not success or args.brute:
        print("\n--- Brute force permutations ---")
        fd2, success2 = brute_force_permutations(arcade, game_id, verbose=True)
        if fd2 and fd2.levels_completed > best_levels:
            best_levels = fd2.levels_completed
            fd = fd2

    print(f"\n{'='*60}")
    print(f"Result: {best_levels}/{fd.win_levels if fd else '?'} levels")

    # Store in membot
    try:
        import requests
        requests.post("http://localhost:8000/api/store",
            json={"text": f"ARC-AGI-3 sb26 solver: {best_levels} levels. "
                  f"Strategy: read destination colors at top, match palette items into frame slots, submit with ACTION5. "
                  f"Game type: sequence matching puzzle."},
            timeout=2)
    except:
        pass


if __name__ == "__main__":
    main()
