#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Deep Probe — Sprout

Try unusual strategies that basic clicking misses:
1. Click same cell repeatedly (maybe games need repeated interaction)
2. Click cells in specific spatial patterns (corners, borders, center)
3. Click cells that CHANGED after a previous click (chain reactions)
4. For move+click: try each direction N times then click ALL non-bg
5. For ACTION7 games: try ACTION7 between clicks
6. Click non-bg in reverse order (bottom-right to top-left)

Focus on games that DON'T score yet but have effective colors.
"""
import sys, os, time, json, random
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


def run_strategy(arcade, game_id, strategy_fn, name=""):
    """Run a strategy, return (levels, win_levels, steps, state)."""
    try:
        env = arcade.make(game_id)
        fd = env.reset()
        return strategy_fn(env, fd)
    except Exception as e:
        return 0, 0, 0, f"ERROR: {e}"


def strat_click_changed_cells(env, fd):
    """After each click, find cells that changed and click those."""
    grid = get_frame(fd)
    bg = int(np.bincount(grid.flatten()).argmax())
    max_levels = fd.levels_completed
    steps = 0

    # Initial click: first non-bg cell
    non_bg = np.argwhere(grid != bg)
    if len(non_bg) == 0:
        return 0, fd.win_levels, 0, "NO_TARGETS"

    # Click first non-bg
    r, c = int(non_bg[0][0]), int(non_bg[0][1])
    prev = grid.copy()
    fd = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
    grid = get_frame(fd)
    steps += 1

    # Now chain: click cells that changed
    for _ in range(500):
        if fd.state.name in ("WON", "GAME_OVER"):
            break
        changed_positions = np.argwhere(prev != grid)
        if len(changed_positions) == 0:
            # No changes — click next non-bg
            non_bg = np.argwhere(grid != bg)
            if len(non_bg) == 0:
                break
            r, c = int(non_bg[steps % len(non_bg)][0]), int(non_bg[steps % len(non_bg)][1])
        else:
            # Click a changed cell
            idx = steps % len(changed_positions)
            r, c = int(changed_positions[idx][0]), int(changed_positions[idx][1])

        prev = grid.copy()
        try:
            fd = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
        except Exception:
            continue
        grid = get_frame(fd)
        steps += 1
        if fd.levels_completed > max_levels:
            max_levels = fd.levels_completed

    return max_levels, fd.win_levels, steps, fd.state.name


def strat_click_same_cell(env, fd, repeat=50):
    """Click the same non-bg cell repeatedly."""
    grid = get_frame(fd)
    bg = int(np.bincount(grid.flatten()).argmax())
    max_levels = fd.levels_completed
    non_bg = np.argwhere(grid != bg)
    if len(non_bg) == 0:
        return 0, fd.win_levels, 0, "NO_TARGETS"

    best_lvl = 0
    # Try 5 different cells
    for cell_idx in range(min(5, len(non_bg))):
        env2 = Arcade().make(fd.game_id if hasattr(fd, 'game_id') else '')
        # Can't get game_id from fd easily, just reuse env pattern
        break  # Just use the first cell for now

    r, c = int(non_bg[0][0]), int(non_bg[0][1])
    steps = 0
    for _ in range(repeat):
        try:
            fd = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
        except Exception:
            continue
        steps += 1
        if fd.levels_completed > max_levels:
            max_levels = fd.levels_completed
        if fd.state.name in ("WON", "GAME_OVER"):
            break

    return max_levels, fd.win_levels, steps, fd.state.name


def strat_click_borders(env, fd):
    """Click cells on the border of colored regions."""
    grid = get_frame(fd)
    bg = int(np.bincount(grid.flatten()).argmax())
    max_levels = fd.levels_completed
    steps = 0

    # Find border cells: non-bg cells adjacent to bg
    h, w = grid.shape
    border_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg:
                continue
            # Check 4 neighbors
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg:
                    border_cells.append((r, c, int(grid[r, c])))
                    break

    if not border_cells:
        return 0, fd.win_levels, 0, "NO_BORDERS"

    for r, c, color in border_cells[:200]:
        try:
            fd = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
        except Exception:
            continue
        grid = get_frame(fd)
        steps += 1
        if fd.levels_completed > max_levels:
            max_levels = fd.levels_completed
        if fd.state.name in ("WON", "GAME_OVER"):
            break

    return max_levels, fd.win_levels, steps, fd.state.name


def strat_click_reverse(env, fd):
    """Click all non-bg cells bottom-right to top-left."""
    grid = get_frame(fd)
    bg = int(np.bincount(grid.flatten()).argmax())
    max_levels = fd.levels_completed
    steps = 0

    non_bg = np.argwhere(grid != bg)
    if len(non_bg) == 0:
        return 0, fd.win_levels, 0, "NO_TARGETS"

    # Reverse order
    non_bg = non_bg[np.lexsort((non_bg[:, 1], non_bg[:, 0]))][::-1]

    for pos in non_bg[:500]:
        r, c = int(pos[0]), int(pos[1])
        try:
            fd = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
        except Exception:
            continue
        grid = get_frame(fd)
        steps += 1
        if fd.levels_completed > max_levels:
            max_levels = fd.levels_completed
        if fd.state.name in ("WON", "GAME_OVER"):
            break

    return max_levels, fd.win_levels, steps, fd.state.name


def strat_a7_then_click(env, fd):
    """ACTION7 then click all non-bg."""
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (fd.available_actions or [])]
    if 7 not in available:
        return 0, fd.win_levels, 0, "NO_A7"

    grid = get_frame(fd)
    bg = int(np.bincount(grid.flatten()).argmax())
    max_levels = fd.levels_completed
    steps = 0

    # Alternate: A7, click, A7, click...
    non_bg = np.argwhere(grid != bg)
    for i, pos in enumerate(non_bg[:100]):
        try:
            fd = env.step(GameAction.ACTION7)
            steps += 1
        except Exception:
            pass

        r, c = int(pos[0]), int(pos[1])
        try:
            fd = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
            steps += 1
        except Exception:
            continue

        grid = get_frame(fd)
        if fd.levels_completed > max_levels:
            max_levels = fd.levels_completed
        if fd.state.name in ("WON", "GAME_OVER"):
            break

    return max_levels, fd.win_levels, steps, fd.state.name


def strat_select_each_click(env, fd):
    """SELECT between each click (some games may need confirmation)."""
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (fd.available_actions or [])]
    if 5 not in available or 6 not in available:
        return 0, fd.win_levels, 0, "NO_SEL_CLK"

    grid = get_frame(fd)
    bg = int(np.bincount(grid.flatten()).argmax())
    max_levels = fd.levels_completed
    steps = 0

    non_bg = np.argwhere(grid != bg)
    for pos in non_bg[:100]:
        r, c = int(pos[0]), int(pos[1])
        try:
            fd = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
            steps += 1
        except Exception:
            continue
        try:
            fd = env.step(GameAction.ACTION5)
            steps += 1
        except Exception:
            pass
        grid = get_frame(fd)
        if fd.levels_completed > max_levels:
            max_levels = fd.levels_completed
        if fd.state.name in ("WON", "GAME_OVER"):
            break

    return max_levels, fd.win_levels, steps, fd.state.name


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE ARC-AGI-3 Deep Probe — Sprout")
    print("=" * 60)

    arcade = Arcade()
    envs = arcade.get_environments()

    # Skip already-scoring games (lp85, vc33)
    skip = {"lp85", "vc33"}

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    else:
        targets = [e for e in envs
                   if (e.game_id if hasattr(e, "game_id") else str(e)).split("-")[0] not in skip]

    strategies = [
        ("chain_changed", strat_click_changed_cells),
        ("same_cell", strat_click_same_cell),
        ("borders", strat_click_borders),
        ("reverse", strat_click_reverse),
        ("a7_click", strat_a7_then_click),
        ("sel_click", strat_select_each_click),
    ]

    print(f"\n{len(targets)} games, {len(strategies)} strategies each\n")

    total_levels = 0
    total_scored = 0
    start = time.time()

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        prefix = game_id.split("-")[0]

        # Check if has click
        env = arcade.make(game_id)
        fd = env.reset()
        available = [a.value if hasattr(a, "value") else int(a)
                     for a in (fd.available_actions or [])]
        if 6 not in available:
            continue

        best_lvl = 0
        best_strat = ""

        for name, strat_fn in strategies:
            env2 = arcade.make(game_id)
            fd2 = env2.reset()
            try:
                lvl, win, steps, state = strat_fn(env2, fd2)
            except Exception as e:
                continue
            if lvl > best_lvl:
                best_lvl = lvl
                best_strat = name

        star = f" ★ {best_lvl}" if best_lvl > 0 else ""
        print(f"  {prefix:6s}  {best_lvl}/{fd.win_levels}{star}  [{best_strat}]")

        if best_lvl > 0:
            total_levels += best_lvl
            total_scored += 1

    elapsed = time.time() - start
    print(f"\n  Total: {total_levels} new levels across {total_scored} games in {elapsed:.0f}s")

    # Store findings in membot
    try:
        import requests
        if total_scored > 0:
            requests.post("http://localhost:8000/api/store",
                json={"text": f"ARC-AGI-3 deep probe: {total_scored} new games scored with unusual strategies"},
                timeout=2)
        else:
            requests.post("http://localhost:8000/api/store",
                json={"text": f"ARC-AGI-3 deep probe: chain_changed, same_cell, borders, reverse, a7_click, sel_click strategies all failed on remaining 23 games. These games likely need understanding of the puzzle rules, not just mechanical clicking."},
                timeout=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
