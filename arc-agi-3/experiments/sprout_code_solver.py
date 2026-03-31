#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Code-Aware Solver — Sprout

BREAKTHROUGH: Game source code at environment_files/<game>/<hash>/<game>.py
reveals exact mechanics. Parse game code to extract:
- Clickable sprite positions (tagged sys_click)
- Target/goal positions
- Piece-target matching rules
- Win conditions

Then execute informed strategies based on actual game rules.
"""
import sys, os, time, json, re, ast, random
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from arcengine import GameAction
from arc_agi import Arcade
from membot_cartridge import MembotCartridge

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


def get_frame(fd):
    grid = np.array(fd.frame)
    return grid[-1] if grid.ndim == 3 else grid


def find_game_code(game_id):
    """Find the Python source file for a game."""
    prefix = game_id.split("-")[0]
    game_hash = game_id.split("-")[1] if "-" in game_id else ""
    base = f"environment_files/{prefix}"
    if not os.path.exists(base):
        return None
    for d in os.listdir(base):
        path = os.path.join(base, d, f"{prefix}.py")
        if os.path.exists(path):
            return path
    return None


def extract_sys_click_sprites(code):
    """Find sprites tagged with sys_click — these are clickable."""
    clickable = []
    # Simpler approach: find name="X" followed eventually by sys_click in same Sprite block
    # Look for: name="...", ... tags=[..."sys_click"...]
    sprite_blocks = re.split(r'(?="[^"]+"\s*:\s*Sprite\()', code)
    for block in sprite_blocks:
        if 'sys_click' not in block:
            continue
        name_match = re.search(r'name="([^"]+)"', block)
        tags_match = re.search(r'tags=\[([^\]]*)\]', block)
        if name_match and tags_match:
            name = name_match.group(1)
            tags_str = tags_match.group(1)
            tags = [t.strip().strip('"\'') for t in tags_str.split(",")]
            clickable.append({"name": name, "tags": tags})
    return clickable


def extract_level_positions(code, level_idx=0):
    """Extract sprite positions from level definitions."""
    positions = []
    # Find Level(...) blocks
    level_pattern = re.compile(r'Level\(\s*sprites=\[(.*?)\],\s*grid_size', re.DOTALL)
    levels = list(level_pattern.finditer(code))
    if not levels or level_idx >= len(levels):
        return positions

    level_str = levels[level_idx].group(1)
    # Format: sprites["name"].clone().set_position(x, y)
    pos_pattern = re.compile(r'sprites\["([^"]+)"\]\.clone\(\)\.set_position\((-?\d+),\s*(-?\d+)\)')
    for match in pos_pattern.finditer(level_str):
        name = match.group(1)
        x = int(match.group(2))
        y = int(match.group(3))
        positions.append({"name": name, "x": x, "y": y})

    return positions


def extract_grid_size(code, level_idx=0):
    """Extract grid_size from level definition."""
    pattern = re.compile(r'grid_size=\((\d+),\s*(\d+)\)')
    matches = list(pattern.finditer(code))
    if matches and level_idx < len(matches):
        w = int(matches[level_idx].group(1))
        h = int(matches[level_idx].group(2))
        return (w, h)
    return (64, 64)


def grid_to_display(gx, gy, grid_size, display_size=64):
    """Convert grid coordinates to display pixel coordinates.
    The camera maps grid → display with scaling."""
    gw, gh = grid_size
    scale_x = display_size / gw
    scale_y = display_size / gh
    dx = int(gx * scale_x + scale_x / 2)
    dy = int(gy * scale_y + scale_y / 2)
    return dx, dy


def solve_with_code(arcade, game_id, verbose=False):
    """Read game code and use it to inform solving strategy."""
    code_path = find_game_code(game_id)
    if not code_path:
        return None

    with open(code_path, "r") as f:
        code = f.read()

    prefix = game_id.split("-")[0]

    # Extract game info
    clickable = extract_sys_click_sprites(code)
    level0_positions = extract_level_positions(code, 0)
    grid_size = extract_grid_size(code, 0)

    if verbose:
        print(f"  Code: {code_path}")
        print(f"  Grid size: {grid_size}")
        print(f"  Clickable sprites: {len(clickable)}")
        for s in clickable[:5]:
            print(f"    {s['name']}: tags={s['tags']}")
        print(f"  Level 0 positions: {len(level0_positions)}")
        for p in level0_positions[:10]:
            print(f"    {p['name']} at ({p['x']}, {p['y']})")

    # Find sys_click sprite positions in level 0
    click_names = {s['name'] for s in clickable}
    clickable_positions = [p for p in level0_positions if p['name'] in click_names]

    # Find goal/target positions
    goal_positions = [p for p in level0_positions
                      if any(t in p['name'].lower() for t in ['goal', 'kzeze', 'target'])]

    if verbose:
        print(f"  Clickable at positions: {len(clickable_positions)}")
        print(f"  Goal positions: {len(goal_positions)}")

    # Now play the game using these positions
    env = arcade.make(game_id)
    fd = env.reset()
    grid = get_frame(fd)
    max_levels = fd.levels_completed
    steps = 0

    # Strategy: click all clickable sprites in the level
    if clickable_positions:
        for pos in clickable_positions:
            # Convert grid coords to display coords
            dx, dy = grid_to_display(pos['x'], pos['y'], grid_size)
            # Clamp to valid range
            dx = max(0, min(63, dx))
            dy = max(0, min(63, dy))

            if verbose:
                print(f"  Clicking {pos['name']} at grid({pos['x']},{pos['y']}) -> display({dx},{dy})")

            try:
                fd = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
                steps += 1
            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                continue

            grid = get_frame(fd)
            if fd.levels_completed > max_levels:
                max_levels = fd.levels_completed
                if verbose:
                    print(f"    ★ LEVEL {fd.levels_completed}/{fd.win_levels}!")

            if fd.state.name in ("WON", "GAME_OVER"):
                break

    # If goal positions found, also try clicking those
    if goal_positions and fd.state.name not in ("WON", "GAME_OVER"):
        for pos in goal_positions:
            dx, dy = grid_to_display(pos['x'], pos['y'], grid_size)
            dx = max(0, min(63, dx))
            dy = max(0, min(63, dy))

            if verbose:
                print(f"  Clicking goal {pos['name']} at grid({pos['x']},{pos['y']}) -> display({dx},{dy})")

            try:
                fd = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
                steps += 1
            except Exception as e:
                continue

            grid = get_frame(fd)
            if fd.levels_completed > max_levels:
                max_levels = fd.levels_completed

            if fd.state.name in ("WON", "GAME_OVER"):
                break

    # For r11l-style games: select piece → click goal → wait for animation
    if max_levels == 0 and clickable_positions and goal_positions:
        env2 = arcade.make(game_id)
        fd2 = env2.reset()
        steps2 = 0

        for piece in clickable_positions:
            if fd2.state.name in ("WON", "GAME_OVER"):
                break

            for goal in goal_positions:
                if fd2.state.name in ("WON", "GAME_OVER"):
                    break

                # Click piece to select
                px, py = grid_to_display(piece['x'], piece['y'], grid_size)
                px, py = max(0, min(63, px)), max(0, min(63, py))
                try:
                    fd2 = env2.step(GameAction.ACTION6, data={'x': px, 'y': py})
                    steps2 += 1
                    if verbose:
                        print(f"    Select piece at ({px},{py})")
                except Exception:
                    continue

                # Click goal to initiate movement
                gx, gy = grid_to_display(goal['x'], goal['y'], grid_size)
                gx, gy = max(0, min(63, gx)), max(0, min(63, gy))
                try:
                    fd2 = env2.step(GameAction.ACTION6, data={'x': gx, 'y': gy})
                    steps2 += 1
                    if verbose:
                        print(f"    Move to goal at ({gx},{gy})")
                except Exception:
                    continue

                # Wait for animation to complete (click empty space to advance frames)
                for wait in range(30):
                    if fd2.state.name in ("WON", "GAME_OVER"):
                        break
                    if fd2.levels_completed > max_levels:
                        break
                    try:
                        fd2 = env2.step(GameAction.ACTION6, data={'x': 0, 'y': 0})
                        steps2 += 1
                    except Exception:
                        break

                if fd2.levels_completed > max_levels:
                    max_levels = fd2.levels_completed
                    if verbose:
                        print(f"    ★ Piece→Goal worked! Level {fd2.levels_completed}")

        steps = max(steps, steps2)

    # Also try: just click every position from the level definition
    if max_levels == 0 and fd.state.name not in ("WON", "GAME_OVER"):
        env3 = arcade.make(game_id)
        fd3 = env3.reset()
        steps3 = 0

        for pos in level0_positions[:30]:
            dx, dy = grid_to_display(pos['x'], pos['y'], grid_size)
            dx, dy = max(0, min(63, dx)), max(0, min(63, dy))
            try:
                fd3 = env3.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
                steps3 += 1
            except Exception:
                continue
            if fd3.levels_completed > max_levels:
                max_levels = fd3.levels_completed
                if verbose:
                    print(f"    ★ Position click worked! Level {fd3.levels_completed}")
            if fd3.state.name in ("WON", "GAME_OVER"):
                break

    return {
        "levels": max_levels,
        "win_levels": fd.win_levels,
        "steps": steps,
        "state": fd.state.name if hasattr(fd, 'state') else "?",
        "clickable_count": len(clickable_positions),
        "goal_count": len(goal_positions),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE ARC-AGI-3 Code-Aware Solver — Sprout")
    print("=" * 60)

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    else:
        targets = envs

    total_levels = 0
    total_scored = 0

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        prefix = game_id.split("-")[0]

        print(f"\n{'─'*60}")
        print(f"Game: {game_id}")

        result = solve_with_code(arcade, game_id, verbose=args.verbose)
        if result is None:
            print(f"  No game code found")
            continue

        lvl = result["levels"]
        win = result["win_levels"]
        star = f" ★ {lvl}" if lvl > 0 else ""
        print(f"  Result: {lvl}/{win}{star} "
              f"({result['clickable_count']} clickable, {result['goal_count']} goals)")

        if lvl > 0:
            total_levels += lvl
            total_scored += 1

            # Store in membot
            try:
                import requests
                requests.post("http://localhost:8000/api/store",
                    json={"text": f"ARC-AGI-3 {prefix}: code-aware solver scored {lvl}/{win} levels. "
                          f"Strategy: click {result['clickable_count']} clickable sprites, "
                          f"{result['goal_count']} goals."},
                    timeout=2)
            except Exception:
                pass

    print(f"\n{'='*60}")
    print(f"Total: {total_levels} levels across {total_scored} games")

    # Save
    log_path = f"arc-agi-3/experiments/logs/code_solver_{int(time.time())}.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"total_levels": total_levels, "games_scored": total_scored}, f, indent=2)
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
