#!/usr/bin/env python3
"""
sb26 World-Model Solver — Sequence-matching puzzle.

Uses game.set_level() to quickly reset between permutation attempts.
For each level, tries all permutations of palette tokens into frame slots.
"""
import sys, os, json, time
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")
import numpy as np
from itertools import permutations

sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

GAME_ID = 'sb26-7fbdac44'


def get_frame(fd):
    grid = np.array(fd.frame)
    return grid[-1] if grid.ndim == 3 else grid


def wait_anim(env, game, fd, max_frames=300):
    for _ in range(max_frames):
        busy = (game.modqnpqfi > 0 or game.ulzvbcvzs or game.xjxrqgaqw >= 0 or
                game.ftyhvmeft >= 0 or game.lmvwmlqtw >= 0 or game.japgbruyb >= 0 or
                game.bbiavyren >= 0 or game.artsfnufc >= 0)
        if not busy:
            break
        fd = env.step(GameAction.ACTION7)
    return fd


def get_palette_and_slots(game):
    """Get palette tokens and empty slots in EVALUATION ORDER.

    The evaluation reads frame contents left-to-right. Portal tokens
    redirect to matching-color frames. After reading the redirected frame,
    evaluation returns and continues in the original frame.
    """
    palette = []
    for s in game.dkouqqads:
        if s.y >= 50:
            color = int(s.pixels[1, 1])
            palette.append((s.x + 3, s.y + 3, color))
    palette.sort(key=lambda t: t[0])

    # Get all frame slots (including portals) in frame order
    frames = sorted(game.qaagahahj, key=lambda f: (f.y, f.x))

    # Build frame contents: for each frame, get items sorted by x
    frame_items = {}
    for frame in frames:
        items = []
        # Check tokens placed in this frame
        for token in game.dkouqqads:
            if token.y < 50 and token.y >= frame.y and token.y < frame.y + frame.height:
                if token.x >= frame.x and token.x < frame.x + frame.width:
                    is_portal = token.name == 'vgszefyyyp'
                    items.append(('token', token.x, token.y, int(token.pixels[1, 1]), is_portal))
        # Check empty slots in this frame
        for slot in game.dewwplfix:
            if not slot.is_visible:
                continue
            if slot.y >= frame.y and slot.y < frame.y + frame.height:
                if slot.x >= frame.x and slot.x < frame.x + frame.width:
                    items.append(('slot', slot.x, slot.y, -1, False))
        items.sort(key=lambda i: i[1])  # Sort by x
        frame_items[id(frame)] = (frame, items)

    # Compute evaluation order by traversing frames with portal jumps
    # Start with first frame
    eval_order = []
    frame_by_border = {}
    for frame in frames:
        border = int(frame.pixels[0, 0])
        frame_by_border[border] = frame

    def traverse(frame, visited=None):
        if visited is None:
            visited = set()
        if id(frame) in visited:
            return
        visited.add(id(frame))

        items = frame_items.get(id(frame), (frame, []))[1]
        deferred = []  # Items after a portal jump

        for item in items:
            item_type, ix, iy, color, is_portal = item
            if is_portal:
                # Jump to frame with matching border color
                target_frame = frame_by_border.get(color)
                if target_frame and id(target_frame) not in visited:
                    traverse(target_frame, visited)
                # After returning, continue with remaining items
            elif item_type == 'slot':
                eval_order.append((ix + 3, iy + 3))

    if frames:
        traverse(frames[0])

    return palette, eval_order


def try_perm(env, game, palette, slots, perm):
    """Place tokens in given order and submit. Returns (won, action_count)."""
    prev = game.level_index
    ac = 0

    for slot_idx, pal_idx in enumerate(perm):
        if slot_idx >= len(slots) or pal_idx >= len(palette):
            break
        tx, ty, _ = palette[pal_idx]
        sx, sy = slots[slot_idx]
        env.step(GameAction.ACTION6, data={'x': tx, 'y': ty})
        env.step(GameAction.ACTION6, data={'x': sx, 'y': sy})
        ac += 2

    fd = env.step(GameAction.ACTION5)
    ac += 1
    fd = wait_anim(env, game, fd)

    won = fd.levels_completed > prev
    return won, ac, fd


def solve_level(env, game, verbose=False):
    """Solve current level by trying permutations with level reset."""
    level_idx = game.level_index
    palette, slots = get_palette_and_slots(game)
    n = min(len(palette), len(slots))

    targets = [int(s.pixels[0, 0]) for s in game.wcfyiodrx]
    pal_colors = [c for _, _, c in palette]

    if verbose:
        print(f"  Targets: {targets}")
        print(f"  Palette: {pal_colors}")
        print(f"  Slots: {len(slots)}")

    if n == 0:
        return None, 0, []

    # Try direct color match first
    pal_by_color = {}
    for i, c in enumerate(pal_colors):
        pal_by_color.setdefault(c, []).append(i)

    direct = []
    temp_map = {c: list(idxs) for c, idxs in pal_by_color.items()}
    for tc in targets[:n]:
        if tc in temp_map and temp_map[tc]:
            direct.append(temp_map[tc].pop(0))
        else:
            direct = None
            break

    if direct:
        won, ac, fd = try_perm(env, game, palette, slots, direct)
        if won:
            # Build click sequence
            clicks = []
            for si, pi in enumerate(direct):
                tx, ty, _ = palette[pi]
                sx, sy = slots[si]
                clicks.extend([('click', tx, ty), ('click', sx, sy)])
            clicks.append(('submit', 0, 0))
            if verbose:
                print(f"  Direct match: {ac} actions")
            return fd, ac, clicks

        # Reset level
        game.set_level(level_idx)
        env.step(GameAction.RESET)

    # Try all permutations
    total = 1
    for i in range(1, n + 1):
        total *= i

    if verbose:
        print(f"  Trying {total} permutations...")

    for idx, perm in enumerate(permutations(range(n))):
        if direct and list(perm) == direct:
            continue

        # Reset level between attempts
        game.set_level(level_idx)
        fd = env.step(GameAction.RESET)

        # Re-read palette and slots (positions reset)
        palette, slots = get_palette_and_slots(game)
        if len(palette) < n or len(slots) < n:
            continue

        won, ac, fd = try_perm(env, game, palette, slots, perm)

        if won:
            perm_colors = [palette[i][2] for i in perm]
            clicks = []
            for si, pi in enumerate(perm):
                tx, ty, _ = palette[pi]
                sx, sy = slots[si]
                clicks.extend([('click', tx, ty), ('click', sx, sy)])
            clicks.append(('submit', 0, 0))
            if verbose:
                print(f"  Perm {idx}/{total}: {perm_colors} SOLVED in {ac} actions")
            return fd, ac, clicks

        if idx % 500 == 0 and verbose and idx > 0:
            print(f"    Tried {idx}/{total}...")

    if verbose:
        print(f"  No permutation worked ({total} tried)")

    # Reset level for next attempt
    game.set_level(level_idx)
    env.step(GameAction.RESET)
    return None, 0, []


def solve_all_levels(verbose=True):
    """Solve all 8 sb26 levels."""
    arcade = Arcade()
    env = arcade.make(GAME_ID)
    fd = env.reset()
    game = env._game

    all_level_clicks = []
    total = 0

    for level_idx in range(8):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Level {level_idx + 1} / 8")

        fd, ac, clicks = solve_level(env, game, verbose=verbose)

        if fd and fd.levels_completed > level_idx:
            all_level_clicks.append(clicks)
            total += ac
            if verbose:
                print(f"  -> Total: {total} actions, {fd.levels_completed} levels")
        else:
            if verbose:
                print(f"  FAILED L{level_idx+1}")
            break

        if fd.state.name == 'GAME_OVER':
            if verbose:
                print(f"  GAME OVER at L{level_idx+1}")
            break

    # Track levels solved
    levels_solved = len(all_level_clicks)

    return all_level_clicks, total, levels_solved, env, game


def replay_and_capture(arcade, all_level_clicks, verbose=True):
    """Replay with visual capture."""
    try:
        from arc_session_writer import SessionWriter
    except ImportError:
        SessionWriter = None

    env = arcade.make(GAME_ID)
    fd = env.reset()
    game = env._game
    grid = get_frame(fd)

    writer = None
    if SessionWriter:
        writer = SessionWriter(
            game_id=GAME_ID,
            win_levels=fd.win_levels,
            available_actions=[int(a) for a in fd.available_actions],
            baseline=153,
            player="sb26_world_model_solver"
        )
        writer.record_step(0, grid, levels_completed=0, state="PLAYING")

    total = 0
    for level_clicks in all_level_clicks:
        for atype, ax, ay in level_clicks:
            if atype == 'submit':
                fd = env.step(GameAction.ACTION5)
            else:
                fd = env.step(GameAction.ACTION6, data={'x': ax, 'y': ay})
            total += 1
            grid = get_frame(fd)
            fd = wait_anim(env, game, fd)

            if writer:
                action_id = 5 if atype == 'submit' else 6
                writer.record_step(action_id, grid,
                    levels_completed=fd.levels_completed,
                    x=ax if atype != 'submit' else None,
                    y=ay if atype != 'submit' else None,
                    state=fd.state.name)

    if writer:
        writer.record_game_end(state=fd.state.name, levels_completed=fd.levels_completed)
    return fd, total


def main():
    import time as _time
    print("=" * 60)
    print("sb26 World-Model Solver")
    print("=" * 60)

    t0 = _time.monotonic()
    all_level_clicks, total, levels_solved, _, _ = solve_all_levels(verbose=True)
    t1 = _time.monotonic()

    print(f"\n{'='*60}")
    print(f"Result: {levels_solved}/8 levels, {total} actions (baseline 153)")
    print(f"Time: {t1-t0:.1f}s")

    if levels_solved >= 1:
        print(f"\nReplaying with visual capture...")
        arcade = Arcade()
        fd2, total2 = replay_and_capture(arcade, all_level_clicks, verbose=True)
        print(f"Replay: {fd2.levels_completed}/8, {total2} actions captured")

    sol_data = {
        "levels_solved": levels_solved,
        "total_actions": total,
    }
    for li, lc in enumerate(all_level_clicks):
        sol_data[f"L{li+1}"] = {"actions": len(lc), "clicks": [(a, x, y) for a, x, y in lc]}

    out_path = os.path.join(os.path.dirname(__file__), "sb26_action_sequence.json")
    with open(out_path, "w") as f:
        json.dump(sol_data, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
