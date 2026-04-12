#!/usr/bin/env python3
"""
Replay all 8 solved games through SessionWriter for visual memory.

Every frame saved as PNG. Animations saved as GIF.
Per-run directories in shared-context/arc-agi-3/visual-memory/{game}/
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_all_frames
from arc_session_writer import SessionWriter

INT_TO_GA = {a.value: a for a in GameAction}
UP, DOWN, LEFT, RIGHT, SELECT, CLICK = 1, 2, 3, 4, 5, 6


def _do_step(env, sw, fd, item):
    """Execute one action, record in SessionWriter. Returns new fd."""
    if isinstance(item, tuple):
        action, x, y = item
        fd = env.step(INT_TO_GA[action], data={"x": x, "y": y})
    else:
        action = item
        x = y = None
        fd = env.step(INT_TO_GA[action])

    all_frames = get_all_frames(fd)
    grid = all_frames[-1]
    sw.record_step(action, grid,
                   all_frames=all_frames if len(all_frames) > 1 else None,
                   levels_completed=fd.levels_completed,
                   x=x, y=y,
                   state=fd.state.name if hasattr(fd.state, 'name') else str(fd.state))
    return fd


def replay_game(game_id, action_sequence, label=""):
    """Replay a game with a flat action sequence."""
    arcade = Arcade()
    env = arcade.make(game_id)
    fd = env.reset()

    sw = SessionWriter(game_id, win_levels=fd.win_levels,
                       available_actions=[
                           a.value if hasattr(a, 'value') else int(a)
                           for a in (fd.available_actions or [])],
                       player=label or "replay")

    for item in action_sequence:
        fd = _do_step(env, sw, fd, item)
        if fd.state.name in ("GAME_OVER", "WON"):
            break

    sw.record_game_end(fd.state.name, fd.levels_completed)
    files = [f for f in os.listdir(sw.run_dir) if f.endswith('.png')]
    gifs = [f for f in os.listdir(sw.run_dir) if f.endswith('.gif')]
    print(f"  {game_id}: {fd.levels_completed}/{fd.win_levels} "
          f"in {sw.step} steps, {len(files)} PNGs, {len(gifs)} GIFs")
    return fd.levels_completed >= fd.win_levels


def replay_game_levels(game_id, level_actions, label=""):
    """Replay a game with per-level action lists. Stops sending actions
    for a level once level-up is detected (prevents action leak)."""
    arcade = Arcade()
    env = arcade.make(game_id)
    fd = env.reset()

    sw = SessionWriter(game_id, win_levels=fd.win_levels,
                       available_actions=[
                           a.value if hasattr(a, 'value') else int(a)
                           for a in (fd.available_actions or [])],
                       player=label or "replay")

    for lv_idx, actions in enumerate(level_actions):
        target_level = lv_idx + 1
        for item in actions:
            fd = _do_step(env, sw, fd, item)
            if fd.state.name in ("GAME_OVER", "WON"):
                break
            if fd.levels_completed >= target_level:
                break  # Level solved, stop sending actions for this level
        if fd.state.name in ("GAME_OVER", "WON"):
            break

    sw.record_game_end(fd.state.name, fd.levels_completed)
    files = [f for f in os.listdir(sw.run_dir) if f.endswith('.png')]
    gifs = [f for f in os.listdir(sw.run_dir) if f.endswith('.gif')]
    print(f"  {game_id}: {fd.levels_completed}/{fd.win_levels} "
          f"in {sw.step} steps, {len(files)} PNGs, {len(gifs)} GIFs")
    return fd.levels_completed >= fd.win_levels


def clicks(coords):
    """Convert click coordinate list to action tuples."""
    return [(CLICK, x, y) for x, y in coords]


# ═══════════════════════════════════════════════════════════
# GAME SOLUTIONS
# ═══════════════════════════════════════════════════════════

def ft09_actions():
    """ft09: click-only color puzzle. 75 clicks."""
    levels = [
        [(38,54),(38,46),(38,38),(54,46)],
        [(22,32),(22,24),(30,48),(22,48),(22,16),(38,32),(38,24)],
        [(38,54),(46,30),(30,22),(22,54),(22,14),(14,22),(30,6),(46,38),(14,30),(30,54),(38,6),(30,38),(22,46),(22,6)],
        [(38,48),(38,48),(46,24),(30,16),(22,48),(22,48),(38,32),(30,24),(22,32),(22,32),(30,48),(30,48),(22,16),(22,16),(46,16),(30,32)],
        [(16,14),(32,38),(32,14),(24,6),(32,22),(40,38),(48,46),(32,46),(40,54),(32,54),(48,38),(16,22),(32,6),(16,38),(16,54),(16,30),(32,30),(48,22),(24,14),(40,46),(24,30)],
        [(22,16),(38,16),(14,24),(22,24),(38,32),(14,32),(46,32),(30,32),(46,40),(22,40),(54,40),(6,16),(6,8)],
    ]
    actions = []
    for lv in levels:
        actions.extend(clicks(lv))
    return actions


def sc25_actions():
    """sc25: spell-casting maze. Per-level sequences (break on level-up)."""
    DIAMOND = clicks([(30,50),(25,55),(35,55),(30,60)])
    LSHAPE = clicks([(25,50),(30,50),(30,55)])
    VLINE = clicks([(30,50),(30,55),(30,60)])

    # Return per-level action lists (replay_game_levels handles breaks)
    return [
        # L1
        [RIGHT] + DIAMOND + [LEFT]*12,
        # L2
        [RIGHT] + LSHAPE + [UP]*8,
        # L3
        [RIGHT] + VLINE + [LEFT]*3 + [DOWN]*4 + [LEFT]*3,
        # L4
        DIAMOND + [DOWN]*5 + [LEFT]*4 + VLINE + DIAMOND + [DOWN]*2 + [RIGHT]*8,
        # L5
        DIAMOND + LSHAPE + [LEFT]*7 + [UP]*12 + VLINE + [DOWN]*14 + [LEFT] + VLINE + DIAMOND + LSHAPE + [UP]*8,
        # L6
        DIAMOND + LSHAPE + [RIGHT]*2 + [UP]*3 + VLINE + DIAMOND + LSHAPE + [LEFT]*3 + VLINE + LSHAPE + DIAMOND + [UP]*2 + [RIGHT]*2 + [UP]*12,
    ]


def tr87_actions():
    """tr87: rewrite rule puzzle. 137 actions."""
    actions = []
    # L1-L4 from solver
    actions.extend([DOWN,DOWN,RIGHT,DOWN,DOWN,RIGHT,UP,UP,UP,RIGHT,DOWN,RIGHT,DOWN,DOWN])
    actions.extend([DOWN,DOWN,DOWN,RIGHT,DOWN,DOWN,RIGHT,UP,UP,UP,RIGHT,UP,UP,RIGHT,UP,UP,UP,RIGHT,UP,UP,UP,RIGHT,DOWN,DOWN,DOWN])
    actions.extend([DOWN,RIGHT,DOWN,DOWN,RIGHT,DOWN,DOWN,DOWN,RIGHT,DOWN,DOWN,RIGHT,UP,UP,UP,RIGHT,DOWN,DOWN,RIGHT,DOWN,DOWN])
    actions.extend([UP,UP,UP,RIGHT,UP,UP,UP,RIGHT,UP,UP,UP,RIGHT,RIGHT,DOWN,DOWN,RIGHT,UP,UP,UP,RIGHT,DOWN])
    # L5 alter_rules
    actions.extend([DOWN,DOWN,DOWN,RIGHT,DOWN,DOWN,RIGHT,UP,RIGHT,UP,RIGHT,DOWN,RIGHT,UP,UP,RIGHT,UP,UP,UP,RIGHT])
    # L6 tree_translation (LEFT first to reset cursor)
    actions.extend([LEFT,UP,UP,UP,RIGHT,DOWN,RIGHT,DOWN,DOWN,DOWN,RIGHT,DOWN,RIGHT,DOWN,RIGHT,UP,UP,UP,RIGHT,UP,UP,UP,RIGHT,DOWN,DOWN,DOWN,RIGHT,DOWN,RIGHT,DOWN,DOWN,DOWN,RIGHT,UP,RIGHT,UP])
    return actions


def tu93_actions():
    """tu93: maze with entities. All 9 levels from tu93_solutions.json."""
    actions = []
    # L1 (18 actions)
    actions.extend([RIGHT,DOWN,DOWN,RIGHT,UP,RIGHT,DOWN,DOWN,LEFT,LEFT,DOWN,RIGHT,RIGHT,DOWN,RIGHT,UP,RIGHT,DOWN])
    # L2 (10 actions — bypass arrow via lower corridor)
    actions.extend([UP,RIGHT,RIGHT,DOWN,RIGHT,RIGHT,UP,RIGHT,RIGHT,UP])
    # L3 (19 actions — destroy 3 arrows in order)
    actions.extend([UP,UP,RIGHT,UP,LEFT,LEFT,UP,LEFT,LEFT,DOWN,RIGHT,DOWN,LEFT,LEFT,LEFT,DOWN,RIGHT,DOWN,RIGHT])
    # L4 (17 actions — bouncer timing + arrow destroy)
    actions.extend([RIGHT,RIGHT,LEFT,RIGHT,RIGHT,RIGHT,UP,UP,DOWN,UP,LEFT,UP,UP,LEFT,LEFT,DOWN,LEFT])
    # L5 (29 actions — waste steps for bouncer timing)
    actions.extend([LEFT,RIGHT,LEFT,RIGHT,LEFT,RIGHT,LEFT,LEFT,LEFT,LEFT,LEFT,LEFT,LEFT,DOWN,DOWN,DOWN,RIGHT,DOWN,DOWN,RIGHT,RIGHT,RIGHT,UP,DOWN,UP,DOWN,UP,UP,LEFT])
    # L6 (32 actions — arrow destruction chain)
    actions.extend([LEFT,LEFT,DOWN,DOWN,RIGHT,DOWN,DOWN,LEFT,DOWN,LEFT,LEFT,RIGHT,RIGHT,LEFT,RIGHT,UP,LEFT,LEFT,RIGHT,RIGHT,RIGHT,UP,UP,LEFT,LEFT,LEFT,DOWN,LEFT,UP,UP,UP,LEFT])
    # L7 (14 actions — simple pathfinding)
    actions.extend([RIGHT,RIGHT,RIGHT,DOWN,DOWN,RIGHT,UP,RIGHT,UP,UP,UP,RIGHT,DOWN,DOWN])
    # L8 (21 actions — goose chase with delayed entity)
    actions.extend([RIGHT,RIGHT,UP,UP,RIGHT,RIGHT,LEFT,LEFT,LEFT,DOWN,DOWN,RIGHT,UP,UP,RIGHT,RIGHT,UP,UP,UP,LEFT,LEFT])
    # L9 (29 actions — BFS-solved, bouncer timing + arrow destruction)
    actions.extend([LEFT,LEFT,UP,UP,RIGHT,DOWN,DOWN,LEFT,UP,UP,RIGHT,UP,UP,RIGHT,RIGHT,RIGHT,DOWN,DOWN,RIGHT,DOWN,LEFT,DOWN,LEFT,DOWN,DOWN,LEFT,LEFT,UP,RIGHT])
    return actions


def cd82_actions():
    """cd82: circular stamp painting. Solved by Nomad, 127 actions.
    We don't have exact sequences — only Nomad's fleet learning has strategies.
    Using the algorithmic solver's L1 sequence + generic approach."""
    # cd82 needs Nomad's actual replay data. For now, skip.
    return None


def sb26_actions():
    """sb26: hierarchical indicator placement. 140 actions.
    Interactive solve — exact click sequences not persisted."""
    return None


def tn36_actions():
    """tn36: programmable block movement. 119 actions.
    Interactive solve — exact click sequences not persisted."""
    return None


def vc33_actions():
    """vc33: dual-button puzzle. 167 actions.
    Interactive solve — exact click sequences not persisted."""
    return None


def lp85_actions():
    """lp85: McNugget's solve. Exact sequences not available here."""
    return None


# ═══════════════════════════════════════════════════════════

def main():
    # Games with flat action sequences
    flat_games = [
        ("ft09-0d8bbf25", ft09_actions(), "ft09-algorithmic"),
        ("tr87-cd924810", tr87_actions(), "tr87-verified"),
        ("tu93-2b534c15", tu93_actions(), "tu93-verified-9of9"),
    ]

    # Games with per-level action lists (need break-on-level-up)
    level_games = [
        ("sc25-f9b21a2f", sc25_actions(), "sc25-verified"),
    ]

    # Games we can't replay (no exact sequences persisted)
    skipped = ["cd82", "sb26", "tn36", "vc33", "lp85"]

    print("Replaying solved games for visual memory...\n")

    for game_id, actions, label in flat_games:
        if actions is None:
            print(f"  {game_id}: SKIPPED (no replay sequence)")
            continue
        try:
            replay_game(game_id, actions, label)
        except Exception as e:
            print(f"  {game_id}: ERROR — {e}")

    for game_id, level_acts, label in level_games:
        if level_acts is None:
            print(f"  {game_id}: SKIPPED")
            continue
        try:
            replay_game_levels(game_id, level_acts, label)
        except Exception as e:
            print(f"  {game_id}: ERROR — {e}")

    if skipped:
        print(f"\nSkipped (no exact sequences): {', '.join(skipped)}")
        print("These were solved in interactive sessions where click coordinates")
        print("weren't persisted. Need to re-solve or extract from session logs.")


if __name__ == "__main__":
    main()
