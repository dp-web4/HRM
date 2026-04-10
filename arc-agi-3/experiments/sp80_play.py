#!/usr/bin/env python3
"""
sp80 Live Player — plays against the ARC-AGI-3 SDK using computed solutions.

Uses SessionWriter for visual memory capture. Solutions L1-L3 pre-computed.
"""

import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from arc_session_writer import SessionWriter

# Pre-computed solutions: game-coordinate actions
# Rotation is handled at play time via the game's internal mapping
SOLUTIONS = {
    0: {  # L1 (rotation=0, grid=16x16)
        "rotation": 0, "grid": (16, 16),
        "game_actions": [
            ("CLICK", 5, 4), ("RIGHT",), ("RIGHT",), ("RIGHT",), ("SELECT",),
        ],
    },
    1: {  # L2 (rotation=180, grid=16x16)
        "rotation": 180, "grid": (16, 16),
        "game_actions": [
            ("CLICK", 7, 9),
            ("LEFT",), ("LEFT",),
            ("UP",), ("UP",), ("UP",), ("UP",), ("UP",), ("UP",),
            ("CLICK", 12, 11),
            ("LEFT",), ("LEFT",), ("LEFT",), ("LEFT",), ("LEFT",), ("LEFT",), ("LEFT",),
            ("SELECT",),
        ],
    },
    2: {  # L3 (rotation=180, grid=16x16)
        "rotation": 180, "grid": (16, 16),
        "game_actions": [
            ("CLICK", 4, 5), ("LEFT",),
            ("CLICK", 3, 8), ("RIGHT",), ("RIGHT",), ("DOWN",),
            ("CLICK", 12, 10), ("RIGHT",), ("RIGHT",), ("RIGHT",), ("RIGHT",),
            ("SELECT",),
        ],
    },
}

# Direction rotation maps (game coord → display coord)
# wdxitozphu from source: maps display→game. We need inverse (game→display)
DIR_GAME_TO_DISPLAY = {
    0: {},  # no transform
    1: {"UP": "LEFT", "DOWN": "RIGHT", "LEFT": "DOWN", "RIGHT": "UP"},
    2: {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"},
    3: {"UP": "RIGHT", "DOWN": "LEFT", "LEFT": "UP", "RIGHT": "DOWN"},
}

ACTION_NAMES_TO_INT = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SELECT": 5, "CLICK": 6}


def grid_to_display_pixel(gx, gy, k, grid_size):
    """Convert game grid coords to display pixel coords."""
    gw, gh = grid_size
    scale = 64 // gw
    px = gx * scale + scale // 2
    py = gy * scale + scale // 2
    if k == 0: return px, py
    elif k == 1: return py, 63 - px
    elif k == 2: return 63 - px, 63 - py
    else: return 63 - py, px


def play_game():
    from arc_agi import Arcade
    from arcengine import GameAction

    INT_TO_GA = {a.value: a for a in GameAction}
    GA_RESET = GameAction.RESET

    # Find game
    arcade = Arcade()
    game_id = None
    for env_info in arcade.get_environments():
        if "sp80" in env_info.game_id:
            game_id = env_info.game_id
            break

    if not game_id:
        print("ERROR: sp80 not found")
        return

    env = arcade.make(game_id)
    fd = env.reset()
    baseline = getattr(fd, 'baseline', 0)
    print(f"sp80: {game_id}, {fd.win_levels} levels")

    writer = SessionWriter(game_id, win_levels=fd.win_levels,
                           available_actions=[1,2,3,4,5,6],
                           baseline=baseline, player="sp80_solver")

    # Get initial frame
    frame = np.array(fd.frame)
    if len(frame.shape) == 3:
        frame = frame[-1]
    writer._level_start_grid = frame

    current_level = 0

    for level in range(fd.win_levels):
        current_level = level
        if level not in SOLUTIONS:
            print(f"\nLevel {level+1}: No solution. Saving frame and stopping.")
            writer.record_step(5, frame, levels_completed=fd.levels_completed, state="PAUSED")
            break

        sol = SOLUTIONS[level]
        k = sol["rotation"] // 90 % 4
        grid_size = sol["grid"]

        print(f"\nLevel {level+1}: {len(sol['game_actions'])} actions (rot={sol['rotation']})")

        for action_tuple in sol["game_actions"]:
            action_type = action_tuple[0]

            if action_type == "CLICK":
                gx, gy = action_tuple[1], action_tuple[2]
                dx, dy = grid_to_display_pixel(gx, gy, k, grid_size)
                print(f"  CLICK grid({gx},{gy}) → display({dx},{dy})", end="")
                fd = env.step(INT_TO_GA[6], data={"x": dx, "y": dy})
                frame = np.array(fd.frame)
                if len(frame.shape) == 3: frame = frame[-1]
                writer.record_step(6, frame, levels_completed=fd.levels_completed,
                                   x=dx, y=dy, state=fd.state.name)

            elif action_type == "SELECT":
                print(f"  SELECT (pour)", end="")
                # Pour triggers animation — collect all frames
                all_frames = []
                fd = env.step(INT_TO_GA[5])
                frame = np.array(fd.frame)
                if len(frame.shape) == 3: frame = frame[-1]
                all_frames.append(frame.copy())

                prev_level = fd.levels_completed
                # Step through spill animation — RESET advances without consuming steps
                for _ in range(300):
                    fd_next = env.step(GA_RESET)
                    frame_next = np.array(fd_next.frame)
                    if len(frame_next.shape) == 3: frame_next = frame_next[-1]
                    all_frames.append(frame_next.copy())

                    if fd_next.levels_completed > prev_level:
                        fd = fd_next
                        frame = frame_next
                        break
                    if fd_next.state.name in ("LOST", "GAME_OVER", "WON"):
                        fd = fd_next
                        frame = frame_next
                        break
                    if np.array_equal(frame, frame_next) and len(all_frames) > 5:
                        fd = fd_next
                        frame = frame_next
                        break
                    fd = fd_next
                    frame = frame_next

                writer.record_step(5, frame, all_frames=all_frames,
                                   levels_completed=fd.levels_completed,
                                   state=fd.state.name)

            else:
                # Directional action
                display_dir = DIR_GAME_TO_DISPLAY.get(k, {}).get(action_type, action_type)
                action_int = ACTION_NAMES_TO_INT[display_dir]
                print(f"  {action_type} → {display_dir}", end="")
                fd = env.step(INT_TO_GA[action_int])
                frame = np.array(fd.frame)
                if len(frame.shape) == 3: frame = frame[-1]
                writer.record_step(action_int, frame,
                                   levels_completed=fd.levels_completed,
                                   state=fd.state.name)

            # Check result
            if fd.levels_completed > level:
                print(f" ★ LEVEL UP!")
                break
            elif fd.state.name in ("LOST", "GAME_OVER"):
                print(f" ✗ {fd.state.name}")
                break
            else:
                print()

        if fd.state.name in ("WON", "GAME_OVER"):
            break

    writer.record_game_end(fd.state.name if hasattr(fd.state, 'name') else "UNKNOWN",
                           levels_completed=fd.levels_completed)
    print(f"\n{'='*60}")
    print(f"Result: {fd.state.name}, Levels: {fd.levels_completed}/{fd.win_levels}")
    print(f"Visual memory: {writer.run_dir}")


if __name__ == "__main__":
    play_game()
