#!/usr/bin/env python3
"""
vc33 adaptive solver — re-discovers buttons after each click.

Strategy: greedy search. At each step:
1. Find all brown(9) button positions in the frame
2. Try each button, measure pixel change + goal movement
3. Pick the best action (most goal movement, or most pixel change)
4. If stalled, try all buttons systematically
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from collections import defaultdict
import copy


def click(env, x, y):
    return env.step(GameAction.ACTION6, {'x': x, 'y': y})


def find_buttons(frame):
    """Find brown(9) pixel clusters, return (x,y) click centers."""
    brown = np.argwhere(frame == 9)
    if len(brown) == 0:
        return []
    clusters = defaultdict(list)
    for r, c in brown:
        clusters[(r // 5, c // 5)].append((r, c))
    buttons = []
    for key in sorted(clusters):
        pts = clusters[key]
        cx = int(np.mean([p[1] for p in pts]))
        cy = int(np.mean([p[0] for p in pts]))
        buttons.append((cx, cy))
    return buttons


def goal_signature(frame):
    """Get a tuple describing goal positions (small HQB-colored regions only)."""
    sig = []
    for color in [14, 11, 15]:  # teal, maroon, white — skip yellow (progress bar)
        pos = np.argwhere(frame == color)
        # Only count small clusters (goals are small, not walls/bars)
        if 0 < len(pos) <= 100:
            # Exclude row 0 (progress bar area)
            pos = pos[pos[:, 0] > 1]
            if len(pos) > 0:
                sig.append((color, int(pos[:, 0].mean()), int(pos[:, 1].mean()), len(pos)))
    return tuple(sig)


def solve_level_greedy(env, fd, level_num, max_clicks=70):
    """Solve a level using greedy button selection."""
    target_level = fd.levels_completed + 1
    frame = np.array(fd.frame)[0]
    buttons = find_buttons(frame)

    print(f"\n=== Level {level_num} ===")
    print(f"  Initial buttons: {len(buttons)} at {buttons}")
    print(f"  Goal sig: {goal_signature(frame)}")

    clicks = 0
    stall = 0
    prev_sig = goal_signature(frame)
    seen_states = set()  # Track visited states to avoid loops

    while clicks < max_clicks:
        frame = np.array(fd.frame)[0]
        buttons = find_buttons(frame)

        if not buttons:
            print(f"  No buttons found after {clicks} clicks!")
            return fd, False

        # Try each button, evaluate the result
        best_fd = None
        best_score = -1
        best_btn = None

        for bx, by in buttons:
            # We can't undo, so we need to clone the env...
            # Since we can't clone, use a different approach:
            # Just try each button in sequence and track progress
            pass

        # Since we can't clone the env, use a simpler strategy:
        # Try the first untried button, if it makes progress keep going
        # If stalled, try the next button

        # Round-robin through buttons
        btn_idx = clicks % len(buttons)
        bx, by = buttons[btn_idx]

        fd = click(env, bx, by)
        clicks += 1
        frame = np.array(fd.frame)[0]

        if fd.levels_completed >= target_level:
            print(f"  SOLVED in {clicks} clicks!")
            return fd, True

        new_sig = goal_signature(frame)
        if new_sig != prev_sig:
            stall = 0
            prev_sig = new_sig
        else:
            stall += 1

        # If we're stalling hard, try a different strategy
        if stall > len(buttons) * 2:
            print(f"  Hard stall after {clicks} clicks, sig={new_sig}")
            # Try all buttons once more
            for bx2, by2 in buttons:
                fd = click(env, bx2, by2)
                clicks += 1
                frame = np.array(fd.frame)[0]
                if fd.levels_completed >= target_level:
                    print(f"  SOLVED in {clicks} clicks!")
                    return fd, True
            stall = 0

    print(f"  FAILED after {clicks} clicks")
    return fd, False


def solve_level_smart(env, fd, level_num, max_clicks=65):
    """Solve a level with live button rediscovery after each click."""
    target_level = fd.levels_completed + 1
    frame = np.array(fd.frame)[0]
    clicks = 0

    print(f"\n=== Level {level_num} ===")

    def discover_buttons(env, fd, max_test=12):
        """Test each current button once, classify as forward/enabler."""
        nonlocal clicks
        frame = np.array(fd.frame)[0]
        buttons = find_buttons(frame)
        if not buttons:
            return fd, [], []

        sig_before = goal_signature(frame)
        forward = []
        enabler = []

        for bx, by in buttons[:max_test]:
            sig_pre = goal_signature(np.array(fd.frame)[0])
            fd = click(env, bx, by)
            clicks += 1

            if fd.levels_completed >= target_level:
                return fd, None, None  # solved!

            sig_post = goal_signature(np.array(fd.frame)[0])
            new_buttons = find_buttons(np.array(fd.frame)[0])

            if sig_post != sig_pre:
                forward.append((bx, by))
                print(f"  ({bx},{by}): FORWARD {sig_pre} -> {sig_post}")
            else:
                enabler.append((bx, by))

            # If buttons changed dramatically, rediscover
            if len(new_buttons) != len(buttons) and abs(len(new_buttons) - len(buttons)) > 2:
                print(f"  Buttons changed {len(buttons)}->{len(new_buttons)}, rediscovering...")
                return discover_buttons(env, fd, max_test=8)

        return fd, forward, enabler

    # Initial discovery
    fd, forward_btns, enabler_btns = discover_buttons(env, fd)
    if forward_btns is None:
        print(f"  SOLVED in {clicks} clicks (discovery)!")
        return fd, True

    if not forward_btns:
        print(f"  No forward buttons found!")
        return fd, False

    print(f"  Forward: {forward_btns}, Enablers: {enabler_btns}")

    # Main solve loop
    stall = 0
    fwd_idx = 0
    enabler_idx = 0

    while clicks < max_clicks:
        # Always use LIVE button positions
        frame = np.array(fd.frame)[0]
        live_buttons = find_buttons(frame)
        sig_before = goal_signature(frame)

        if not live_buttons:
            print(f"  No live buttons at click {clicks}")
            return fd, False

        # Find the closest live button to our forward target
        target_btn = forward_btns[fwd_idx % len(forward_btns)]
        # Use the live button closest to the target
        best_btn = min(live_buttons, key=lambda b: abs(b[0]-target_btn[0]) + abs(b[1]-target_btn[1]))

        fd = click(env, best_btn[0], best_btn[1])
        clicks += 1

        if fd.levels_completed >= target_level:
            print(f"  SOLVED in {clicks} clicks!")
            return fd, True

        sig_after = goal_signature(np.array(fd.frame)[0])

        if sig_after != sig_before:
            stall = 0
        else:
            stall += 1

        if stall >= 2:
            # Try enabler using live buttons
            live_buttons = find_buttons(np.array(fd.frame)[0])
            if enabler_btns and live_buttons:
                etarget = enabler_btns[enabler_idx % len(enabler_btns)]
                ebtn = min(live_buttons, key=lambda b: abs(b[0]-etarget[0]) + abs(b[1]-etarget[1]))

                fd = click(env, ebtn[0], ebtn[1])
                clicks += 1
                if fd.levels_completed >= target_level:
                    print(f"  SOLVED in {clicks} clicks!")
                    return fd, True

                # Try forward again
                live_buttons = find_buttons(np.array(fd.frame)[0])
                if live_buttons:
                    best_btn = min(live_buttons, key=lambda b: abs(b[0]-target_btn[0]) + abs(b[1]-target_btn[1]))
                    fd = click(env, best_btn[0], best_btn[1])
                    clicks += 1
                    if fd.levels_completed >= target_level:
                        print(f"  SOLVED in {clicks} clicks!")
                        return fd, True

                    new_sig = goal_signature(np.array(fd.frame)[0])
                    if new_sig != sig_after:
                        stall = 0
                    else:
                        enabler_idx += 1

            if stall >= 4:
                fwd_idx += 1
                stall = 0
                if fwd_idx >= len(forward_btns) * 3:
                    # Rediscover
                    print(f"  Rediscovering at click {clicks}...")
                    fd, forward_btns, enabler_btns = discover_buttons(env, fd, max_test=6)
                    if forward_btns is None:
                        print(f"  SOLVED in {clicks} clicks!")
                        return fd, True
                    if not forward_btns:
                        # Try every button as last resort
                        for bx, by in live_buttons:
                            fd = click(env, bx, by)
                            clicks += 1
                            if fd.levels_completed >= target_level:
                                print(f"  SOLVED in {clicks} clicks!")
                                return fd, True
                    fwd_idx = 0

    print(f"  FAILED after {clicks} clicks")
    return fd, False


def main():
    arcade = Arcade()
    env = arcade.make('vc33')
    fd = env.reset()

    total_clicks = 0

    # L1: 3 clicks at bottom-right button
    print("=== Level 1 ===")
    for i in range(3):
        fd = click(env, 61, 33)
        total_clicks += 1
    print(f"  Solved L1 in 3 clicks (levels={fd.levels_completed})")

    # L2: known sequence
    print("=== Level 2 ===")
    l2_seq = [(1, 45)] * 3 + [(1, 25), (1, 45)] * 2
    for x, y in l2_seq:
        fd = click(env, x, y)
        total_clicks += 1
    print(f"  Solved L2 in {len(l2_seq)} clicks (levels={fd.levels_completed})")

    # L3-L7: adaptive solver
    for level in range(3, 8):
        if fd.levels_completed >= 7:
            break
        fd, solved = solve_level_smart(env, fd, level)
        if not solved:
            print(f"\nFailed at level {level}")
            break

    print(f"\n{'=' * 40}")
    print(f"Final: levels={fd.levels_completed}/7, state={fd.state}")

    if fd.levels_completed >= 7:
        print("GAME COMPLETE!")


if __name__ == "__main__":
    main()
