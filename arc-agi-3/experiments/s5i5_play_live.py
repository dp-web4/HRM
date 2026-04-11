#!/usr/bin/env python3
"""Play s5i5 L1-L7 live via ARC-AGI-3 SDK using verified solutions."""

import sys, time
sys.path.insert(0, '/Users/dennispalatov/repos/SAGE')

from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
CLICK = INT_TO_GA[6]

# Verified solutions per level (0-indexed internally, 1-indexed display)
SOLUTIONS = {
    1: [  # L1: 13 clicks
        (46,20),(46,20),(46,20),(46,20),(46,20),(46,20),(46,20),  # C4R x7
        (24,44),(24,44),(24,44),(24,44),(24,44),(24,44),          # C2R x6
    ],
    2: [  # L2: 26 clicks
        (12,57),(12,57),(12,57),(12,57),(12,57),(12,57),(12,57),(12,57),  # C1R x8
        (27,57),(27,57),(27,57),(27,57),(27,57),(27,57),(27,57),(27,57),  # C2R x8
        (42,57),(42,57),(42,57),(42,57),                                  # C3R x4
        (57,57),(57,57),(57,57),(57,57),(57,57),(57,57),                  # C4R x6
    ],
    3: [  # L3: 46 clicks
        (15,57),(15,57),(15,57),(15,57),(15,57),(15,57),(15,57),(15,57),  # C4R x8
        (10,48),(10,48),(10,48),(10,48),(10,48),(10,48),                  # C1L x6
        (53,57),(53,57),(53,57),(53,57),(53,57),(53,57),(53,57),          # C6R x7
        (48,48),(48,48),(48,48),(48,48),(48,48),                          # C3L x5
        (34,48),(34,48),(34,48),(34,48),(34,48),(34,48),(34,48),(34,48),  # C2R x12
        (34,48),(34,48),(34,48),(34,48),
        (34,57),(34,57),(34,57),(34,57),(34,57),(34,57),(34,57),(34,57),  # C5R x8
    ],
    4: [  # L4: 27 clicks (A*-solved)
        (34,54),(34,54),(34,54),                                  # CTR x3
        (6,57),(6,57),(6,57),(6,57),(6,57),(6,57),(6,57),        # BLL x7
        (34,54),(34,54),(34,54),(34,54),                          # CTR x4
        (57,48),(12,48),(57,57),(34,54),                          # TRR,TLR,BRR,CTR
        (57,48),(12,48),(57,57),(34,54),                          # TRR,TLR,BRR,CTR
        (57,48),(12,48),(57,57),(34,54),(34,54),                  # TRR,TLR,BRR,CTR,CTR
    ],
    5: [  # L5: 30 clicks (optimized)
        (51,56),(51,56),                                          # C4L x2
        (6,56),                                                    # C1L x1
        (27,56),(27,56),(27,56),                                  # C2R x3
        (42,56),(42,56),(42,56),(42,56),                          # C3R x4
        (6,56),(6,56),(6,56),(6,56),(6,56),(6,56),                # C1L x6
        (36,56),(36,56),(36,56),                                  # C3L x3
        (27,56),(27,56),(27,56),(27,56),(27,56),(27,56),          # C2R x6
        (42,56),(42,56),(42,56),                                  # C3R x3
        (57,56),(57,56),                                          # C4R x2
    ],
    6: [  # L6: 12 clicks (A*-solved)
        (15,57),                  # S1R
        (28,57),(28,57),          # S2L x2
        (49,47),(49,47),(49,47),  # R3 x3
        (34,57),(34,57),          # S2R x2
        (53,57),(53,57),          # S3R x2
        (15,57),                  # S1R
        (11,47),                  # R1
    ],
    7: [  # L7: 9 clicks (rotation trick)
        (32,58),  # CYN-R
        (10,51),  # WHT-R
        (17,51),  # WHT+
        (17,51),  # WHT+
        (32,51),  # MRN-R
        (32,51),  # MRN-R
        (32,51),  # MRN-R
        (32,51),  # MRN-R
        (39,51),  # MRN+
    ],
}

def main():
    arcade = Arcade()

    # Find s5i5
    game_id = None
    for env_info in arcade.get_environments():
        if 's5i5' in env_info.game_id:
            game_id = env_info.game_id
            break

    if not game_id:
        print("ERROR: s5i5 not found in API")
        return

    env = arcade.make(game_id)
    fd = env.reset()
    print(f"Game: {game_id}, {fd.win_levels} levels")
    print(f"State: {fd.state.name}, levels_completed: {fd.levels_completed}")

    total_actions = 0

    for level_num in range(1, 8):  # L1-L7
        clicks = SOLUTIONS[level_num]
        level_start = fd.levels_completed

        print(f"\nLevel {level_num}: {len(clicks)} clicks...")

        for i, (x, y) in enumerate(clicks):
            if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                print(f"  Game ended: {fd.state.name}")
                break

            fd = env.step(CLICK, data={"x": x, "y": y})
            total_actions += 1

            if fd.levels_completed > level_start:
                print(f"  ★ Level {level_num} complete after {i+1} clicks! (total: {total_actions})")
                break
        else:
            if fd.levels_completed == level_start:
                print(f"  Level {level_num}: {len(clicks)} clicks done, NO level-up")
                print(f"  State: {fd.state.name}")

        if fd.state.name in ("WON", "LOST", "GAME_OVER"):
            break

    print(f"\n{'='*40}")
    print(f"Final: {fd.state.name}, {fd.levels_completed}/{fd.win_levels} levels")
    print(f"Actions used: {total_actions}")

    if fd.state.name == "WON":
        print("★★★ GAME WON! ★★★")

if __name__ == "__main__":
    main()
