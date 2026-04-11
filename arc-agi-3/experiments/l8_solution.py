#!/usr/bin/env python3
"""
L8 Solution for su15 Suika Vacuum Game
=======================================

15 clicks (budget: 48). All goals met.

Goal: [[4,2],['yckgseirmu',1]] = 2 tier-4 fruits + 1 enemy centered on goals
Goals: (7,19), (56,19), (7,55), (56,55)

Final state:
  tier-4 center (6.5,18.5) on goal (7,19): dist 0.7
  tier-4 center (56.5,18.5) on goal (56,19): dist 0.7
  Enemy center (7.5,55.0) on goal (7,55): dist 0.5

Strategy:
  Phase 1 (1 click): Push E2 to bottom edge to clear merge area
  Phase 2 (1 click): Merge two tier-3 fruits into tier-4
  Phase 3 (4 clicks): Pull merged tier-4 up to goal (7,19)
    - E0 naturally chases tier-5 and knocks it to tier-4 at click 4
  Phase 4 (5 clicks): Pull knocked-down tier-4 right to goal (56,19)
    - E0 follows behind with cooldown protection
  Phase 5 (1 click): Push E0+E1 away from tier-4(kd) using single vacuum
    - Click (37,21) catches both E0 and E1 in radius 8
    - E0 pulled through to (37,30), E1 to (28,19) -- buys 5+ clicks
  Phase 6 (3 clicks): Push E2 to goal (7,55) via 3 pull-throughs
    - Push via left wall bounce: (11,24) -> (0,26) -> (0,40) -> (5,53)
    - Final E2 center at (7.5,55.0), dist 0.5 from goal
"""

import sys
sys.path.insert(0, '/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments')
from su15_sim import *


def make_l8():
    fruits = [
        Fruit(4, 41, 3),   # tier-3
        Fruit(14, 43, 3),  # tier-3
        Fruit(23, 17, 5),  # tier-5
    ]
    enemies = [
        Enemy(41, 28, 1),  # E0
        Enemy(45, 45, 1),  # E1
        Enemy(27, 50, 1),  # E2
    ]
    return fruits, enemies


L8_SOLUTION = [
    # Phase 1: Clear E2 from merge area
    (29, 56),  #  1: Push E2 to bottom edge via pull-through

    # Phase 2: Merge tier-3 fruits
    (10, 42),  #  2: Merge both tier-3s -> tier-4 at (8,40)

    # Phase 3: Pull merged tier-4 up to goal (7,19)
    ( 7, 36),  #  3: Pull tier-4(m) up
    ( 7, 30),  #  4: Pull up (E0 knocks tier-5 -> tier-4 here)
    ( 7, 24),  #  5: Pull up
    ( 7, 19),  #  6: tier-4(m) arrives at goal (7,19)

    # Phase 4: Pull knocked-down tier-4 right to goal (56,19)
    (31, 19),  #  7: Pull tier-4(kd) right
    (37, 19),  #  8: Pull right
    (44, 19),  #  9: Pull right
    (51, 19),  # 10: Pull right
    (56, 19),  # 11: tier-4(kd) arrives at goal (56,19)

    # Phase 5: Push E0+E1 away (both caught in same vacuum)
    (37, 21),  # 12: E0 pulled to (37,30), E1 pulled to (28,19)

    # Phase 6: Push E2 to goal (7,55) via 3 pull-throughs
    ( 6, 27),  # 13: Push E2 left to wall: (11,24) -> (0,26)
    ( 2, 33),  # 14: Push E2 down along wall: (0,26) -> (0,40)
    ( 4, 46),  # 15: Push E2 to goal: (0,40) -> (5,53), center (7.5,55)
]

GOALS = [(7, 19), (56, 19), (7, 55), (56, 55)]


def verify_l8(verbose=True):
    fruits, enemies = make_l8()
    sim = SuikaSimulator(fruits, enemies)

    if verbose:
        print("=" * 60)
        print("L8 Solution Verification — 15 clicks")
        print("=" * 60)

    for i, (cx, cy) in enumerate(L8_SOLUTION):
        r = sim.simulate_click(cx, cy)
        if verbose:
            active = [f for f in r.fruits if not f.merged]
            merge_str = "+".join(f"t{a}->t{b}" for a, b, _ in r.merges) or "-"
            kd_info = f" KD:{r.knockdowns}" if r.knockdowns else ""
            print(f"  Click {i+1:2d} ({cx:2d},{cy:2d}): merge={merge_str}{kd_info} flash={r.flashes}")
            for f in active:
                print(f"      {f} center=({f.cx:.1f},{f.cy:.1f})")
            for e in r.enemies:
                print(f"      {e} center=({e.cx:.1f},{e.cy:.1f}) cd={e.cooldown}")
        sim.apply_result(r)

    if verbose:
        print(f"\n{'='*60}")
        print("FINAL STATE")
        print(f"{'='*60}")

    t4_on_goal = 0
    enemy_on_goal = 0

    for f in sim.fruits:
        if not f.merged:
            if verbose:
                print(f"  {f} center=({f.cx:.1f},{f.cy:.1f})")
            for gx, gy in GOALS:
                d = dist(f.cx, f.cy, gx, gy)
                if d < 5 and f.tier == 4:
                    t4_on_goal += 1
                    if verbose:
                        print(f"    ON goal ({gx},{gy}): dist={d:.1f}")

    for e in sim.enemies:
        if verbose:
            print(f"  {e} center=({e.cx:.1f},{e.cy:.1f})")
        for gx, gy in GOALS:
            d = dist(e.cx, e.cy, gx, gy)
            if d < 5:
                enemy_on_goal += 1
                if verbose:
                    print(f"    ON goal ({gx},{gy}): dist={d:.1f}")

    success = t4_on_goal >= 2 and enemy_on_goal >= 1
    if verbose:
        print(f"\n  tier-4 on goals: {t4_on_goal}/2")
        print(f"  enemies on goals: {enemy_on_goal}/1")
        print(f"\n  {'*** L8 SOLVED ***' if success else 'FAILED'}")

    return success


if __name__ == "__main__":
    print("L8 Click Sequence:")
    for i, (cx, cy) in enumerate(L8_SOLUTION):
        print(f"  {i+1:2d}. ({cx}, {cy})")
    print()
    verify_l8(verbose=True)
