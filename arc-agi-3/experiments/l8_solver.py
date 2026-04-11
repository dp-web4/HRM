#!/usr/bin/env python3
"""L8 Solver for su15 vacuum simulator."""

import sys
import math
sys.path.insert(0, '/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments')
from su15_sim import *


def make_l8():
    fruits = [
        Fruit(4, 41, 3),
        Fruit(14, 43, 3),
        Fruit(23, 17, 5),
    ]
    enemies = [
        Enemy(41, 28, 1),
        Enemy(45, 45, 1),
        Enemy(27, 50, 1),
    ]
    return fruits, enemies


GOALS = [(7, 19), (56, 19), (7, 55), (56, 55)]


def sim_full(clicks, verbose=True, label=""):
    fruits, enemies = make_l8()
    sim = SuikaSimulator(fruits, enemies)
    if label and verbose:
        print(f"\n{'='*60}")
        print(f"{label} ({len(clicks)} clicks)")
        print(f"{'='*60}")
    for i, (cx, cy) in enumerate(clicks):
        r = sim.simulate_click(cx, cy)
        if verbose:
            active = [f for f in r.fruits if not f.merged]
            merge_str = "+".join(f"t{a}->t{b}" for a, b, _ in r.merges) or "-"
            kd_info = f" KD:{r.knockdowns}" if r.knockdowns else ""
            print(f"  C{i+1:2d}({cx:2d},{cy:2d}) m={merge_str}{kd_info} fl={r.flashes}")
            for f in active:
                print(f"      {f} c=({f.cx:.1f},{f.cy:.1f})")
            for e in r.enemies:
                print(f"      {e} c=({e.cx:.1f},{e.cy:.1f}) cd={e.cooldown}")
        sim.apply_result(r)

    if verbose:
        print(f"\n--- FINAL STATE ---")
        t4_on_goal = 0
        enemy_on_goal = 0
        for f in sim.fruits:
            if not f.merged:
                on_goal = False
                print(f"  {f} c=({f.cx:.1f},{f.cy:.1f})")
                for gx, gy in GOALS:
                    d = dist(f.cx, f.cy, gx, gy)
                    if d < 10:
                        print(f"    -> goal ({gx},{gy}): dist={d:.1f}", "ON GOAL!" if d < 5 else "")
                        if d < 5 and f.tier == 4:
                            t4_on_goal += 1
        for e in sim.enemies:
            print(f"  {e} c=({e.cx:.1f},{e.cy:.1f}) cd={e.cooldown}")
            for gx, gy in GOALS:
                d = dist(e.cx, e.cy, gx, gy)
                if d < 10:
                    print(f"    -> goal ({gx},{gy}): dist={d:.1f}", "ON GOAL!" if d < 5 else "")
                    if d < 5:
                        enemy_on_goal += 1
        print(f"\n  SCORE: {t4_on_goal} tier-4 on goals, {enemy_on_goal} enemies on goals")
        print(f"  NEED:  2 tier-4 on goals, 1 enemy on goal")
        if t4_on_goal >= 2 and enemy_on_goal >= 1:
            print(f"  *** LEVEL COMPLETE! ***")
    return sim


def search_push(ex, ey, ew, eh, gx, gy):
    """Search for best click to push enemy closest to goal center."""
    ecx, ecy = ex + ew/2, ey + eh/2
    best = None
    best_dist = float('inf')
    for cx in range(64):
        for cy in range(64):
            d = dist(ecx, ecy, cx, cy)
            if d < 0.5 or d > VACUUM_RADIUS:
                continue
            dx, dy = (cx - ecx)/d, (cy - ecy)/d
            speed = VACUUM_SPEED * ENEMY_VACUUM_FACTOR
            nx = ex + dx * speed * 4
            ny = ey + dy * speed * 4
            nx, ny = clamp_pos(nx, ny, ew, eh)
            nx, ny = round(nx), round(ny)
            ncx, ncy = nx + ew/2, ny + eh/2
            gd = dist(ncx, ncy, gx, gy)
            if gd < best_dist:
                best_dist = gd
                best = (cx, cy, nx, ny, ncx, ncy)
    return best, best_dist


# STRATEGY I: Push E0+E1 with one click, then 3 pushes on E2
# After click 12 (push E0+E1): E0 at (37,30), E1 at (28,19)
# E0 c=(39.5,32): dist to tier-4(kd) c=(56.5,18.5) = 21.1. At 4px/click: 5.3 clicks.
# E1 c=(30.5,21): dist to kd c=(56.5,18.5) = 26.2. At 4px/click: 6.6 clicks.
# So we have 5 more clicks before E0 reaches kd. We can do 3 E2 pushes + 2 spare.
#
# After click 12: E2 at (11,24).
# Need to find optimal 3-push sequence from (11,24) to goal (7,55).

print("="*60)
print("SEARCHING OPTIMAL 3-PUSH SEQUENCE FOR E2")
print("="*60)

# Brute force 3-push search
# Push 1: from (11,24)
ecx1, ecy1 = 13.5, 26.0
best_total = None
best_total_dist = float('inf')

# For efficiency, search push 1, then for each result search push 2, then push 3
for cx1 in range(max(0, int(ecx1)-8), min(64, int(ecx1)+9)):
    for cy1 in range(max(0, int(ecy1)-8), min(64, int(ecy1)+9)):
        d1 = dist(ecx1, ecy1, cx1, cy1)
        if d1 < 0.5 or d1 > VACUUM_RADIUS:
            continue
        dx1, dy1 = (cx1 - ecx1)/d1, (cy1 - ecy1)/d1
        speed = VACUUM_SPEED * ENEMY_VACUUM_FACTOR
        nx1 = 11 + dx1 * speed * 4
        ny1 = 24 + dy1 * speed * 4
        nx1, ny1 = clamp_pos(nx1, ny1, 5, 4)
        nx1, ny1 = round(nx1), round(ny1)

        # Push 2
        ecx2, ecy2 = nx1 + 2.5, ny1 + 2.0
        for cx2 in range(max(0, int(ecx2)-8), min(64, int(ecx2)+9)):
            for cy2 in range(max(0, int(ecy2)-8), min(64, int(ecy2)+9)):
                d2 = dist(ecx2, ecy2, cx2, cy2)
                if d2 < 0.5 or d2 > VACUUM_RADIUS:
                    continue
                dx2, dy2 = (cx2 - ecx2)/d2, (cy2 - ecy2)/d2
                nx2 = nx1 + dx2 * speed * 4
                ny2 = ny1 + dy2 * speed * 4
                nx2, ny2 = clamp_pos(nx2, ny2, 5, 4)
                nx2, ny2 = round(nx2), round(ny2)

                # Push 3: find best
                r3, d3 = search_push(nx2, ny2, 5, 4, 7, 55)
                if d3 < best_total_dist:
                    best_total_dist = d3
                    best_total = (cx1, cy1, nx1, ny1,
                                  cx2, cy2, nx2, ny2,
                                  r3[0], r3[1], r3[2], r3[3], r3[4], r3[5])

if best_total:
    print(f"Best 3-push sequence: dist={best_total_dist:.2f}")
    print(f"  Push 1: click ({best_total[0]},{best_total[1]}) -> E2@({best_total[2]},{best_total[3]})")
    print(f"  Push 2: click ({best_total[4]},{best_total[5]}) -> E2@({best_total[6]},{best_total[7]})")
    print(f"  Push 3: click ({best_total[8]},{best_total[9]}) -> E2@({best_total[10]},{best_total[11]}) "
          f"c=({best_total[12]},{best_total[13]})")

    # Now build the full solution
    clicks_full = [
        (29, 56),  # 1: push E2 down
        (10, 42),  # 2: merge
        (7, 36),   # 3: pull tier-4(m) up
        (7, 30),   # 4: pull up (knockdown)
        (7, 24),   # 5: pull up
        (7, 19),   # 6: tier-4(m) at goal
        (31, 19),  # 7: pull tier-4(kd) right
        (37, 19),  # 8: right
        (44, 19),  # 9: right
        (51, 19),  # 10: right
        (56, 19),  # 11: tier-4(kd) at goal
        (37, 21),  # 12: push E0+E1 away (both in vacuum!)
        (best_total[0], best_total[1]),   # 13: push E2 - 1st
        (best_total[4], best_total[5]),   # 14: push E2 - 2nd
        (best_total[8], best_total[9]),   # 15: push E2 - 3rd (LAST)
    ]

    sim = sim_full(clicks_full, verbose=True, label="STRATEGY I: FINAL SOLUTION")
