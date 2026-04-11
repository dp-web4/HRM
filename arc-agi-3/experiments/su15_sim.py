#!/usr/bin/env python3
"""
su15 Suika-style merge game — Vacuum Simulator & Solver

Grid: 64x64. Click triggers vacuum at radius 8.
Fruits merge when same-tier touching after vacuum. Enemies knock fruits down a tier.

Sprite sizes: tier0=1x1, tier1=2x2, tier2=3x3, tier3=4x4, tier4=5x5, tier5=7x7
Enemy type 1: 5x4 sprite

Vacuum mechanics (4 frames):
  - Fruits initially in radius: pulled toward click at 4px/frame (converge, no overshoot)
  - Type 1 enemies initially in radius: stored direction = (click - enemy_center),
    speed = 4*0.85 = 3.4px/frame. They move in that fixed direction for 4 frames,
    overshooting the click point (pull-through effect).
  - Enemies NOT in radius: chase nearest fruit at 1px/frame (recalculated each frame)

After vacuum:
  1. Round positions to integers
  2. Merge check: same-tier fruits touching → next tier at centroid (cascading)
  3. Flash check: different-tier fruits touching → penalty
  4. Enemy knockdown: enemy overlapping fruit → fruit drops 1 tier (or destroyed if tier 0)

Two sprites "touch" if bounding boxes overlap or share an edge (adjacent).
"""

import math
import copy
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 64
VACUUM_RADIUS = 8
VACUUM_FRAMES = 4
VACUUM_SPEED = 4.0        # px/frame for fruits
ENEMY_VACUUM_FACTOR = 0.85  # enemies move at 85% of vacuum speed
ENEMY_CHASE_SPEED = 1.0   # px/frame when not in vacuum
ENEMY_COOLDOWN_FRAMES = 8

TIER_SIZES = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 7}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Fruit:
    x: float
    y: float
    tier: int
    merged: bool = False

    @property
    def size(self) -> int:
        return TIER_SIZES[self.tier]

    @property
    def cx(self) -> float:
        return self.x + self.size / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.size / 2.0

    def bbox(self) -> Tuple[float, float, float, float]:
        s = self.size
        return (self.x, self.y, self.x + s, self.y + s)

    def __repr__(self):
        return f"Fruit(t{self.tier} @{self.x:.0f},{self.y:.0f})"


@dataclass
class Enemy:
    x: float
    y: float
    etype: int
    cooldown: int = 0
    w: int = 5
    h: int = 4

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def __repr__(self):
        return f"Enemy(t{self.etype} @{self.x:.0f},{self.y:.0f})"


@dataclass
class SimResult:
    fruits: List[Fruit]
    enemies: List[Enemy]
    merges: List[Tuple[int, int, int]]  # (tier_from, tier_to, count)
    knockdowns: List[Tuple[int, int]]   # (fruit_idx, old_tier)
    flashes: int = 0
    enemy_fruit_collisions: int = 0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def dist(x1, y1, x2, y2) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def touching(a_bbox, b_bbox) -> bool:
    """Overlap or edge-sharing (gap <= 0 in both axes)."""
    al, at, ar, ab = a_bbox
    bl, bt, br, bb = b_bbox
    return max(al - br, bl - ar) <= 0 and max(at - bb, bt - ab) <= 0


def overlap(a_bbox, b_bbox) -> bool:
    """Strict overlap (not just adjacent)."""
    al, at, ar, ab = a_bbox
    bl, bt, br, bb = b_bbox
    return al < br and ar > bl and at < bb and ab > bt


def clamp_pos(x, y, w, h):
    return max(0, min(GRID_SIZE - w, x)), max(0, min(GRID_SIZE - h, y))


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

class SuikaSimulator:
    def __init__(self, fruits: List[Fruit], enemies: List[Enemy]):
        self.fruits = [copy.deepcopy(f) for f in fruits]
        self.enemies = [copy.deepcopy(e) for e in enemies]

    def simulate_click(self, cx: float, cy: float, verbose: bool = False) -> SimResult:
        """Simulate a vacuum click. Returns new state."""
        fruits = [copy.deepcopy(f) for f in self.fruits]
        enemies = [copy.deepcopy(e) for e in self.enemies]

        # --- Phase 1: Vacuum (4 frames) ---
        # Determine which entities are initially in vacuum radius
        fruit_in_vacuum = [
            (not f.merged and dist(f.cx, f.cy, cx, cy) <= VACUUM_RADIUS)
            for f in fruits
        ]

        enemy_stored_dir = []
        enemy_in_vacuum = []
        for e in enemies:
            d = dist(e.cx, e.cy, cx, cy)
            in_vac = (d <= VACUUM_RADIUS and d > 0.1)
            enemy_in_vacuum.append(in_vac)
            if in_vac:
                enemy_stored_dir.append(((cx - e.cx) / d, (cy - e.cy) / d))
            else:
                enemy_stored_dir.append((0, 0))

        for frame in range(VACUUM_FRAMES):
            # Fruits: recalculate direction toward click each frame, no overshoot
            for i, f in enumerate(fruits):
                if f.merged or not fruit_in_vacuum[i]:
                    continue
                d = dist(f.cx, f.cy, cx, cy)
                if d > 0.5:
                    dx, dy = (cx - f.cx) / d, (cy - f.cy) / d
                    move = min(VACUUM_SPEED, d)
                    f.x += dx * move
                    f.y += dy * move
                    f.x, f.y = clamp_pos(f.x, f.y, f.size, f.size)

            # Enemies
            for i, e in enumerate(enemies):
                if enemy_in_vacuum[i]:
                    # Stored direction, overshoot (pull-through)
                    edx, edy = enemy_stored_dir[i]
                    speed = VACUUM_SPEED * ENEMY_VACUUM_FACTOR
                    e.x += edx * speed
                    e.y += edy * speed
                    e.x, e.y = clamp_pos(e.x, e.y, e.w, e.h)
                else:
                    # Chase nearest fruit
                    nearest, nearest_d = None, float('inf')
                    for f in fruits:
                        if f.merged:
                            continue
                        fd = dist(e.cx, e.cy, f.cx, f.cy)
                        if fd < nearest_d:
                            nearest_d, nearest = fd, f
                    if nearest and nearest_d > 0.1:
                        ndx = (nearest.cx - e.cx) / nearest_d
                        ndy = (nearest.cy - e.cy) / nearest_d
                        e.x += ndx * ENEMY_CHASE_SPEED
                        e.y += ndy * ENEMY_CHASE_SPEED
                        e.x, e.y = clamp_pos(e.x, e.y, e.w, e.h)

        if verbose:
            print(f"  After vacuum: fruits={[str(f) for f in fruits if not f.merged]}")
            print(f"  After vacuum: enemies={enemies}")

        # --- Phase 2: Round positions ---
        for f in fruits:
            f.x, f.y = round(f.x), round(f.y)
        for e in enemies:
            e.x, e.y = round(e.x), round(e.y)

        # --- Phase 3: Merge (cascading) ---
        merges = []
        merge_happened = True
        while merge_happened:
            merge_happened = False
            active = [f for f in fruits if not f.merged]
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    fi, fj = active[i], active[j]
                    if fi.merged or fj.merged:
                        continue
                    if fi.tier == fj.tier and touching(fi.bbox(), fj.bbox()):
                        new_tier = fi.tier + 1
                        if new_tier > 5:
                            fi.merged = fj.merged = True
                            continue
                        new_size = TIER_SIZES[new_tier]
                        new_x = round((fi.cx + fj.cx) / 2.0 - new_size / 2.0)
                        new_y = round((fi.cy + fj.cy) / 2.0 - new_size / 2.0)
                        new_x, new_y = clamp_pos(new_x, new_y, new_size, new_size)
                        fi.merged = fj.merged = True
                        fruits.append(Fruit(x=new_x, y=new_y, tier=new_tier))
                        merges.append((fi.tier, new_tier, 1))
                        merge_happened = True
                        break
                if merge_happened:
                    break

        # --- Phase 4: Flash check ---
        flashes = 0
        active = [f for f in fruits if not f.merged]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if active[i].tier != active[j].tier and touching(active[i].bbox(), active[j].bbox()):
                    flashes += 1

        # --- Phase 5: Enemy knockdown ---
        knockdowns, enemy_collisions = [], 0
        for e in enemies:
            if e.cooldown > 0:
                e.cooldown -= 1
                continue
            for f in active:
                if f.merged:
                    continue
                if overlap(e.bbox(), f.bbox()):
                    old_tier = f.tier
                    if f.tier == 0:
                        f.merged = True
                    else:
                        f.tier -= 1
                    knockdowns.append((fruits.index(f), old_tier))
                    e.cooldown = ENEMY_COOLDOWN_FRAMES
                    enemy_collisions += 1
                    break

        return SimResult(
            fruits=[f for f in fruits if not f.merged],
            enemies=enemies,
            merges=merges,
            knockdowns=knockdowns,
            flashes=flashes,
            enemy_fruit_collisions=enemy_collisions,
        )

    def apply_result(self, result: SimResult):
        self.fruits = result.fruits
        self.enemies = result.enemies


# ---------------------------------------------------------------------------
# Level setup
# ---------------------------------------------------------------------------

def make_l5():
    """L5: 4 tier-1 fruits, 4 tier-0 fruits, 2 type-1 enemies.
    Goal: [3,1] = get 1 tier-3 fruit to goal at (32,15)."""
    fruits = [
        Fruit(44, 53, 0), Fruit(14, 54, 0), Fruit(58, 59, 0), Fruit(3, 60, 0),
        Fruit(6, 25, 1), Fruit(42, 26, 1), Fruit(53, 26, 1), Fruit(14, 28, 1),
    ]
    enemies = [Enemy(4, 37, 1), Enemy(46, 37, 1)]
    return fruits, enemies


def check_goal(fruits: List[Fruit], goal_tier: int, goal_count: int) -> bool:
    return sum(1 for f in fruits if not f.merged and f.tier >= goal_tier) >= goal_count


# ---------------------------------------------------------------------------
# L5 Solution
# ---------------------------------------------------------------------------

L5_SOLUTION = [
    (7, 44),   # 1: Push E1 downward (pull-through to y~51)
    (47, 42),  # 2: Push E2 downward (pull-through to y~47)
    (48, 27),  # 3: Merge C+D → tier-2 (enemies far at y>47)
    (10, 26),  # 4: Merge A+B → tier-2 (enemies far at y>50)
    (42, 23),  # 5: Pull tier-2R toward goal (alternating R/L)
    (16, 23),  # 6: Pull tier-2L toward goal
    (36, 19),  # 7: Pull tier-2R toward goal
    (22, 20),  # 8: Pull tier-2L toward goal
    (28, 20),  # 9: Merge tier-2 pair → tier-3 at (26,18)
    (32, 15),  # 10: Move tier-3 to goal position (32,15)
]

L5_SOLUTION_NOTES = """
Strategy: Push enemies down first, then merge pairs, then converge.

Phase 1 - Enemy clearance (2 clicks):
  Clicks 1-2 push both type-1 enemies down to y>47 using the pull-through
  mechanic. Enemy stored direction = (click - center), 85% speed, 4 frames.
  Starting at y=37, they overshoot the click by ~13px, ending at y>50.

Phase 2 - Tier-1 merges (2 clicks):
  Click 3: Merge C(42,26)+D(53,26). Midpoint click at (48,27). Both centers
  within radius 8 (5.1 and 5.8 px). Enemies at y>47, safe.
  Click 4: Merge A(6,25)+B(14,28). Click at (10,26). Both within radius
  (4.1 and 4.5 px). Enemies at y>50, safe.

Phase 3 - Tier-2 convergence (5 clicks):
  Two tier-2 fruits 38px apart. Alternate pulling R and L toward goal (32,15).
  Each pull moves fruit ~7px. After 4 pulls, gap is ~14px (within 2*radius).
  Click 9 merges them → tier-3 at (26,18).

Phase 4 - Position to goal (1 click):
  Click 10 pulls tier-3 from (26,18) to goal (30,13). Center = (32,15). Done.

Total: 10 clicks. No high-value fruit knockdowns (2 tier-0 knockdowns at click 3,
irrelevant to goal).
"""


def verify_solution(clicks, verbose=True):
    """Replay a click sequence and verify it achieves the goal."""
    fruits, enemies = make_l5()
    sim = SuikaSimulator(fruits, enemies)

    if verbose:
        print(f"\n=== Verifying {len(clicks)}-click solution ===")

    for i, (cx, cy) in enumerate(clicks):
        result = sim.simulate_click(cx, cy)
        if verbose:
            active = [f for f in result.fruits if not f.merged]
            tiers = {}
            for f in active:
                tiers.setdefault(f.tier, []).append(f)
            tier_str = ", ".join(f"t{t}x{len(fs)}" for t, fs in sorted(tiers.items()))
            merge_str = "+".join(f"t{a}→t{b}" for a, b, _ in result.merges) or "none"
            print(f"  Click {i+1:2d} ({cx:2d},{cy:2d}): merge={merge_str}, "
                  f"knockdown={len(result.knockdowns)}, "
                  f"ecoll={result.enemy_fruit_collisions} → [{tier_str}]")
        sim.apply_result(result)

    goal = check_goal(sim.fruits, 3, 1)
    if verbose:
        t3 = [f for f in sim.fruits if f.tier == 3 and not f.merged]
        if t3:
            print(f"\n  Tier-3 at sprite ({t3[0].x},{t3[0].y}), "
                  f"center ({t3[0].cx},{t3[0].cy})")
        print(f"  Goal met: {goal}")
    return goal, sim.fruits, sim.enemies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("su15 Suika Vacuum Simulator — L5 Solver")
    print("=" * 60)

    print(L5_SOLUTION_NOTES)

    print("Solution:", L5_SOLUTION)
    goal, fruits, enemies = verify_solution(L5_SOLUTION, verbose=True)

    if goal:
        print("\n*** L5 SOLVED ***")
    else:
        print("\nSolution did not achieve goal.")
