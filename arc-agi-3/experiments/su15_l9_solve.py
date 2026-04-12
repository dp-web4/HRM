#!/usr/bin/env python3
"""
su15 L9 solver — BFS over click sequences.

L9 Layout (level index 8):
  Fruits: 2x tier-1 at (18,46) and (23,52), 1x tier-5 at (35,48)
  Enemies: 4x type-1 at (51,13), (14,12), (15,22), (54,33)
  Goals: 3 goal zones centered at (11,41), (53,55), (11,55)
  Goal: 1x tier-4 + 1x enemy-type-3 (vptxjilzzk) + 1x tier-2, all on goals
  Steps: 48

Strategy:
  1. Knockdown tier-5 -> tier-4 via enemy collision
  2. Merge 2x tier-1 -> tier-2
  3. Merge enemies: 4x type-1 -> 2x type-2 -> 1x type-3
  4. All three goal items must be on goal zones simultaneously

Key: win check fires right after enemy merge in ivbqcpwjdw.fzolkosujg,
before chase frames. So pre-position fruits on goals, then trigger the
enemy merge that creates type-3 on/near a goal.
"""

import sys
import copy
import itertools
import time

sys.path.insert(0, 'environment_files/su15/4c352900')
from su15 import Su15, GameAction, ActionInput


def make_game():
    """Create a fresh game at level 9 (index 8)."""
    game = Su15()
    game.set_level(8)
    return game


def click(game, x, y):
    """Perform a click action."""
    action = ActionInput(id=GameAction.ACTION6.value, data={'x': x, 'y': y})
    game.perform_action(action)


def get_state(game):
    """Get current state summary."""
    fruits = []
    for f in game.hmeulfxgy:
        t = game.amnmgwpkeb.get(f, 0)
        cx, cy = game.qmecbepbyz(f)
        fruits.append((t, f.x, f.y, cx, cy))
    enemies = []
    for e in game.peiiyyzum:
        etype = game.hirdajbmj.get(e, '?')
        cx, cy = game.qmecbepbyz(e)
        enemies.append((etype, e.x, e.y, cx, cy))
    return fruits, enemies


def state_key(game):
    """Compact hashable state key."""
    fruits = tuple(sorted((game.amnmgwpkeb.get(f, 0), f.x, f.y) for f in game.hmeulfxgy))
    enemies = tuple(sorted((game.hirdajbmj.get(e, '?'), e.x, e.y) for e in game.peiiyyzum))
    return (fruits, enemies)


def try_sequence(clicks, verbose=False):
    """Try a click sequence and return (won, state_after)."""
    game = make_game()
    for i, (x, y) in enumerate(clicks):
        if game.level_index != 8:
            return True, get_state(game)  # Won!
        click(game, x, y)
        if verbose:
            f, e = get_state(game)
            print(f"  Click {i+1} ({x},{y}): fruits={f}, enemies={e}")
    won = game.level_index != 8
    return won, get_state(game)


def explore_initial():
    """Explore what happens with single clicks to understand the dynamics."""
    game = make_game()
    f, e = get_state(game)
    print("Initial state:")
    print(f"  Fruits: {f}")
    print(f"  Enemies: {e}")
    print(f"  Goals at: (11,41), (53,55), (11,55)")
    print()

    # Try clicking near each fruit to understand vacuum behavior
    test_clicks = [
        (19, 47, "near tier-1 A"),
        (24, 53, "near tier-1 B"),
        (38, 51, "near tier-5"),
        (15, 52, "between tier-1 A and B, toward goal"),
        (20, 49, "between both tier-1s"),
        (11, 55, "on goal (11,55)"),
        (11, 41, "on goal (11,41)"),
        (53, 55, "on goal (53,55)"),
    ]

    for x, y, desc in test_clicks:
        game = make_game()
        click(game, x, y)
        f, e = get_state(game)
        print(f"  Click ({x},{y}) [{desc}]:")
        print(f"    Fruits: {f}")
        print(f"    Enemies: {e}")
        print()


def phase1_knockdown_search():
    """Search for clicks that knock tier-5 down to tier-4 via enemy collision.

    We need an enemy to reach the tier-5 at (35,48). The closest enemy is at
    (54,33) or (15,22). Enemies chase nearest fruit at 1px/frame during vacuum
    (if outside vacuum radius). But during vacuum that's only 4 frames.

    Alternative: vacuum pull an enemy through the tier-5.
    """
    print("\n=== Phase 1: Knockdown tier-5 to tier-4 ===")

    # The tier-5 is at (35,48), center (38,51), size 7x7
    # Enemy at (54,33) center (56,35) — need to pull toward tier-5
    # Enemy at (15,22) center (17,24) — need to pull toward tier-5

    # Try clicks that vacuum-pull enemies toward the tier-5
    # vacuum radius = 8, so we need click within 8 of enemy center

    results = []
    for x in range(0, 64, 2):
        for y in range(10, 63, 2):
            game = make_game()
            click(game, x, y)
            f, e = get_state(game)
            # Check if tier-5 became tier-4
            tiers = [t for t, _, _, _, _ in f]
            if 4 in tiers and 5 not in tiers:
                print(f"  Knockdown at ({x},{y})! Fruits: {f}")
                results.append((x, y, f, e))
            elif 4 in tiers:
                print(f"  Got tier-4 (but tier-5 still exists?) at ({x},{y})")

    if not results:
        print("  No single-click knockdown found. Trying multi-click...")
        # Maybe we need to first move enemies closer, then knockdown
        for x1 in range(0, 64, 4):
            for y1 in range(10, 63, 4):
                game = make_game()
                click(game, x1, y1)
                f1, e1 = get_state(game)
                # Now try knockdown
                for x2 in range(0, 64, 4):
                    for y2 in range(10, 63, 4):
                        game2 = make_game()
                        click(game2, x1, y1)
                        click(game2, x2, y2)
                        f2, e2 = get_state(game2)
                        tiers = [t for t, _, _, _, _ in f2]
                        if 4 in tiers:
                            print(f"  2-click knockdown: ({x1},{y1}),({x2},{y2}) -> {f2}")
                            results.append(((x1,y1),(x2,y2), f2, e2))
                            if len(results) >= 10:
                                return results

    return results


def phase1_enemy_approach():
    """Track how enemies move toward the tier-5 over multiple clicks."""
    print("\n=== Enemy approach tracking ===")
    game = make_game()

    # Just do empty clicks (far from everything) to let enemies chase
    for i in range(10):
        # Click in empty corner — enemies will chase nearest fruit
        click(game, 5, 60)
        f, e = get_state(game)
        t5_info = [(t,x,y) for t,x,y,_,_ in f if t == 5]
        print(f"  After click {i+1}: enemies={[(et,x,y) for et,x,y,_,_ in e]}")
        print(f"    tier-5: {t5_info}")
        print(f"    tier-1s: {[(t,x,y) for t,x,y,_,_ in f if t == 1]}")


def search_full_solution():
    """
    Systematic search for L9 solution.

    Goal: [[4,1], [vptxjilzzk,1], [2,1]]
    = 1 tier-4 on goal + 1 enemy-type-3 on goal + 1 tier-2 on goal

    Goals at centers: (11,41), (53,55), (11,55)

    Initial:
      tier-1 at (18,46), (23,52)
      tier-5 at (35,48)
      4x type-1 enemies at (51,13), (14,12), (15,22), (54,33)

    Steps needed:
    1. Knock tier-5 down to tier-4 (enemy collision)
    2. Merge 2x tier-1 -> tier-2
    3. Position tier-4 and tier-2 on goals
    4. Merge enemies: pairs -> type-2 -> type-3
    5. Position type-3 on remaining goal

    All must be simultaneous on goals for win check.
    """
    print("\n=== Full solution search ===")

    # First, let's understand what clicks are productive
    # by exploring the space systematically

    game = make_game()

    # The win check function uses epvtlqtczz which checks if center (x,y) is within goal sprite bounds
    # Goal sprite "avvxfurrqu" is 9x9
    # Goal 1: at (7,37), bounds (7,37)-(16,46), center (11,41)
    # Goal 2: at (49,51), bounds (49,51)-(58,60), center (53,55)
    # Goal 3: at (7,51), bounds (7,51)-(16,60), center (11,55)

    # For a fruit/enemy center to be "on" a goal, epvtlqtczz checks:
    #   x >= goal.x AND x < goal.x + width AND y >= goal.y AND y < goal.y + height
    # So center (cx,cy) must be within [goal.x, goal.x+9) x [goal.y, goal.y+9)

    # Goal 1: cx in [7,16), cy in [37,46)
    # Goal 2: cx in [49,58), cy in [51,60)
    # Goal 3: cx in [7,16), cy in [51,60)

    # tier-2 is 3x3, center = sprite.x + 1, sprite.y + 1
    # For tier-2 center on goal 3: x+1 in [7,16), y+1 in [51,60)
    #   -> sprite at x in [6,15), y in [50,59)

    # tier-4 is 5x5, center = sprite.x + 2, sprite.y + 2
    # For tier-4 center on goal 1: x+2 in [7,16), y+2 in [37,46)
    #   -> sprite at x in [5,14), y in [35,44)

    # enemy-type-3 is 5w x 4h, center = x+2, y+2
    # For e3 center on goal 2: x+2 in [49,58), y+2 in [51,60)
    #   -> sprite at x in [47,56), y in [49,58)

    print("Goal bounds for centers:")
    print("  Goal 1 (11,41): cx in [7,16), cy in [37,46)")
    print("  Goal 2 (53,55): cx in [49,58), cy in [51,60)")
    print("  Goal 3 (11,55): cx in [7,16), cy in [51,60)")
    print()

    # Strategy: First search for the knockdown, then build from there

    # Let's do a brute-force search over short click sequences
    # We have 48 steps budget, so up to 48 clicks

    # Key observations:
    # - tier-5 at (35,48) with center (38,51) — enemies chase it
    # - Over time, enemies will converge on the tier-5
    # - The tier-1s at (18,46) and (23,52) are small and close to goals 1 and 3

    # Let's first see how many chase clicks it takes for enemies to reach tier-5
    game = make_game()
    for i in range(20):
        # Click in empty area — each click = 1 chase step for non-vacuumed enemies
        # But wait — vacuum pulls 4 frames, then 1 enemy chase per frame
        # So each click: enemies outside vacuum get 4 chase frames
        click(game, 60, 60)  # click far away
        f, e = get_state(game)
        t5 = [(t,x,y,cx,cy) for t,x,y,cx,cy in f if t == 5]
        if not t5:
            print(f"  Click {i+1}: tier-5 gone! Fruits: {f}")
            break
        # Check if any enemy overlaps tier-5
        t5_x, t5_y = t5[0][1], t5[0][2]
        for et, ex, ey, ecx, ecy in e:
            # enemy is 5x4, tier-5 is 7x7
            # overlap check
            if ex < t5_x + 7 and ex + 5 > t5_x and ey < t5_y + 7 and ey + 4 > t5_y:
                print(f"  Click {i+1}: enemy at ({ex},{ey}) overlaps tier-5 at ({t5_x},{t5_y})")
        tiers = set(t for t,_,_,_,_ in f)
        if 4 in tiers:
            print(f"  Click {i+1}: tier-4 found! Fruits: {f}")
            print(f"    Enemies: {e}")
            break
        if i < 5 or i % 5 == 0:
            print(f"  Click {i+1}: enemies at {[(ex,ey) for _,ex,ey,_,_ in e]}")


def search_methodical():
    """
    Methodical search: try sequences of clicks and track results.

    Phase approach:
    A) Use several clicks to get enemy to knockdown tier-5 -> tier-4
    B) Merge tier-1 pair -> tier-2
    C) Position tier-4 and tier-2 on goals
    D) Merge all enemies into type-3 on a goal
    """
    print("\n=== Methodical search ===")

    best_sequences = []

    # Phase A: Knockdown
    # Enemy chase speed is 1px/frame during non-vacuum
    # Each click = 4 vacuum frames where non-vacuumed enemies chase
    # So each click moves enemies ~4px toward nearest fruit

    # Nearest enemy to tier-5 (center 38,51):
    #   (15,22) center (17,24) — dist ~30
    #   (54,33) center (56,35) — dist ~22
    #   (14,12) center (16,14) — dist ~42
    #   (51,13) center (53,15) — dist ~38

    # At 4px/click (approximate), (54,33) reaches tier-5 in ~6 clicks
    # But enemy chase is only during vacuum frames, not all the time

    # Actually, looking at the code more carefully:
    # In wwvazosegn (called from lyaaynsyhw each vacuum frame),
    # enemies NOT in vacuum AND NOT in chase set chase nearest fruit.
    # Chase speed depends on enemy type: type-1 = 1px, type-3 = 2px
    # But it's per-axis, not diagonal. Each frame: +/-speed in x, +/-speed in y

    # So actually each frame, enemy moves 1px in x toward fruit AND 1px in y
    # That's ~1.4px/frame diagonal, * 4 frames/click = ~5.6px/click

    # Let me just test: click far corners and watch enemies converge
    game = make_game()
    print("Chase convergence test (clicking (5,60) repeatedly):")
    for i in range(15):
        click(game, 5, 60)
        f, e = get_state(game)
        tiers = {t for t,_,_,_,_ in f}
        if 4 in tiers:
            print(f"  Click {i+1}: KNOCKDOWN! {f}")
            print(f"    Enemies: {e}")
            return i + 1
        t5 = [(x,y) for t,x,y,_,_ in f if t == 5]
        e_pos = [(ex,ey) for _,ex,ey,_,_ in e]
        if t5:
            print(f"  Click {i+1}: t5@{t5[0]}, enemies@{e_pos}")

    return None


def run_all():
    print("=" * 60)
    print("su15 L9 Solver")
    print("=" * 60)

    explore_initial()
    search_methodical()


if __name__ == "__main__":
    run_all()
