"""Solve re86 levels 4-8."""
import sys
sys.path.insert(0, 'environment_files/re86/4e57566e')
from re86 import Re86, sprites, levels, hfoimipuna
from arcengine import GameAction, ActionInput
import numpy as np

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
SEL = GameAction.ACTION5


def act(game, action):
    return game.perform_action(ActionInput(id=action))


def run_moves(game, level_idx, move_sequence, verbose=False):
    """Run moves. move_sequence: list of (action, count) tuples or single actions.
    perform_action handles animation internally - each call = 1 step.
    Returns (success, total_steps)."""
    game.set_level(level_idx)
    initial = game.level_index

    flat = []
    for item in move_sequence:
        if isinstance(item, tuple):
            a, c = item
            flat.extend([a] * c)
        else:
            flat.append(item)

    for i, a in enumerate(flat):
        act(game, a)

        if game.level_index != initial:
            if verbose:
                print(f"  Level {level_idx+1} PASSED at step {i+1}/{len(flat)}")
            return True, i + 1

    if verbose:
        pieces = game.current_level.get_sprites_by_tag('vfaeucgcyr')
        for p in pieces:
            try: color = int(hfoimipuna(p))
            except: color = '?'
            cv = int(p.pixels[p.height//2, p.width//2])
            print(f"  Final: {p.name} at ({p.x},{p.y}) color={color} center={cv}")
        print(f"  Win check: {game.cdjxpfqest()}")

    return game.level_index != initial, len(flat)


def analyze_level(game, level_idx):
    """Print level analysis."""
    game.set_level(level_idx)
    level = game.current_level

    print(f"\n=== LEVEL {level_idx + 1} ===")
    print(f"Step budget: {level.get_data('StepCounter')}")

    pieces = level.get_sprites_by_tag('vfaeucgcyr')
    targets = level.get_sprites_by_tag('vzuwsebntu')
    zones = level.get_sprites_by_tag('ozhohpbjxz')
    walls = level.get_sprites_by_tag('miqpqafylc')

    print(f"Pieces: {len(pieces)}")
    for i, p in enumerate(pieces):
        try: color = int(hfoimipuna(p))
        except: color = '?'
        center = int(p.pixels[p.height//2, p.width//2])
        print(f"  {i}: {p.name} at ({p.x},{p.y}) {p.width}x{p.height} color={color} center={center} tags={p.tags}")

    print(f"Targets: {len(targets)}")
    for t in targets:
        tp = t.pixels
        required = {}
        for r in range(tp.shape[0]):
            for c in range(tp.shape[1]):
                v = int(tp[r, c])
                if v != -1 and v != 4:
                    if v not in required: required[v] = []
                    required[v].append((r + t.y, c + t.x))
        wildcards = int(np.sum(tp == 4))
        print(f"  required_colors={list(required.keys())} wildcards={wildcards}")
        for color, positions in sorted(required.items()):
            print(f"    Color {color}: {positions}")

    print(f"Paint zones: {len(zones)}")
    for z in zones:
        inner = int(z.pixels[1, 1])
        print(f"  at ({z.x},{z.y}) {z.width}x{z.height} color={inner}")

    print(f"Walls: {len(walls)}")
    for w in walls:
        print(f"  at ({w.x},{w.y}) {w.width}x{w.height}")


# ============== LEVEL SOLUTIONS ==============

def solve_level_4():
    """Cross (c6) at (41,23) -> recolor to c12 via zone (28,4) -> final (2,17)
    Diamond (c10) at (14,11) -> recolor to c14 via zone (52,54) -> final (29,20)
    """
    return [
        (UP, 7), (LEFT, 13), (DOWN, 5),  # Cross: UP to y=2, LEFT to x=2 (paints c13 then c12), DOWN to y=17
        SEL,
        (RIGHT, 8), (DOWN, 8),            # Diamond: RIGHT to x=38, DOWN to y=35 (paints c14)
        (UP, 5), (LEFT, 3),              # Diamond: UP to y=20, LEFT to x=29
    ]


def solve_level_5():
    """
    L5: 3 pieces, needs color 9 and color 8 targets.
    Piece 0: fklxqevxxp (diamond 19x19, c14, deformable/nogegkgqgd) at (21,9)
    Piece 1: lsdbpojpqq (diamond 23x23, c11, ldkpywfara) at (13,31) - ACTIVE
    Piece 2: qjmaetmxgi (cross 29x29, c12) at (40,19)

    Target color 9: (6,21), (6,39), (45,33), (51,24), (51,45), (60,33)
    Target color 8: (27,51), (33,57), (36,42), (42,54)

    Plan:
    - Cross at (19,37) colored 9 covers 4 color-9 targets: (45,33),(51,24),(51,45),(60,33)
    - Diamond23 at (19,4) colored 9 covers 2 color-9 targets: (6,21),(6,39)
    - Diamond19 at (42,27) colored 8 covers 4 color-8 targets: (27,51),(33,57),(36,42),(42,54)

    Paint zones: c8 at (54,52), c9 at (3,52), c14 at (3,27), c10 at (54,3), c11 at (3,3)

    Route plan:
    1. Diamond23 (active, c11) at (13,31): DOWN 1 (y->34), LEFT 3 (x->4, hits c9 zone at ~x=7)
       -> gets painted c9. Then RIGHT 5 (x->19), UP 10 (y->4). Total: 19 moves.
    2. SEL to cross (piece 2).
    3. Cross (c12) at (40,19): need to go to (19,37) colored c9.
       Route: go DOWN to overlap c9 zone at (3,52). Cross arm at row py+14.
       For row in [52,57]: py in [38,43]. py=40: from 19, DOWN 7 (19->40).
       Then LEFT to x=19: from 40, LEFT 7 (40->19).
       Cross arm at col 19+14=33, row 40+14=54. Zone c9 at (3,52) cols 3-8: 33 not in range.
       But c8 zone at (54,52) cols 54-59. Cross arm col 33 not in range.
       Hmm, need to actually pass through zone.

       Alternative: go LEFT first to have arm col overlap zone.
       Zone c9 at (3,52) cols 3-8. Cross arm at px+14. px+14 in [3,8] -> px in [-11,-6].
       So go LEFT far enough. From 40 to -8: 16 LEFT. Then DOWN to overlap zone rows.
       Zone c9 rows 52-57. Cross arm at row py+14. py+14 in [52,57] -> py in [38,43].
       From 19, DOWN 7 (19->40).
       During LEFT from 40, cross hits zone c13 at (52,4) - cols 52-57.
       Cross arm at col 40+14=54 -> in zone c13 at start! Wait, at x=40 arm at 54.
       Zone c13 at (52,4) cols 52-57: 54 in range. Zone rows 4-9.
       Cross at y=19: row arm at 33. Vertical arm rows 19-47. Zone rows 4-9: no overlap. OK safe.

       Hmm let me just let the cross pass through whatever zones during its route and check.
    """
    # Let me compute this more carefully with a simulation approach
    return _compute_l5_solution()


def _compute_l5_solution():
    """
    Active piece order: lsdbpojpqq(1) -> qjmaetmxgi(2) -> fklxqevxxp(0)

    Step 1: Diamond23 (active, c11) at (13,31) -> final (19,4) painted c9
      Route: DOWN 1 to avoid c14 zone, LEFT to x=4 (hits c9 zone), then RIGHT 5 UP 10
      But let me just go directly and let zones happen. The key is ending at right position+color.

      From (13,31): need to reach (19,4) with color 9.
      c9 zone at (3,52) cols 3-8 rows 52-57.
      Diamond23 is 23x23. Pixel (22,0) at canvas (py+22, px+0). Zone: px in [3,8], py+22 in [52,57] -> py in [30,35].
      At (4,34): pixel (22,0) at (56,4) in zone. Then go to (19,4): RIGHT 5, UP 10.

      Route: LEFT 3 (13->4), DOWN 1 (31->34), <paint c9>, UP 10 (34->4), RIGHT 5 (4->19).
      But LEFT at y=31 hits c14 zone. Going DOWN first: DOWN 1 (31->34), LEFT 3 (13->4).
      At y=34, LEFT to 4: hits c9 at x=7 (good). No c14 hit (rows 27-32, diamond at rows 34-56).
      Then UP 10 (34->4), RIGHT 5 (4->19).
      Total: 1+3+10+5 = 19 moves.

    Step 2: SEL to cross.

    Step 3: Cross (c12) at (40,19) -> final (19,37) painted c9
      c9 zone at (3,52). Cross vertical arm at px+14, horizontal arm at py+14.
      Need cross to pass through c9 zone.
      Route: DOWN to y=37 first (6 DOWN from 19). Then LEFT to x=19 (7 LEFT from 40).
      But cross at (40,37): arm col 54, arm row 51. Zone c9 cols 3-8: 54 not in range.
      Zone c8 at (54,52) cols 54-59: 54 IN range! Rows 52-57: arm row 51 not in range.
      Vertical arm rows 37-65. Zone c8 rows 52-57: overlap at rows 52-57!
      So at x=40, y=37: cross vertical arm col 54, rows 37-65 -> zone c8 (54,52) overlap!
      Cross gets painted c8 during DOWN moves.

      Then LEFT to x=19: arm col at 19+14=33. c8 zone col 54: 33 != 54, no hit.
      But we need c9, not c8! Need to pass through c9 zone.

      Alternative: go LEFT first to overlap c9 zone, then DOWN.
      From (40,19): LEFT 16 to x=-8 (arm at col 6, in c9 zone cols 3-8).
      Then DOWN 6 to y=37 (arm row 51, zone c9 rows 52-57: row 51 not in range).
      Hmm, need row overlap too. Cross at x=-8, arm col 6. Zone c9 rows 52-57.
      Cross rows 19+0 to 19+28 = 19-47. Zone rows 52-57: no overlap.

      Need both: arm col in [3,8] AND piece rows overlapping zone rows [52,57].
      px in [-11,-6], py+r in [52,57] for some r in [0,28]. So py <= 57 and py+28 >= 52.
      py >= 24 and py <= 57.

      From (40,19): LEFT 16 (to x=-8), DOWN 6 (to y=37). At y=37, rows 37-65.
      Zone c9 rows 52-57: overlap at 52-57. arm col 6 in [3,8]. YES!
      But during LEFT, cross passes through many zones. Let me just check.

      Actually, the key insight: zones pass through automatically during movement.
      Last paint zone hit determines final color. So I need to make sure the LAST zone
      the cross touches before reaching final position has color 9.

      Route: LEFT 16 (40->-8), DOWN 6 (19->37), RIGHT 9 (-8->19).
      During LEFT: cross arm col decreases from 54 to 6. Passes through:
        - c13 zone at (52,4) cols 52-57: arm col 54 at x=40, zone rows 4-9.
          Cross rows 19-47, no overlap with 4-9? YES: rows 19 onward don't reach 4.
          Actually wait, cross vertical arm spans full height. At x=40, arm col 54.
          Cross at y=19: vertical arm at (19+0, 54) to (19+28, 54) = rows 19-47.
          Zone c13 at (52,4) rows 4-9: rows 19+ doesn't include 4-9. NO overlap.
          But wait: zone c10 at (54,3) cols 54-59, rows 3-8.
          Cross at (40,19): pixel (14,14) at canvas (33, 54). Zone c10 cols 54-59: 54 in range.
          Zone c10 rows 3-8: 33 not in range. Cross rows 19-47 don't overlap 3-8. Safe.

      During DOWN from y=19 to y=37 at x=-8:
        arm col 6 in c9 zone cols [3,8]. Need row overlap: cross rows y to y+28.
        Zone c9 rows 52-57. Cross rows at y=37: 37-65. Overlap at 52-57. Hit!
        Also arm col 6 in c14 zone (3,27) cols [3,8]. Zone c14 rows 27-32.
        Cross at y=19: rows 19-47. Overlap at 27-32. Hit c14!
        So cross gets painted c14 first, then c9. Final color c9.

      During RIGHT from x=-8 to x=19 at y=37:
        arm col goes from 6 to 33. Check zones.
        c14 zone (3,27) cols 3-8: arm col 6 at x=-8, exits at x=-5 (arm=9). Then no more.
        c9 zone (3,52) cols 3-8: same, exit at x=-5.
        But wait, we're at y=37. Horizontal arm at row 51. Vertical arm rows 37-65.
        During RIGHT, any zone at cols traversed?
        c11 zone (3,3) cols 3-8: cross at y=37, rows 37-65. Zone rows 3-8: no overlap.
        c8 zone (54,52) cols 54-59: arm col at x=-8 is 6, goes up to 33 at x=19. Never reaches 54.
        Safe.

      Total: 16 + 6 + 9 = 31 moves. But during DOWN, hits c14 then c9. Last hit = c9. Good.

      Hmm but during DOWN, which zone is hit first? At y=19 going DOWN:
        - At some y, cross rows overlap c14 zone rows 27-32 with arm col 6 in zone cols 3-8.
          y + 0 <= 32 and y + 28 >= 27. y <= 32 and y >= -1. Already at y=19, overlap!
          So at y=19, first DOWN (y=22), cross rows 22-50, zone rows 27-32: overlap.
          Actually y=19: rows 19-47, zone 27-32: overlap from the start!
          But cross starts at x=40 (arm col 54) so zone cols 3-8 not in range.
          Only after LEFT to x=-8 does arm col become 6.
          At (x=-8, y=19): rows 19-47, arm col 6. Zone c14 (3,27) cols 3-8, rows 27-32.
          Overlap: arm col 6 in [3,8], rows 27-32 in [19,47]. YES - hits c14 immediately!

        Then going DOWN: at y=22, rows 22-50. Still overlaps c14 (27-32). Gets repainted c14 again (same color, no trigger).
        Eventually at y=24 (rows 24-52), zone c9 (3,52) rows 52-57: row 52 overlap at arm col 6!
        Cross is c14, hits c9 zone (color 9, different) -> repainted to c9.

      Wait, at y=24: rows 24-52. Zone c9 rows 52-57: overlap at row 52. But does the cross actually have a pixel at row 52 in the zone? The cross vertical arm goes from piece row 0 to 28. At y=24: canvas rows 24 to 52. Row 52 is piece row 28. Piece pixel at (28, 14). On canvas: (52, 6). Zone c9 cols 3-8: 6 in range. Zone rows 52-57: 52 in range. YES!

      But there's also the c14 zone overlap still. The collision check goes through ALL ozhohpbjxz zones and picks the first one where colors differ. If cross is c14 and overlaps both c14 zone (same color, skip) and c9 zone (different), c9 triggers.

      Actually looking at the code: the zone loop in ufgmbvuqnv iterates through all paint zones and triggers on the FIRST one where the piece color differs from zone color. The order matters.

      But actually after the first paint zone triggers, the function returns, so only one zone per move.

      So at y=24: cross overlaps both c14 zone (same color since already painted c14) and c9 zone (different). c14 zone skipped (same color). c9 zone triggers. Cross gets painted c9. Good!

      Then DOWN continues. At y=25 (from 24+3=27... wait, step is 3px).
      From y=19: 22, 25, 28, 31, 34, 37. That's 6 DOWN moves.
      At y=22: hits c14 (different from c12). Painted c14.
      At y=25: overlaps c14 (same now), overlaps c9? rows 25-53, zone c9 52-57: row 52-53 overlap. Painted c9.
      At y=28: overlaps c14 (c9 != c14, so triggers c14!). Painted c14 again!
      At y=31: same issue - hits c14 then needs to hit c9 again.

      This is getting complicated. Multiple repaints waste steps (animation).

      Better idea: avoid the c14 zone entirely. Route the cross so it doesn't pass through cols 3-8 at rows 27-32.

      Actually, since perform_action handles animation internally and counts as 1 step, repaints don't waste steps! Each repaint is just 1 step. So the total step count is just the number of moves.

      The only issue is whether the FINAL color is correct. Let me trace it:

      Route: LEFT 16 (40->-8 at y=19), DOWN 6 (19->37 at x=-8), RIGHT 9 (-8->19 at y=37).

      During LEFT at y=19 (rows 19-47):
        Arm col goes from 54 down to 6.
        Zone c10 (54,3) cols 54-59: arm col 54 at x=40. Zone rows 3-8 vs cross rows 19-47: no overlap.
        Zone c13 (52,4) cols 52-57: arm 54 at x=40 in range. Zone rows 4-9: no overlap.
        Zone c11 (3,3) cols 3-8: arm reaches 6 at x=-8. Zone rows 3-8: no overlap with 19-47.
        Zone c14 (3,27) cols 3-8: arm 6 at x=-8. Zone rows 27-32: overlap with 19-47! At (27-32).
        But this only happens at x=-8 (the last LEFT move). arm col at x=-8: -8+14=6. In [3,8].
        At x=-5: arm col 9. Not in [3,8].
        So c14 triggers only at x=-8.
        Cross painted from c12 to c14 at x=-8.

      During DOWN at x=-8 (arm col 6):
        y goes from 19 to 22,25,28,31,34,37.
        At each position, check c14 zone (3,27 cols 3-8 rows 27-32) and c9 zone (3,52 cols 3-8 rows 52-57):
        y=22: rows 22-50. c14 rows 27-32: overlap. c9 rows 52-57: no (50 < 52).
          Cross is c14 = same as c14 zone. No trigger.
        y=25: rows 25-53. c14 rows 27-32: overlap. c9 rows 52-57: overlap at 52-53.
          Cross is c14. c14 zone: same, skip. c9 zone: different (c14 vs c9). TRIGGER c9.
          Cross painted to c9.
        y=28: rows 28-56. c14 rows 27-32: overlap. c9 rows 52-57: overlap.
          Cross is c9. c14 zone: different (c9 vs c14). TRIGGER c14.
          Cross painted to c14.
        y=31: same: overlap both. Cross c14. c14 zone same, c9 different. Painted c9.
        y=34: same: overlap both. Cross c9. c14 different. Painted c14.
        y=37: rows 37-65. c14 rows 27-32: 37 > 32, no overlap. c9 rows 52-57: overlap.
          Cross c14. c9 different. Painted c9.

      So at y=37, cross is c9. During RIGHT from -8 to 19 at y=37:
        arm col goes from 6 to 33. At x=-8, arm=6 in c9 zone cols 3-8. Same color, no trigger.
        At x=-5, arm=9. Not in any zone cols range. Safe.
        Final: cross at (19,37) color c9.

      Total moves: 16+6+9 = 31. Plus 1 SEL.

      Step 4: SEL to diamond19 (piece 0).

    Step 5: Diamond19 (c14, deformable) at (21,9) -> final (42,27) painted c8.
      c8 zone at (54,52) cols 54-59 rows 52-57.
      Diamond19 is 19x19. Pixel (0,0) at canvas (py, px). Need some pixel in zone.
      Pixel (0,18) at canvas (py, px+18). Zone cols 54-59: px+18 in [54,59] -> px in [36,41].
      Zone rows 52-57: py in [52,57].
      At (38, 52): pixel (0,18) at (52, 56) in zone. Center at (38+9, 52+9)=(47,61) in bounds.
      From (21,9): RIGHT (21->38 = 17/3 = not exact. 21+3*6=39. 21+3*5=36).
      Actually x=21+3*6=39. Then py=9+3*n. 9+3*14=51.
      At (39,51): pixel (0,18) at (51,57). Zone rows 52-57: 51 not in range.
      At (39,54): pixel (0,18) at (54,57) in zone! py=9+3*15=54. 15 DOWN from 9.
      Center at (39+9,54+9)=(48,63). In bounds.
      But wait, deformable pieces may reshape on wall collisions! No walls in L5 though.

      Then from (39,54) to (42,27): RIGHT 1 (39->42), UP 9 (54->27).
      Total: 6 RIGHT + 15 DOWN + paint + 1 RIGHT + 9 UP = 31 moves.

      But need to check if diamond19 hits any zones on the way.
      Going RIGHT at y=9: diamond rows 9-27. Zones: c14 (3,27) rows 27-32, row 27 overlap!
      Zone c14 cols 3-8. Diamond at x going from 21 to 39: pixels at various cols.
      At x=21: pixel (18,0) at canvas (27, 21). Zone cols 3-8: 21 not in range. Safe.
      Pixel (0,0) at canvas (9, 21). Not in zone range.
      Actually, for diamond19 X-shape: offsets (r, r) and (r, 18-r).
      At x=21, y=9: (18, 0) -> canvas (27, 21). (18, 18) -> canvas (27, 39). Neither in zone cols.
      Safe during RIGHT.

      Going DOWN from y=9 to 54 at x=39:
      Diamond rows y to y+18. Arm pixels at (r, r) -> canvas (y+r, 39+r) and (r, 18-r) -> canvas (y+r, 39+18-r) = (y+r, 57-r).
      Zone c8 (54,52) cols 54-59. Pixel (r, 18-r) at canvas (y+r, 57-r).
      For col 57-r in [54,59]: r in [-2,3]. r=0: col 57, row y+0. r=1: col 56, row y+1.
      r=2: col 55, row y+2. r=3: col 54, row y+3.
      Zone rows 52-57. y+r in [52,57]:
      For r=0: y=52-57. For r=3: y=49-54.
      So overlap starts around y=49.
      At y=48 (9+3*13=48): row 0 at 48, row 3 at 51. Zone rows 52-57: no.
      At y=51 (9+3*14=51): row 0 at 51, row 3 at 54. (54, 54) in zone? cols 54-59 yes, rows 52-57 yes.
        But diamond pixel (3, 15) at canvas (54, 54). Is (3,15) a pixel? Diamond19 has offsets (r,r) and (r,18-r). (3,15): 18-3=15, so yes! Overlap!
        Cross c14 vs c8 zone: different. Painted c8.

      But I was planning 15 DOWN to y=54. If paint happens at y=51, diamond becomes c8. Then continues to y=54.
      At y=54: any other zones? c9 zone at (3,52) cols 3-8. Diamond at x=39, cols 39-57. Not in [3,8]. Safe.

      After paint at ~y=51, diamond is c8. Continue DOWN to y=54, then RIGHT 1 to x=42, UP 9 to y=27.
      Wait, (42,27) is the final position.
      From (39,54): x needs to go 39->42 (RIGHT 1). y needs to go 54->27 (UP 9).
      Total: 6+15+1+9 = 31.

      Check final position overlap: will diamond at (42,27) overlap any piece zone?
      Diamond19 at (42,27): rows 27-45, cols 42-60.
      Zones: c8 (54,52) cols 54-59 rows 52-57: rows 27-45 don't reach 52. No overlap.

    Grand total: diamond23 (19) + SEL (1) + cross (31) + SEL (1) + diamond19 (31) = 83.
    Budget: 250. Well within budget.
    """
    moves = []
    # Diamond23 (active, piece 1): (13,31) -> paint c9 -> (19,4)
    moves.extend([(DOWN, 1), (LEFT, 3)])     # to (4,34), hits c9 zone
    moves.extend([(UP, 10), (RIGHT, 5)])     # to (19,4)
    moves.append(SEL)                         # switch to cross (piece 2)

    # Cross (piece 2): (40,19) -> paint c9 -> (19,37)
    moves.extend([(LEFT, 16)])               # to (-8,19), hits c14 zone at last step
    moves.extend([(DOWN, 6)])                # to (-8,37), alternating c14/c9, ends c9
    moves.extend([(RIGHT, 9)])               # to (19,37)
    moves.append(SEL)                         # switch to diamond19 (piece 0)

    # Diamond19 (piece 0, deformable): (21,9) -> paint c8 -> (42,27)
    moves.extend([(RIGHT, 6)])               # to (39,9)
    moves.extend([(DOWN, 15)])               # to (39,54), hits c8 zone around y=51
    moves.extend([(RIGHT, 1)])               # to (42,54)
    moves.extend([(UP, 9)])                  # to (42,27)

    return moves


def solve_level_6():
    """
    L6: 2 pieces, 1 wall at (28,28) 8x8, NO paint zones.
    Piece 0: eyneohpmjj (19x19, c11, deformable/nogegkgqgd) at (6,39) - ACTIVE
    Piece 1: quydqebohi (25x25, c9) at (36,3)

    Target:
    Color 9: (6,12), (9,9), (9,30), (27,12)
    Color 11: (30,45), (30,54), (57,45), (57,54)

    No paint zones -> pieces keep original colors. c9 cross at final pos for color 9 targets.
    c11 deformable diamond at final pos for color 11 targets.

    Cross (25x25, c9) at (36,3): center at offset (12,12). Arm col px+12, arm row py+12.
    Target c9: (6,12), (9,9), (9,30), (27,12)
    If cross at px,py: arm col px+12, arm row py+12.
    Need (6,12) on arm col: px+12=12 -> px=0. Then (6,0+r)=6: r=6, piece row 6 at col 12. Yes.
    Need (27,12) on arm col: r=27, 0+27=27. Piece has 25 rows (0-24). 27 > 24. NO.

    Hmm. px=0, arm col 12. Cross rows 0-24. (6,12): r=6 yes. (27,12): r=27, out of range.

    If arm row py+12 = 9: py=-3. Arm row 9, cols -3 to 21. (9,9) at col 9 in range. (9,30): col 30 not in range (max 21). NO.

    Need arm row covering (9,9) and (9,30). Arm row at py+12=9 -> py=-3. Cols -3 to -3+24=21. Max col 21. 30 > 21. Doesn't reach.

    For (9,30) on vertical arm: px+12=30 -> px=18. Then (9,18+r)=9: r=9-py. py+r=9.
    Also need (9,9) on horizontal arm: py+12=9 -> py=-3. Then (9, px+c)=(9,9): 18+c=9 -> c=-9. Out of range.

    Two separate crosses can't cover this. One cross at (0,-3): arm col 12 (covers (6,12),(27,12)?), arm row 9 (covers (9,9)).
    Wait, (27,12): py=-3, r=30. Piece height 25 (rows 0-24). 30 > 24. No.

    Hmm, let me think differently. Maybe the cross needs to be positioned so that DIFFERENT target points are on different arms.

    Cross arms: vertical col=px+12, horizontal row=py+12.
    (6,12) on vertical: px+12=12, py+r=6 where r in [0,24] -> py <= 6 and py+24 >= 6 -> py in [-18,6].
    (9,9) on horizontal: py+12=9, px+c=9 where c in [0,24] -> px in [-15,9]. py=-3.
    (9,30) on horizontal: py+12=9, px+c=30 -> px in [6,30]. py=-3.
    (27,12) on vertical: px+12=12 -> px=0. py+r=27 -> py <= 27 and py >= 3. py in [3,27].

    For both (6,12) and (27,12) on vertical: px=0, py in [-18,6] AND py in [3,27] -> py in [3,6].
    py=3: arm row 15. (9,9) needs row 9: 15 != 9.
    (9,9) on vertical: px+12=9 -> px=-3. Conflicts with px=0.

    So the cross can cover (6,12) and (27,12) on vertical arm (px=0) OR (9,9) and (9,30) on horizontal arm (py=-3), but not all four.

    Unless the cross changes shape? The cross is not deformable (no nogegkgqgd tag). But it IS a cross-type piece that can border-shift on wall collision!

    There's a wall at (28,28) 8x8. If the cross hits the wall, border-shifting occurs. The description says: 'if the piece's leading edge overlaps the wall along one axis, a border column/row is removed from the collision side and added to the opposite side.'

    This changes the cross shape! Maybe the cross needs to be reshaped by hitting the wall to cover all 4 targets.
    """
    # Actually, let me reconsider. Maybe I need to check if the deformable diamond can cover the c11 targets.
    # c11 targets: (30,45), (30,54), (57,45), (57,54)
    # These are 4 corners of a rectangle! Row span: 30-57=27. Col span: 45-54=9.
    # Diamond19 is an X pattern. Can't cover rectangle corners.
    # But deformable pieces change shape on wall collision!
    #
    # The deformable piece at (6,39) is 19x19, hollow rectangle (nogegkgqgd).
    # When it hits a wall horizontally: becomes taller/narrower (width-3, height+3).
    # When it hits a wall vertically: becomes wider/shorter (height-3, width+3).
    #
    # Starting at 19x19 -> hit wall right -> 16x22 -> hit again -> 13x25 -> hit again -> 10x28
    # Each hit: width-3, height+3 (horizontal) or height-3, width+3 (vertical)
    # Minimum dimension 6 before deformation stops.
    #
    # For c11 targets (30,45), (30,54), (57,45), (57,54):
    # Need piece pixels at these 4 positions. Deformable piece is a hollow rectangle.
    # After deformation to say 10x28: hollow rect 10 wide 28 tall.
    # Placed at (px, py): corners at (py,px), (py,px+9), (py+27,px), (py+27,px+9).
    # For (30,45) and (30,54): py=30. px=45 and px+9=54 -> px=45. Row 30 = top edge.
    # For (57,45) and (57,54): py+27=57 -> py=30. px=45.
    # So 10x28 at (45,30)! All 4 targets on corners of the hollow rect!
    # Need: deform 19x19 into 10x28 by hitting wall 3 times horizontally.
    #
    # But wait - the hollow rect's pixels are the border. At 10x28:
    # border pixels include (0,0)=top-left, (0,9)=top-right, (27,0)=bottom-left, (27,9)=bottom-right.
    # These map to canvas (30,45), (30,54), (57,45), (57,54).

    # For the cross covering c9 targets (6,12), (9,9), (9,30), (27,12):
    # Maybe I can use wall collision to modify the cross too.
    # Or maybe a different approach. Let me check if the cross can cover all 4 with border shifting.

    # Actually wait - let me re-examine. The deformable piece is the diamond initially (X-shaped, hollow center).
    # After deformation, it becomes a hollow rectangle. The pixels form a border rectangle.

    # Actually, looking at the code for nogegkgqgd deformation:
    # Horizontal collision: height,width -> height+3, width-3. The new shape is a rectangle border.
    # So 19x19 diamond -> 22x16 -> 25x13 -> 28x10 (3 horizontal deformations).
    # At 28x10: border rect. Corners at (0,0), (0,9), (27,0), (27,9).
    # If at (45,30): canvas corners (30,45), (30,54), (57,45), (57,54). YES!

    # Now for the cross c9 targets. The cross is 25x25.
    # Cross at (-3,-3): arm col 9, arm row 9.
    # (6,12) on arm col 9: col 9 != 12. No.
    # Cross at (0,-3): arm col 12, arm row 9.
    # (6,12): arm col 12, row 6 = py+6 = -3+6=3. Piece row 3 at col 12. Piece has vertical arm at col 12 from row 0-24. Row 3 at col 12 = yes. Canvas (6, 12)? py+3 = -3+3 = 0? No, py+r = canvas_row. r=6-py = 6-(-3) = 9. Piece row 9 at col 12 = part of vertical arm. Canvas row = -3+9 = 6. Yes!
    # (9,9): arm row 9 = py+12 = -3+12 = 9. Col 9 = px+c = 0+c = 9. c=9. In [0,24]. Yes!
    # (9,30): arm row 9. Col 30 = 0+c = 30. c=30. But piece width 25, max c=24. OUT OF RANGE!
    # (27,12): arm col 12. Row 27 = -3+r. r=30. Piece height 25, max r=24. OUT OF RANGE!

    # The cross is too small! 25x25 can't reach from (6,12) to (27,12) (span 21) AND from (9,9) to (9,30) (span 21).
    # Vertical span needed: 27-6=21. Cross arm length 25. 21 < 25. OK for vertical.
    # But py=-3 so vertical arm goes from -3 to 21. (27,12): row 27. py+24=21. 27 > 21. Doesn't reach.

    # Need py such that py <= 6 and py+24 >= 27 -> py <= 6 and py >= 3. py in [3,6].
    # At py=3: arm row = 15. (9,9) needs arm row 9: 15 != 9.
    # At py=6: arm row = 18. (9,9) at row 9, but arm row is 18.

    # Can't cover both (9,9) and (6,12) + (27,12) with the standard cross.
    # Need border-shifting! When cross hits a wall, a border column/row moves.

    # Let me re-examine the cross targets. Maybe the cross needs to be modified by wall collision.
    # Let me think about what border-shifting does to a cross:
    # If cross moves RIGHT into wall: leading edge (right column) retracts, opposite side extends.
    # For a cross, the "columns" are mostly -1 except the arm. The vertical arm is at col 12.
    # Border-shifting removes a column from the collision side and adds to the opposite.

    # This is getting complex. Let me try a different approach: simulate and search.
    return _compute_l6_solution()


def _compute_l6_solution():
    """Solve L6 by computing required deformations and positions."""
    # Deformable piece (c11, 19x19) at (6,39) needs to become 28x10 hollow rect at (45,30)
    # Wall at (28,28) 8x8
    #
    # To deform: push piece into wall 3 times horizontally (RIGHT into wall from left side)
    # Wall occupies cols 28-35, rows 28-35.
    # Piece at (6,39): center at (15, 48). Move piece LEFT to approach wall from left.
    # Wait, piece is at x=6, wall at x=28. Piece is LEFT of wall. Move RIGHT into wall.
    # But piece is 19x19 at x=6, right edge at 25. Wall at 28. Need to move RIGHT 1 (3px) to x=9, edge at 28. Collides!
    # But piece center needs to be in bounds, and piece rows must overlap wall rows.
    # Piece at y=39, rows 39-57. Wall rows 28-35. No overlap!
    # Need to move piece UP so rows overlap wall.

    # Move piece UP: from y=39. Piece rows 39-57. Wall rows 28-35. Need overlap.
    # Piece at y=28: rows 28-46. Wall rows 28-35. Overlap!
    # UP from 39 to 28: (39-28)/3 = 11/3 not exact. 39-3*4=27. 39-3*3=30.
    # y=30: rows 30-48. Wall rows 28-35: overlap at 30-35. Good.
    # y=27: rows 27-45. Wall rows 28-35: overlap at 28-35. Better.

    # Actually, for deformation to trigger, piece must collide with wall after moving.
    # collides_with checks pixel overlap (non-(-1) pixels of both sprites overlapping).
    # Wall pixels are all solid. Piece border pixels must overlap wall pixels.

    # At y=27, x=6: right border of piece at x+18=24. Wall at x=28. No overlap.
    # Move RIGHT: x=9, right border at 27. Still < 28.
    # x=12, right border at 30. Wall cols 28-35. Overlap at 30!
    # But piece border pixel at col 18 (rightmost). Canvas col = 12+18=30.
    # Wall pixels at (28-35, 28-35). (28, 30): wall row 28, col 30. In wall.
    # Piece pixel at row 28-27=1 (relative), col 18. Is piece pixel (1, 18) a border pixel?
    # Piece is X-shaped (diamond/nogegkgqgd). Pixel (1, 18-1=17)? Let me check.
    # Deformable piece fklxqevxxp is 19x19 X-shape. But this is eyneohpmjj which may be different!

    # Let me look at eyneohpmjj's actual pixels.
    import sys
    sys.path.insert(0, 'environment_files/re86/4e57566e')
    from re86 import sprites as sp_dict
    s = sp_dict['eyneohpmjj']
    print(f'eyneohpmjj: {s.width}x{s.height}')
    pix = s.pixels
    for r in range(s.height):
        row = ''
        for c in range(s.width):
            v = int(pix[r,c])
            if v == -1: row += '.'
            elif v == 11: row += '#'
            else: row += str(v)
        print(f'  {r:2d}: {row}')
    return []

_compute_l6_solution()


def solve_level_7():
    return []


def solve_level_8():
    return []


if __name__ == '__main__':
    game = Re86()

    # Test L1-L3
    l1 = [(RIGHT,4),(UP,7),SEL,(LEFT,2),(UP,6)]
    ok, s = run_moves(game, 0, l1)
    print(f"L1: {'PASS' if ok else 'FAIL'} ({s} steps)")

    l2 = [(LEFT,3),(DOWN,10),SEL,(LEFT,6),(UP,6),SEL,(LEFT,7),(DOWN,2)]
    ok, s = run_moves(game, 1, l2)
    print(f"L2: {'PASS' if ok else 'FAIL'} ({s} steps)")

    l3 = [(LEFT,2),(UP,13),SEL,(LEFT,9),(UP,6),SEL,(RIGHT,8),(UP,8)]
    ok, s = run_moves(game, 2, l3)
    print(f"L3: {'PASS' if ok else 'FAIL'} ({s} steps)")

    # L4
    l4 = solve_level_4()
    ok, s = run_moves(game, 3, l4, verbose=True)
    print(f"L4: {'PASS' if ok else 'FAIL'} ({s} steps, budget=200)")
