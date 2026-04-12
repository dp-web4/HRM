"""Final re86 solver for levels 4-8.
Verified solutions for L1-L7.
L8 remains unsolved - zone gauntlet at rows 28-32 prevents maintaining c6 color."""
import sys
sys.path.insert(0, 'environment_files/re86/4e57566e')
from re86 import Re86, hfoimipuna
from arcengine import GameAction, ActionInput
import numpy as np

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
SEL = GameAction.ACTION5

# Verified solutions L1-L6
L1 = [RIGHT]*4+[UP]*7+[SEL]+[LEFT]*2+[UP]*6
L2 = [LEFT]*3+[DOWN]*10+[SEL]+[LEFT]*6+[UP]*6+[SEL]+[LEFT]*7+[DOWN]*2
L3 = [LEFT]*2+[UP]*13+[SEL]+[LEFT]*9+[UP]*6+[SEL]+[RIGHT]*8+[UP]*8
L4 = [UP]*7+[LEFT]*13+[DOWN]*5+[SEL]+[RIGHT]*8+[DOWN]*8+[UP]*5+[LEFT]*3
L5 = ([DOWN]+[LEFT]*3+[RIGHT]*2+[UP]*10+[RIGHT]*3+
      [SEL]+[LEFT]*16+[DOWN]*6+[RIGHT]*9+
      [SEL]+[RIGHT]*6+[DOWN]*15+[RIGHT]+[UP]*9)
L6 = ([UP]*3+[RIGHT]*4+[DOWN]*4+[RIGHT]*9+[UP]*2+
      [SEL]+[LEFT]*12+[DOWN]+[RIGHT]*7+[DOWN]*5+[LEFT]*5+[UP]*6)

# L7: 3 pieces
# cross19 (active): shift arm_row 9->18 via wall, paint c11, position (36,30)
# rect13: deform to 19x7, paint c9, position (39,18)
# cross37: shift arms (arm_col 18->9, arm_row 9->6) via wall, paint c8, position (0,9)
L7_cross19 = [RIGHT]*3+[UP]*7+[LEFT]+[UP]*6+[RIGHT]*3+[DOWN]*3+[RIGHT]*6+[DOWN]*7
L7_rect = [RIGHT]*5+[UP]*5+[LEFT]*6+[UP]*11+[LEFT]+[DOWN]*2+[RIGHT]*11+[DOWN]*3
L7_cross37 = [RIGHT]*2+[UP]*4+[LEFT]*7+[UP]*9+[DOWN]+[RIGHT]*9+[UP]+[RIGHT]*3+[UP]*2+[DOWN]*2+[LEFT]*9
L7 = L7_cross19+[SEL]+L7_rect+[SEL]+L7_cross37

# L8: 2 deformable hollow rectangles
# piece 0 (active, c12, 13x13 at (45,42)): deform to 16x10, paint c6, position (6,45)
# piece 1 (c10, 13x13 at (48,39)): deform to 7x19, paint c11, position (9,39)

# piece 0 route:
# RIGHT 1 (x->48), UP 13 (deform at wall 2 to 16x10 at (45,6))
# Now at (45,6) 16x10 c12. Need c6 at (6,45).
# Go LEFT to reach zone c6 at (5,20) cols 5-9 rows 20-24.
# LEFT 13 (45->6): at x=6. DOWN to y=15 (rows 15-24 overlap zone rows 20-24).
# At x=6 cols 6-21. Zone c6 cols 5-9: overlap at 6-9. Should paint c6.
# But also zone c11 at (5,5) cols 5-9 rows 5-9: at y=6 rows 6-15 overlap zone rows 5-9.
# Going DOWN from y=6: at y=6 (current), rows 6-15 overlap c11 zone rows 5-9 at 6-9.
# Zone c11 (index 6) vs zone c6 (index 3). c6 has lower index but cols 5-9 rows 20-24.
# At y=6 rows 6-15: c6 zone rows 20-24 NOT overlapping (15 < 20).
# c11 zone rows 5-9: overlap at 6-9. TRIGGERS c11!

# Need to go DOWN past c11 zone first. At y=6: c11 zone rows 5-9. Need y+9 > 9 AND y > 9.
# At y=10: rows 10-19. c11 rows 5-9: 10 > 9. Safe.
# But from y=6 to y=10: DOWN from y=6 to y=9: rows 9-18. c11 rows 5-9: overlap at 9.
# DOWN to y=12: rows 12-21. c11 rows 5-9: 12 > 9. Safe.
# So need 2 DOWN to reach y=12. But at y=9, c11 zone overlap occurs.

# Alternative: go LEFT at y=6 far enough to avoid c11 zone cols, then DOWN, then RIGHT.
# c11 at (5,5) cols 5-9. At x=-4: cols -4..11. Overlap at 5-9. Still hits!
# At x=-5: cols -5..10. Still hits (5-9 overlap).
# At x=-10: cols -10..5. Zone cols 5-9: overlap at 5.
# At x=-11: cols -11..4. Zone cols 5-9: 4 < 5. NO overlap!
# center = -11+8=-3 < 0. OUT OF BOUNDS!

# Hmm. x=-8: center = 0. cols -8..7. Zone cols 5-9: overlap at 5-7.
# Can't avoid c11 zone at y=6 with this piece width.

# Different approach: deform at wall 1 instead of wall 2.
# Wall 1 at (34,55) cols 34-38 rows 55-59.
# Piece at (45,42) 13x13. Move LEFT 3 to (36,42). Overlap wall 1 cols 34-38: piece cols 36-48 overlap at 36-38.
# Move DOWN: y=45 rows 45-57. Wall rows 55-59: overlap at 55-57.
# DOWN deformation: 13x13 -> 10x16. Width 16, height 10.
# At (36,45) deformed: center (44,50). In bounds.

# Then from there: go LEFT to x=6, then UP through c6 zone.
# But we face the same zone gauntlet issue.

# Actually, let me try a COMPLETELY different approach for L8.
# What if the piece hits zone c6 on its way to the final position,
# and ends at (6,45) with c6 as the last zone?

# From the original position going LEFT and UP, we'll hit various zones.
# The key: arrive at (6,45) with c6 being the last zone touched.

# Zone c6 at (5,20) rows 20-24. If the piece passes through this zone going DOWN,
# it gets painted c6. If no other zone is hit after that, it stays c6.

# Route: deform, move to zone c6, go DOWN through it to y=45.
# From zone c6 at rows 20-24: at y=15, piece rows 15-24 overlap. Painted c6.
# Then DOWN: y=18 (rows 18-27). Zone c14 at (11,28) rows 28-32: 27 < 28. OK!
# y=21 (rows 21-30). Zone rows 28-32: overlap at 28-30. BUT cols?
# At x=6: cols 6-21. Zone c14 at (11,28) cols 11-15: overlap at 11-15.
# Zone c8 at (4,28) cols 4-8: overlap at 6-8.
# Painted c14 or c8 (whichever has lower index).
# c14 at index 2, c8 at index 10. c14 triggers.

# So we can't go DOWN from zone c6 to y=45 at x=6.
# But what if we go DOWN at a different x where no zone cols overlap at rows 28-32?
# Need piece cols to avoid ALL zone cols at rows 28-32.
# With 16-wide piece, impossible (zones tile all cols).

# WAIT. Let me check if 10x16 (width 10, height 16) would work instead.
# 13x13 -> horizontal deformation -> 10x16 (width-3, height+3).
# Width 10. Zone gaps at rows 28-32: between zone col ranges.
# Zones: -3..1, 4..8, 11..15, 18..22, 25..29, 32..36, 39..43, 50..54, 57..61
# Gaps (>= 10 wide): 44..49 = 6 wide. Nope, all gaps < 10.
# Still can't fit.

# For width 7 (after 2 deformations): 7x19.
# Gaps: 2..3 (2), 9..10 (2), 16..17 (2), 23..24 (2), 30..31 (2), 37..38 (2), 44..49 (6), 55..56 (2)
# 44..49 is 6 wide >= 7? No, 6 < 7.
# 62..end is open. At x=62: cols 62-68. Zone at 57-61: 62 > 61. But center at 62+3=65 > 63. Out of bounds.

# Width 4 (after 3 deformations): 4x22.
# Gaps: many 2-wide gaps. 44-49 is 6 wide >= 4. YES!
# At x=44: cols 44-47. None of the zones at rows 28-32 cover cols 44-47.
# Zone c8 at (39,28) cols 39-43: 44 > 43. Zone c10 at (50,28) cols 50-54: 47 < 50.
#

# But target needs 16x10 for c6 corners: (45,6),(45,21),(54,6),(54,21).
# 4x22 can't cover those. Need width 16 and height 10.

# I'm stuck. Let me think differently about L8.
# Maybe pieces DON'T need to cover all 4 corners individually.
# Target has wildcards. Let me re-examine what exactly needs to match.
# Color 6 at (45,6), (45,21), (54,6), (54,21): these are on the border of a 16x10 rect.
# Color 11 at (39,9), (39,15), (57,9), (57,15): these are on the border of a 7x19 rect.
# Can one piece cover targets of both colors? No, each piece is one color.
# Can a piece overlap only SOME targets of one color if other targets are covered by the other piece?
# Color 6 requires ALL 4 positions. Each piece can only be one color.
# So one piece must cover all 4 c6 positions, the other all 4 c11 positions.

# For c6 at (45,6), (45,21), (54,6), (54,21):
# 16x10 at (6,45): border at rows 45,54, cols 6,21.
# 10x16 at (6,45): border at rows 45,60, cols 6,15. (54,6) at row 54 - not border (row 15=60 is border).
# Doesn't work.
# So must be 16 wide, 10 tall. Width 16, height 10.

# For c11 at (39,9), (39,15), (57,9), (57,15):
# 7x19 at (9,39): border at rows 39,57, cols 9,15. YES!

# OK so I definitely need 16x10 and 7x19. The 16x10 piece can't navigate through rows 28-32.
# BUT: what if I deform the piece AFTER painting? Paint it c6 while still 13x13,
# then deform to 16x10 at the final position!

# 13x13 rect at x in [6-x..6+x]: zone c6 at cols 5-9. 13 wide at x=6: cols 6-18. Overlap at 6-9.
# 13 wide piece going DOWN through rows 28-32: cols 6-18. Zone cols:
# c8 (4-8): overlap at 6-8. c14 (11-15): overlap at 11-15. Multiple zones hit!

# Actually let me try width 13 through a gap.
# Zones at rows 28-32 cols: ..., 4-8, 11-15, 18-22, ...
# At x=9: cols 9-21. Zone 11-15: overlap. Zone 18-22: overlap at 18-21.
# At x=-1: cols -1-11. Zone -3..1: overlap. Zone 4-8: overlap. Zone 11-15: overlap at 11.
# No gap fits 13 either.

# NEW IDEA: paint the piece, then move it AROUND the wall/zone area by going WAY left
# or right, beyond all zones.

# The zones at rows 28-32 extend from cols -3 to 61. But cols 44-49 are gap-free.
# Actually cols 44-49 have no zone. 13 wide piece at x=44: cols 44-56.
# Zone c10 (50-54): overlap at 50-54. BAD.
# At x=44 cols 44-56: zone at 50-54 overlaps.

# What about going BELOW y=59 (wall rows 55-59)? Piece rows 60+ are past all zones.
# Zone rows are all <= 32 (except zones at row 28).
# Actually, zones at (-3,28) rows 28-32, (50,28) rows 28-32, etc. Max zone row = 32.
# Below y=33: no zones until wall at rows 55-59.
# And no zones at rows 33-54 at all!

# So: paint piece c6 at y <= 24 (zone c6 rows 20-24), then go DOWN past zone area,
# then RIGHT/LEFT to x=6, then continue to y=45. But crossing rows 28-32 is the problem.

# Wait, I just said zones max row = 32. So at y=33, piece rows 33-42 are past all zones.
# And at y=45, rows 45-54 are also past zones.
# The issue is getting FROM y=24 (c6 zone) TO y=33 without crossing zone rows 28-32.
# 24 to 33 crosses rows 28-32.

# UNLESS: the piece is positioned so no pixels overlap any zone at rows 28-32.
# Piece is 13x13 (before deformation) or 16x10 (after).
# 13x13 at some x: border cols x and x+12, border rows y and y+12.
# Going DOWN from y=24: at y=25, border row 25 (top) and 37 (bottom, out of zone).
# At y=28: border row 28 and 40. Row 28 in zone rows 28-32!
# Border row 28 spans cols x to x+12. These cols overlap various zones.

# For 13x13, need x to x+12 to avoid ALL zone cols at rows 28-32.
# Zones cols: -3..1, 4..8, 11..15, 18..22, 25..29, 32..36, 39..43, 50..54, 57..61
# Need 13-wide gap. Between 43 and 50: gap is 44-49 = 6 wide. Not enough.
# No 13-wide gap.

# For the CENTER ONLY: center = x+6. Center in bounds.
# Only border pixels matter for collision. Border cols: x and x+12.
# Need BOTH border cols to avoid zone cols.
# x not in any zone col range AND x+12 not in any zone col range.
# x avoids: -3..1, 4..8, 11..15, 18..22, 25..29, 32..36, 39..43, 50..54, 57..61
# Valid x: 2,3, 9,10, 16,17, 23,24, 30,31, 37,38, 44-49, 55,56, 62+
# x+12 avoids same: valid x+12: same list. So x+12 in {2,3,9,10,...}
# x+12=44: x=32. x=32 in zones 32-36. BAD.
# x+12=45: x=33. Not in any zone. x+12=45 not in zones. GOOD!
# x=33: not in -3..1,4..8,...,32..36(32-36 includes 33!). BAD.
# x+12=9: x=-3. In -3..1. BAD.
# x+12=10: x=-2. In -3..1? -2 in [-3,1]? YES. BAD.
# x+12=2: x=-10. Not in any zone. -10 < -3. x=-10: center=-10+6=-4 < 0. BAD.
# x+12=44: x=32. In 32-36. BAD.
# x+12=55: x=43. In 39-43. BAD.
# x+12=56: x=44. Not in any zone! x=44, center=50. In bounds.
#   x=44: zone 39-43: 44>43. Zone 50-54: 44<50. Safe.
#   x+12=56: zone 50-54: 56>54. Zone 57-61: 56<57. Safe.
# YES! x=44 works!

# At x=44: border cols 44 and 56. Both avoid all zone cols.
# Border rows at y to y+12. Zone rows 28-32: border rows pass through but cols are safe.
# So crossing rows 28-32 at x=44 is safe!

# Updated route for piece 0:
# 1. Move RIGHT 1 (45->48), UP 13 (deform to 16x10 at (45,6))
# Wait, I need the piece BEFORE deformation to paint, or after?
# If I paint 13x13 then deform, the deformation rebuilds pixels. Border color stays.
# Let me paint c6 while 13x13, then deform at final position.

# Actually, deformation creates new pixel array. All border pixels get body color.
# So if painted c6 before deformation, after deformation body color is c6.
# Because hfoimipuna gets the first non-zero non-transparent pixel = body color.
# After painting, all border pixels are c6 (body color = 6).
# After deformation, new border pixels are set to body color (6). Still c6!
# So paint first, then deform. That works!

# New plan:
# 1. Move piece to x=44 (safe col for crossing zones)
# 2. Move DOWN to overlap c6 zone at rows 20-24
# 3. Painted c6
# 4. Move DOWN through rows 28-32 (safe at x=44)
# 5. Move DOWN to near wall 1 for deformation
# 6. Deform at wall 1
# 7. Move to final position (6,45)

# Piece 0 at (45,42). Move LEFT 1 to x=42. Not x=44.
# From 45: 45-3=42, 42+3=45. Can't reach 44 from 45 in steps of 3!
# 45-3*n: 45,42,39,36,33,30,27,24,21,18,15,12,9,6,3,0
# Can't reach 44. Hmm.

# Actually, for 13x13: I need x and x+12 both in safe cols.
# x=42: 42 in zone 39-43. BAD.
# x=45: 45+12=57 in zone 57-61. BAD.
# x=33: in zone 32-36. BAD.
# x=30: 30+12=42 in zone 39-43. BAD.
# x=48: 48+12=60 in zone 57-61. BAD.

# Hmm none of the reachable x values (from 45 in steps of 3) give safe cols.
# x=24: 24+12=36 in zone 32-36. BAD.
# x=27: 27 in zone 25-29. BAD.
# x=21: 21+12=33 in zone 32-36. BAD.

# What about border ROW only? At x=45, border row at y passes through.
# Border row spans cols 45-57. Zone cols at rows 28-32: 50-54 overlap at 50-54.
# BAD.

# x=0: 0+12=12. Zone 11-15: 12 in range. BAD.
# x=3: 3+12=15. Zone 11-15: 15 in range. BAD.
# x=6: 6+12=18. Zone 18-22: 18 in range. BAD.
# x=9: 9+12=21. Zone 18-22: 21 in range. BAD.
# x=12: 12+12=24. Zone 25-29: 24<25. zone 18-22: 24>22. x=12 in zone 11-15. BAD.

# EVERY reachable x from 45 (step 3) has at least one border col in a zone!
# This is by design - the zones are placed to cover all positions divisible by 3.

# So I CAN'T avoid zones at rows 28-32 with a 13x13 piece.
# The only option is to make sure c6 is the LAST zone hit before final position.
# That means c6 zone must be BELOW (higher y) than the other zones.
# c6 zone is at rows 20-24. Other zones at rows 28-32. c6 is ABOVE them.
# Going DOWN: hit c6 first, then zones at 28-32 repaint.
# Going UP: hit zones at 28-32 first, then c6 paints.
# So: approach c6 from BELOW (go UP through 28-32 zones, end at c6).

# I already tried this. The problem is multiple repaints during UP.
# But the FINAL paint is c6 (last zone hit going UP is c6 at rows 20-24).
# Then after c6, continue UP to wherever, then reverse course.
# Going DOWN from above c6: first hit c6 (repaint c6, same color, skip).
# Then hit zones at 28-32. Repainted!

# The fundamental issue is: c6 zone is between the starting area (below) and
# the zone gauntlet at rows 28-32. There's no way to paint c6 and then
# go to y=45 without crossing rows 28-32.

# UNLESS: the piece is at y >= 33 when painted c6. How?
# Zone c6 at rows 20-24. Piece at y=33 means piece rows 33-45 (for 13x13).
# Zone rows 20-24: no overlap (33 > 24).
# Can't paint c6 from below the zone without overlapping it.

# Wait - what about painting via a DIFFERENT mechanism?
# The piece can be painted if it overlaps the zone while MOVING.
# If the piece is moving DOWN through the zone, it overlaps during the move.
# Piece at y=12 (rows 12-24) moves DOWN to y=15 (rows 15-27): zone rows 20-24 overlap at 20-24.
# Painted c6. Then at y=15, are there any other zones?
# Zones at rows 28-32: piece rows 15-27, max 27 < 28. NO!
# Then DOWN to y=18: rows 18-30. Zone rows 28-32: overlap at 28-30. Other zones HIT.

# So between y=15 and y=18 (one DOWN move), we cross from zone-free to zone-hit.
# At y=15, piece bottom at 27. Next DOWN: y=18, bottom at 30. Zone rows 28-32: 30 in range.

# If we could somehow skip y=18-28... but we can only move in steps of 3.
# y=15 -> 18 -> 21 -> 24 -> 27 -> 30 -> 33
# At y=18: rows 18-30. Zone rows 28-32: overlap at 28-30. HIT.
# At y=15: rows 15-27. Safe.

# Can I deform BEFORE painting? If I deform to 16x10 first (height 10 instead of 13):
# At y=15, piece rows 15-24. Zone c6 rows 20-24: overlap. Painted c6.
# DOWN: y=18, rows 18-27. Zone rows 28-32: 27 < 28. SAFE!
# y=21, rows 21-30: overlap at 28-30. HIT.
# y=24, rows 24-33: overlap at 28-32. HIT.

# With height 10: y=15 to y=18 is safe. y=18 to y=21 is safe (rows 21-30, 30 in zone).
# Actually y=18: rows 18-27. Zone 28-32: 27 < 28. SAFE!
# y=21: rows 21-30. Zone 28-32: overlap at 28-30. HIT.

# So with height 10, I get ONE extra safe step. Still not enough.

# Height 7 (3 deformations): y=15 rows 15-21. DOWN: y=18 rows 18-24. Zone 28: 24<28 SAFE.
# y=21 rows 21-27. SAFE. y=24 rows 24-30. Zone 28-32: overlap at 28-30. HIT.
# Height 4 (4 deformations): y=15 rows 15-18. DOWN: y=18 rows 18-21. y=21 rows 21-24.
# y=24 rows 24-27. y=27 rows 27-30. Zone 28-32: overlap. HIT.

# With any height, eventually rows overlap zone area at 28-32. Only delay, not prevent.

# FINAL APPROACH: Use the zone iteration order to our advantage.
# When multiple zones overlap, the FIRST one with different color triggers.
# If the piece is already c6 and hits multiple zones, c6 zone is skipped (same color).
# The first OTHER zone triggers.
# But we want c6 to stay. So we need no OTHER zone to overlap.

# We established that no safe x exists for full piece width. BUT:
# With height 10 piece at y=21 (rows 21-30): zones at 28-32 only overlap at 28-30.
# At these rows, WHICH zone triggers depends on border col positions.
# What if the border cols happen to hit a zone with color 6? Then same color, skip!
# Zone c6 at (5,20) rows 20-24: at rows 28-32, this zone has NO pixels. Can't help.

# Or: what if we paint the piece c6, then repaint to another color, then repaint c6 again
# by hitting zone c6 one more time? But zone c6 is at rows 20-24, which is BEFORE rows 28-32.

# I think the only approach is: deform the piece such that when going DOWN through rows 28-32,
# the LAST zone hit is one that happens to produce c6 via another mechanism.
# But there's no c6 zone at rows 28-32.

# OR: Create a different route. What if the piece goes DOWN through a WALL to skip rows?
# There's no mechanism for that in the game.

# Let me try the brute force: deform, paint, and accept multiple repaints.
# Keep track of what color the piece has after each step.

# Actually, I just realized: deformation creates NEW pixels. When a deformable piece
# is deformed, the code creates a new pixel array with np.zeros and fills borders.
# The border color is set to hfoimipuna(piece) = body color.
# So after deformation, the piece color is preserved.
# But more importantly: I can deform the piece AT the final position.

# Piece 0 needs to end at (6,45) as 16x10 c6.
# If I deform at (6,45) using wall 1 at (34,55):
# Piece 13x13 at (6,45): rows 45-57. Wall rows 55-59: overlap at 55-57.
# Cols 6-18. Wall cols 34-38: 18 < 34. NO COL OVERLAP.
# Can't deform here - no wall collision.

# Need cols to overlap wall 1 (34-38). Piece cols = x to x+12. x+12 >= 34 -> x >= 22.
# If x=22: cols 22-34. Overlap at 34. Move down: rows overlap wall.
# Actually, for deformation during DOWN: piece moves DOWN, collides with wall.
# At (22,42): DOWN to y=45, rows 45-57. Wall rows 55-59: overlap at 55-57.
# Cols 22-34. Wall cols 34-38: overlap at 34.
# Collision: piece border pixel at (row, 12) = (55-45, 12) = (10, 12).
# Pixel (10, 12) is border (right column). Canvas (55, 34). In wall. YES!
# Deformation: 13x13 -> 10x16 at (22-1, 42+3) (adjusted position).

# Then from deformed position, go LEFT to x=6.
# But 16 wide piece: border at x=22-?.

# This is extremely involved. Let me just try a full simulation to find a working L8 solution.
print("L8 requires extensive route planning. Testing L4-L7 first.")

# Run all verified levels
solutions = {
    'L1': (0, L1), 'L2': (1, L2), 'L3': (2, L3), 'L4': (3, L4),
    'L5': (4, L5), 'L6': (5, L6), 'L7': (6, L7),
}

for name, (idx, acts) in solutions.items():
    game = Re86()
    game.set_level(idx)
    for i, a in enumerate(acts):
        prev = game.level_index
        game.perform_action(ActionInput(id=a))
        if game.level_index != prev:
            print(f"{name}: PASS at step {i+1}/{len(acts)} (budget: {game.current_level.get_data('StepCounter') if game.level_index == prev+1 else '?'})")
            break
    else:
        print(f"{name}: FAIL after {len(acts)} steps")


if __name__ == '__main__':
    pass
PYEOF
