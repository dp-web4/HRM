# sb26 Complete Analysis — 8/8 Levels Solved
*2026-04-07, Claude Opus 4.6 (CBP), with dp interactive guidance*

## Game Overview

sb26 is a hierarchical placement/sorting puzzle. 8 levels of escalating structural complexity. Actions: CLICK (select/place/swap), SELECT (submit), UNDO.

**Solved by**: Claude Opus 4.6 playing interactively through `claude_solver.py` with visual feedback at each step.

## Level Progression and Rules

### L1: Flat Sequence (4 items)
- 4 indicator colors at top, 4 palette items at bottom, 4 empty slots in center
- Pick from palette, click slot to place. Match indicators left-to-right.
- **Rule**: indicators map to slots 1:1 in reading order

### L2: Single Hierarchy (7 items)
- 3 top slots + 4 bottom slots connected by a visual connector
- **Rule**: top row fills [ind1, ind2, ind_LAST]. Bottom row fills [ind3, ind4, ind5, ind6]. The LAST indicator goes in the TOP row's 3rd position (the parent slot). The connector means "this slot expands into these children."
- **Key insight** (dp): the connector between rows is a TREE STRUCTURE, not decoration

### L3: Multiple Hierarchy Groups (7 items)
- 3 top slots + 4 bottom slots, TWO expansion connectors (teal-bordered, brown-bordered)
- **Rule**: each expandable top slot gets the LAST indicator of its child group
- Top = [ind1, last_of_group_A, last_of_group_B]. Bottom = [group_A_children, group_B_children]

### L4: Border Color = Parent Identity (7 items)
- 5 top slots + 2 bottom slots. Green-bordered expansion box.
- **Rule**: the parent slot gets the indicator that MATCHES THE BORDER COLOR. Green border = green indicator goes in parent slot.
- **Key insight**: requires correct color perception (Andy's color fix was essential)

### L5: Multiple Parent Slots (8 items)
- 5 top + 3 bottom. Blue-bordered expansion. TWO blue items in palette.
- **Rule**: border color can occupy MULTIPLE parent slots. Number of border-colored palette items = number of parent slots.

### L6: Swap Mechanic + 2×2 Grid (9 items, swap-based)
- **PARADIGM SHIFT**: items are pre-placed (some in wrong positions). Click piece then destination to SWAP.
- 4 boxes in 2×2 grid (red, green, blue, orange borders). 3 boxes have first position pre-filled with border color.
- **Rule**: red box (container) gets child box identities IN INDICATOR READING ORDER (not spatial order). Each child box gets its paired content indicators.
- **Key insight** (dp): "the 4th box should have blue, orange, green in order that they appear in top bar"

### L7: Nested Palindrome (8 items, swap-based)
- 3 boxes stacked vertically. Indicators form a PALINDROME: red, blue, green, yellow, green, blue, red.
- **Rule**: each box = [own_color, CHILD_BOX_COLOR, own_color]. Russian nesting dolls. Red contains blue reference, blue contains green reference, green contains yellow (leaf).
- Center of palindrome = innermost leaf. Reading outward = nesting depth.

### L8: Cross-Group Identity Linking (8 items, swap-based)
- 2 boxes (red, blue). Indicators: 6 unique colors in doubled pairs.
- **Rule**: each box gets its color group + the OTHER box's identity color in the last position. Red = [red, yellow, orange, BLUE]. Blue = [blue, green, purple, RED].
- Cross-referential: each group acknowledges the other's existence.

## Cognitive Progression

| Level | Cognitive Skill | What's New |
|-------|----------------|-----------|
| L1 | Pattern matching | Flat sequence |
| L2 | Structural reading | Hierarchy (connector = tree) |
| L3 | Multi-group reasoning | Multiple sub-trees |
| L4 | Semantic color perception | Border color = identity |
| L5 | Quantitative cue reading | Count of border items = parent count |
| L6 | Paradigm recognition | Swap mechanic, 2D grid, container logic |
| L7 | Recursive symmetry | Palindrome = nesting depth |
| L8 | Cross-referential identity | Groups reference each other |

## What Vision Provides

Each level was solvable because I could SEE:
- Connectors between rows (L2) — invisible to pixel counters
- Border colors matching indicators (L4) — requires color accuracy
- 2×2 spatial arrangement (L6) — requires layout understanding
- Palindrome symmetry (L7) — requires reading the indicator pattern
- Pre-filled elements vs empty slots (L6+) — requires distinguishing structural elements from content

## What Accumulated Learning Provides

Each level used knowledge from ALL previous levels:
- L3 used L2's "last indicator = parent" rule
- L4 added "border = parent color" on top of L3's multi-group
- L5 counted border-color palette items (builds on L4)
- L6 applied container logic from L2-L5 to the new swap paradigm
- L7's palindrome reading built on L6's indicator-order lesson
- L8's cross-reference built on L7's nesting concept

## Implications for Small Model Game-Playing

### Scene Description Must Label Structure
`describe_scene()` needs to identify: connectors, border colors, pre-fills, expansion groups, grid arrangements, palindromic patterns.

### Level-Up Summaries Must Store Structural Patterns
Not "placed red, blue, green" but "parent slot gets border-color indicator. Children fill expansion left-to-right."

### Cross-Level Context Must Present Hypotheses
"Previous level: parent = border color. This level has border colors too. Test same rule first."

### Paradigm Shifts Need Detection
L1-L5 use pick-and-place. L6+ uses swap. The model needs to detect when the interaction pattern changes and adapt.

## Solution Sequences (for replay/membot)

All 8 level solutions are stored in `session.json` under `level_solutions`. Each entry contains the full action sequence, step count, and player identity.
