# ARC-AGI-3 Game Strategy Guide

*Deep analysis of all 25 game source files. For each game: mechanic, win condition, cognitive skill tested, strategy, and perception requirements.*

*Generated: 2026-04-01 by CBP (5 parallel Opus agents analyzing source code)*

---

## Capability Taxonomy

Before diving into individual games, here's what the 25 games collectively test:

| Capability | Games | Count |
|-----------|-------|-------|
| **Spatial reasoning / rotation** | lp85, cn04, ar25, re86, s5i5 | 5 |
| **Path planning / navigation** | tu93, g50t, sc25, ls20, sp80 | 5 |
| **Constraint satisfaction** | ft09, sk48, sb26, tr87 | 4 |
| **Physics / causal reasoning** | bp35, su15, wa30 | 3 |
| **Sequential planning** | tn36, dc22, m0r0, lf52 | 4 |
| **Pattern matching / sorting** | vc33, cd82 | 2 |
| **Multi-agent coordination** | ka59, r11l | 2 |

Most games require **multiple capabilities** simultaneously. The scoring (RHAE²) means efficiency matters more than completion — doing 3 games well beats doing 10 games poorly.

---

## Game-by-Game Analysis

### cn04 — Jigsaw Connector Puzzle
**Mechanic**: Arrange sprites so their connector pins (color 8) overlap with pins on other sprites. Select, move, rotate pieces.
**Win condition**: Every connector pin matched to a pin on another sprite.
**Actions**: 1-4 (move selected), 5 (rotate 90° CW), 6 (click to select — needs x,y)
**Skill tested**: Spatial reasoning + connector alignment. Mental rotation of irregular shapes.
**Difficulty**: 2-3 sprites → 4 with pre-rotation → visual obfuscation (only selected sprite visible).
**Strategy**: Identify complementary pin layouts, move/rotate to align. Level 5 requires memorizing hidden shapes.
**Perception needs**: Pin locations (color 8), sprite boundaries, matched vs unmatched pins (color 3 = matched).

### cd82 — Octagonal Color-Fill Carousel
**Mechanic**: Paint a 10×10 target grid by navigating a color basket around 8 octagonal positions and firing inward to fill sectors.
**Win condition**: All non-diagonal pixels match goal pattern.
**Actions**: 1-4 (move basket), 5 (fire to paint sector), 6 (click palette to change color)
**Skill tested**: Pattern decomposition. Understand how 8 directional sectors map to fill operations.
**Strategy**: Decompose goal into 8 sectors, match each sector's color, navigate and fire.
**Perception needs**: Goal pattern sectors, current basket position/color, palette colors.

### sp80 — Fluid Flow Pipe Puzzle
**Mechanic**: Arrange pipe segments and L-corners to guide fluid from sources to receptacles. ACTION5 tests the layout.
**Win condition**: All receptacles reached by fluid AND no fluid hits walls. Max 4 test attempts.
**Actions**: 1-4 (move pipe), 5 (test/spill), 6 (click to select pipe — needs x,y)
**Skill tested**: Path planning + fluid simulation. Display can be rotated 0/90/180/270°.
**Strategy**: Trace required paths from sources to receptacles, position pipes accordingly. Account for display rotation remapping controls.
**Perception needs**: Source/receptacle/pipe/wall locations, pipe types (straight vs L), display rotation.

### m0r0 — Paired Token Navigation Maze
**Mechanic**: 2-4 tokens move simultaneously with mirrored/inverted rules. Guide them through a maze to merge at same positions.
**Win condition**: All tokens paired (occupying same cell).
**Actions**: 1-4 (all tokens move, with inversions per token type), 6 (click blockers)
**Skill tested**: Simultaneous constraint reasoning with mirrored movement. Multi-agent coordination.
**Difficulty**: Adds colored barriers with trigger switches, wall-dots that reset tokens.
**Strategy**: Map inversion rules per token, plan sequences that navigate all tokens to rendezvous points. Use trigger switches to remove barriers.
**Perception needs**: Token positions + types (which inversion), maze walls, barriers, triggers.

### sk48 — Conveyor Belt Color Matching
**Mechanic**: Manipulate linked conveyor pairs by extending/retracting to align colored block sequences.
**Win condition**: All paired conveyor positions have matching colors simultaneously.
**Actions**: 1-4 (push/pull conveyor), 6 (click to select head), 7 (undo)
**Skill tested**: Sequence alignment. Like aligning DNA sequences or a sliding puzzle.
**Strategy**: Read sequences on each linked pair, compute extend/retract operations to align. Monitor indicator dots (green = match).
**Perception needs**: Conveyor heads/directions, block color sequences, linked pairs, junction connectors, match indicators.

### tr87 — Pattern Rule Application
**Mechanic**: Cycle tile variants to transform a source sequence into a target sequence according to substitution rules.
**Win condition**: Target row matches source row under all rule pair transformations.
**Actions**: 1-2 (cycle variant backward/forward), 3-4 (move cursor)
**Skill tested**: Abstract rule application / analogy. Chained translations and tree-structured lookups in later levels.
**Difficulty**: Single rules → chained translations → bidirectional rules + tree lookups. Energy budget: 128-256.
**Strategy**: Identify rule pairs, match source segments to rule left sides, set target tiles to corresponding right sides.
**Perception needs**: 7 tile variants, rotation states, rule pair boundaries, cursor position, energy bar.

### sb26 — Color Sequence Matching
**Mechanic**: Arrange colored items in frames so that traversal (following portal references) produces a specific checker sequence.
**Win condition**: Checker row fully filled by traversing frames in order.
**Actions**: 5 (scan/test), 6 (click to swap items), 7 (undo)
**Skill tested**: Sequential reasoning with pointer/reference indirection. Working memory.
**Strategy**: Map frame connectivity (portals), determine traversal order, arrange items to produce checker sequence.
**Perception needs**: Frame contents, portal references, checker row colors, energy bar.

### tu93 — Grid Maze with Moving Entities
**Mechanic**: Navigate a maze with autonomous entities (bouncers, followers, arrows) to reach exit tiles.
**Win condition**: Player at exit position.
**Actions**: 1-4 (move in maze)
**Skill tested**: Spatial planning under constraints. Predict entity behavior, plan paths.
**Difficulty**: 9 levels, step budgets 20-60. Entities react to player proximity.
**Strategy**: Find shortest path to exit, account for entity movements. Entities: bouncers reverse at walls, followers trail player, arrows activate on proximity.
**Perception needs**: Maze layout (color 2 = walkable), player/exit/entity positions, entity types, step counter.

### wa30 — Tile Puzzle with Autonomous Agents
**Mechanic**: Link/unlink targets, drag them to docking zones while autonomous boxes/blockers pathfind around you.
**Win condition**: All targets in docking zones and unclaimed by agents.
**Actions**: 1-4 (move hero), 5 (toggle link with adjacent target)
**Skill tested**: Multi-agent coordination and indirect planning. Anticipate BFS pathfinding behavior.
**Strategy**: Link targets to hero, drag to docking zones. Manage autonomous agents — clear/block paths as needed.
**Perception needs**: Hero/target/box/blocker positions, docking zones, target border colors (state indicators), step counter.

### ka59 — Sokoban with Bombs and Enemies
**Mechanic**: Push blocks into goal frames while managing bomb explosions and enemies chasing the goal piece.
**Win condition**: All goal frames contain correct pieces.
**Actions**: 1-4 (push selected piece), 6 (click to select different piece)
**Skill tested**: Spatial reasoning with multi-step consequences. Chain pushing, bomb timing, enemy avoidance.
**Difficulty**: 7 levels, 100-200 steps. Bombs charge and explode, enemies advance each turn.
**Strategy**: Plan push sequences to goal frames. Time around bomb explosions (can be strategic). Protect goal piece from enemies.
**Perception needs**: Piece types/sizes, goal frames, bomb charge state, enemy positions, selected piece, step counter.

### r11l — Indirect Body Positioning
**Mechanic**: Move leg sprites to position body sprites (which auto-center on their legs) over target zones.
**Win condition**: Every body collides with its corresponding target. Composite bodies must have matching colors.
**Actions**: 6 only (click leg to select, click destination to move)
**Skill tested**: Indirect spatial control. Calculate where legs must be for body centroid to hit target.
**Strategy**: For each target, compute required leg positions (centroid math). Move legs avoiding obstacles. Route through paint sprites for color matching.
**Perception needs**: Body-leg associations, target positions, obstacles, paint pickups, strike counter.

### ar25 — Reflection Symmetry Puzzle
**Mechanic**: Position shapes and reflection axes so that reflected copies cover all target positions.
**Win condition**: Union of shapes + reflections covers every target pixel.
**Actions**: 1-4 (move selected), 5 (cycle selection), 6 (click to select), 7 (undo)
**Skill tested**: Symmetry reasoning. Understand reflection across vertical/horizontal axes.
**Difficulty**: Multiple axes, constrained reflection directions, fixed objects, auto-rotation on movement.
**Strategy**: Position shapes and axes so reflections fill target positions. Plan reflection geometry.
**Perception needs**: Shapes, axes, targets, reflected overlay pattern, selection state.

### ls20 — State-Cycling Grid Navigation
**Mechanic**: Move through grid, stepping on triggers that cycle shape/color/rotation state. Reach goals with matching state.
**Win condition**: Player on goal tile with matching shape+color+rotation.
**Actions**: 1-4 (grid movement)
**Skill tested**: Sequential planning and state management. Plan trigger-visit order to configure state correctly.
**Difficulty**: Fog of war, wall pushers, energy pickups, 3-life system. Multiple goals with different requirements.
**Strategy**: Map grid (triggers, goals, energy). Plan path visiting triggers in correct order then reaching goals.
**Perception needs**: Trigger types (shape/color/rotation), goal requirements, current state, fog visibility, energy/lives.

### vc33 — Track Switching Puzzle
**Mechanic**: Click junctions to reroute track segments, click switch levers to swap cargo between tracks.
**Win condition**: All cargo colors match their destination markers.
**Actions**: 6 only (click junctions and levers)
**Skill tested**: Logical routing and network reasoning. Plan rerouting sequence.
**Strategy**: Map track network connectivity, click junctions to route each cargo to matching-color destination.
**Perception needs**: Tracks, junctions, cargo items (colors), destinations (colors), switch levers, track orientation.

### re86 — Shape Deformation and Color Matching
**Mechanic**: Navigate shaped pieces through barriers (which reshape them) and color zones (which recolor them) to recreate a target pattern.
**Win condition**: Composite of all pieces matches target pattern.
**Actions**: 1-4 (move active piece), 5 (cycle to next piece)
**Skill tested**: Spatial manipulation with deformation reasoning. Plan paths through barriers and color zones.
**Strategy**: Examine target, determine required shape/color per piece, route pieces through appropriate barriers and zones.
**Perception needs**: Piece shapes/colors, barriers, color zones, target pattern, active piece indicator.

### su15 — Suika Merge Game
**Mechanic**: Click to create vacuum pulls that group same-tier fruits for merging. Enemies eat fruits.
**Win condition**: Required fruit tier counts in goal zones.
**Actions**: 6 (click to vacuum pull — needs x,y), 7 (undo)
**Skill tested**: Physics prediction + merge chain reasoning. Plan vacuum pulls to group and merge.
**Strategy**: Compute merge requirements (what tier fruits needed), click to pull same-tier fruits together, avoid enemies, position results in goal zones.
**Perception needs**: Fruit tiers (by size/color), enemy positions, goal zones, vacuum physics.

### s5i5 — Arm/Beam Kinematic Chain
**Mechanic**: Extend/retract arms and rotate them via colored buttons to position carried objects onto targets.
**Win condition**: All targets covered by carried objects.
**Actions**: 6 only (click tracks to extend/retract, click buttons to rotate)
**Skill tested**: Hierarchical kinematic reasoning. Parent-child arm relationships.
**Strategy**: Trace kinematic chains from targets to controlling arms. Plan extend/retract/rotate sequences.
**Perception needs**: Arms (with directional indicators), parent-child relationships, rotation buttons, tracks, targets.

### ft09 — Constraint Satisfaction Coloring
**Mechanic**: Click tiles to cycle colors. Each click propagates to neighbors per a pattern matrix. Constraints require specific same/different relationships.
**Win condition**: All constraint sprites satisfied simultaneously.
**Actions**: 6 only (click tiles)
**Skill tested**: SAT-like constraint reasoning. Understand propagation effects.
**Strategy**: Read constraints (center = required color, 0 = must match, non-0 = must differ), build constraint graph, solve by clicking tiles accounting for propagation.
**Perception needs**: Constraint sprites (3×3 patterns), tile colors, propagation patterns for NTi tiles.

### tn36 — Block Programming Puzzle
**Mechanic**: Toggle bits on a programmable tablet to create instruction sequences. Execute to move/rotate/scale/recolor a block to match a target.
**Win condition**: Block position, rotation, scale, and color match target outline.
**Actions**: 6 only (click tablet bits, tab buttons, play/check buttons)
**Skill tested**: Sequential program construction. Predict spatial transformations from instruction sequences.
**Difficulty**: 7 levels. Adds rotation, scaling, color changes, toggle barriers, multiple program slots.
**Strategy**: Determine required transformation sequence, encode as bitmask instructions. Handle toggle barriers (appear/disappear every 3 instructions).
**Perception needs**: Block/target position+rotation+scale+color, tablet bit states, wall layout, toggle barriers.

### sc25 — Spell-Casting Dungeon Crawler
**Mechanic**: Navigate dungeon, cast spells by toggling a 3×3 spell grid (teleport, resize, fireball). Open doors, use teleporters.
**Win condition**: Reach exit sprite.
**Actions**: 1-4 (move), 6 (click spell slots and UI)
**Skill tested**: Multi-step puzzle solving with size-dependent pathfinding. Which spells, when, in what order.
**Difficulty**: 6 levels. Combines all 3 spells, limited action budget, complex dungeon layouts.
**Strategy**: Plan route to exit, identify door/teleporter/resize requirements. Cast spells in right order. Collect energy packs.
**Perception needs**: Player, exit, doors, switches, teleport pads, energy packs, spell grid state, allowed spells per level.

### g50t — Pac-Man Maze with Logic Gates
**Mechanic**: Navigate maze to reach goal while avoiding ghosts. Multi-section levels with gate/button puzzles.
**Win condition**: Player reaches goal position.
**Actions**: 1-4 (move), 5 (undo/rewind path)
**Skill tested**: Spatial planning under adversarial time pressure. Predict ghost movement patterns.
**Strategy**: Plan shortest path avoiding ghosts (prefer current heading, then left/right, then reverse). Use undo to retreat.
**Perception needs**: Player/goal/ghost positions, ghost facing directions, maze walls, gate states, timer bar.

### dc22 — Grid Puzzle with Crane/Bridges
**Mechanic**: Move player piece to goal. Use crane/cursor to carry bridge pieces, step on pressure plates, click color-cycling buttons.
**Win condition**: Player piece at goal position.
**Actions**: 1-4 (move piece), 6 (click buttons/crane)
**Skill tested**: Multi-system coordination. Understand how bridges, pressure plates, and color-cycling interact.
**Strategy**: Locate goal, plan path. Use crane to bridge gaps, pressure plates to reveal paths, color buttons to change barriers.
**Perception needs**: Player/goal positions, crane/bridge positions, pressure plates, barriers, color-cycle buttons, step counter.

### bp35 — Gravity Platformer
**Mechanic**: Move left/right on platforms. Click breakable blocks to create fall paths. Gravity pulls player down (or up in inverted levels).
**Win condition**: Land on gem tile.
**Actions**: 3-4 (move left/right), 6 (click to break blocks or grow vines), 7 (undo)
**Skill tested**: Physics/causal reasoning. Predict fall trajectories after removing blocks. Chain reactions.
**Difficulty**: 10 levels. Advancing hazards (levels 1-3), inverted gravity, vine growth blocks.
**Strategy**: Find gem position, plan which blocks to break to create fall path. Avoid spikes.
**Perception needs**: Player/gem/spike positions, breakable vs solid blocks, gravity direction, vine blocks, advancing hazard.

### lf52 — Sliding Merge Puzzle
**Mechanic**: Select pieces and launch them in cardinal directions until they hit walls/pieces. Identical pieces sandwich to merge/eliminate middle piece.
**Win condition**: Reduce to 1 piece (or 2 in later levels), then click confirm button.
**Actions**: 1-4 (slide all pieces), 6 (click to select/launch/confirm), 7 (undo)
**Skill tested**: Combinatorial puzzle solving. Plan merge sequences considering slide mechanics.
**Difficulty**: 10 levels. More pieces, larger grids, camera scrolling, special piece types.
**Strategy**: Plan launch sequences that create sandwich merges. Use bulk moves (ACTION1-4) for repositioning.
**Perception needs**: Piece positions/colors, grid boundaries, directional arrows, confirm button, camera offset.

### lp85 — Rotation Ring Puzzle
**Mechanic**: Click left/right buttons to rotate tiles along ring tracks. Tiles must end at goal positions.
**Win condition**: Every tile piece at a goal marker, every secondary piece at secondary goal marker.
**Actions**: 6 only (click rotation buttons — needs x,y)
**Skill tested**: Permutation group reasoning. Cyclic planning with interlocking ring constraints.
**Difficulty**: 8 levels. 1 ring → 3 interlocking → 4 rotation groups with mirrored quadrants. Step budgets 13-150.
**Strategy**: Identify tile and goal positions per ring. Compute rotation count: (goal_slot - current_slot) mod ring_size. Solve simultaneous constraints when rings share tiles.
**Perception needs**: Tile positions (color 9), goal positions (color 11 dots), button positions (color 8 left, color 14 right), ring geometry, step counter.

---

## What This Means for SAGE

**The competition games will NOT be these 25 games.** These are practice environments. The actual competition uses novel games we've never seen. Game-specific solvers are useless — they'd score 100% on practice and 0% on competition. This analysis exists to identify what **general capabilities** SAGE needs, not to build per-game solutions.

*"Build the driver, not a track-specific solution. The consciousness loop IS the driver. ARC-AGI-3 IS the track. If we build for the track, we get 97% on one game and 0% on others. If we build the driver, the driver handles any track."*

### What the 25 Games Collectively Demand

From analyzing the source code, every game requires some combination of these **general cognitive capabilities**:

**1. Object Detection & Tracking**
Every game has discrete objects (sprites, tiles, pieces, tokens) on a grid. SAGE must perceive them — not as pixel changes, but as entities with position, color, shape, and relationships to other entities. Andy's connected component analysis is the foundation. But SAGE also needs to track objects across frames (this piece moved from here to there) and recognize object types by visual features (this is a button, this is a wall, this is a goal marker).

*Tested by: all 25 games. Non-negotiable.*

**2. Action-Outcome Learning**
Every game has actions that produce observable effects. SAGE must learn the mapping: "when I do X in state S, outcome Y happens." This is exactly what McNugget's game runner demonstrated — forming hypotheses like "UP shifts grid upward." The consciousness loop already has SNARC salience scoring and membot cartridges for this. The gap is speed — learning must happen within the first 10-20 actions of a new game, not after 200.

*Tested by: all 25 games. The two-phase approach (fast exploration → hypothesis formation) is the right pattern, but it must be general, not game-specific.*

**3. Goal Detection**
The hardest gap. SAGE explores well but doesn't know what it's trying to achieve. It needs to detect goal signals:
- **Level completion** — the strongest signal. When `levels_completed` increments, SAGE must record exactly what preceded it and prioritize repeating/extending that pattern.
- **Visual goal markers** — many games display goal positions (color 11 in lp85, exit sprites in tu93, docking zones in wa30). SAGE needs to recognize "this looks like a target."
- **Progress indicators** — step counters, energy bars, score displays. Frame-change patterns that correlate with progress.
- **State convergence** — detecting when the game state is approaching a "solved" configuration (pieces aligning, constraints being satisfied).

*Most critical missing capability. Without goal detection, exploration is aimless.*

**4. Spatial Reasoning**
20 of 25 games require understanding spatial relationships: adjacency, containment, alignment, symmetry, rotation, path connectivity. SAGE needs to go beyond "48 cells changed" to "object A moved 3 cells right and is now adjacent to object B."

This maps to Andy's perception layer — structured GridObservation with objects, bounding boxes, movement tracking. The consciousness loop consumes this as spatial facts, not raw pixel data.

*Tested by: cn04 (alignment), lp85 (rotation), sp80 (path flow), tu93 (maze navigation), ar25 (reflection), m0r0 (mirrored movement), and many others.*

**5. Hypothesis Formation & Testing**
McNugget's game runner already demonstrates this: the LLM forms theories about game rules and designs experiments to test them. This is the core cognitive loop:
- Observe → hypothesize → predict → act → compare prediction to outcome → revise

The consciousness loop's 12-step cycle maps directly: Sense (observe) → Salience (what's surprising) → Select (what to attend to) → Execute (act) → Learn (compare to prediction) → Remember (update hypothesis).

*The LLM is the reasoning engine. SAGE provides the loop structure that makes reasoning systematic rather than ad-hoc.*

**6. Coordinate Interaction**
Many games require clicking specific grid positions (ACTION6 with x,y). This means SAGE must:
- Identify clickable targets from the grid
- Choose which target to click based on current hypothesis
- Map grid coordinates to SDK action parameters

This is fundamentally different from simple directional movement (ACTION1-4). The agent needs spatial targeting, not just direction selection.

*Tested by: cn04, cd82, ft09, tn36, sc25, vc33, s5i5, lp85, su15, r11l — at least 10 of 25 games.*

**7. Multi-Step Planning**
Several games require action sequences where individual actions have no visible effect but the sequence as a whole achieves a goal. SAGE must plan beyond single actions:
- Rotation sequences (lp85: rotate ring A left 3 times, then ring B right 2 times)
- Navigation paths (tu93: up, up, right, right, down to reach exit)
- Causal chains (bp35: break block → fall → land on platform → break next block → reach gem)

The consciousness loop's ATP budgeting already supports this — allocate resources for multi-step plans, not just reactive single actions.

*Tested by: lp85, sp80, ka59, bp35, tn36, sc25, m0r0, lf52 — at least 8 of 25 games.*

**8. Transfer Across Levels**
Within a single game environment, levels get harder but follow the same mechanics. What SAGE learns in Level 1 should accelerate Level 2. This is exactly what membot cartridges provide — action-outcome pairs searchable by state similarity.

Across game environments (different games), transfer is harder — but the cognitive capabilities are the same. An agent that learned "clicking colored cells sometimes triggers level completion" in one game should try that strategy early in a new game.

*This is the competition differentiator. StochasticGoose scored 12.58% with no transfer. Memory IS the scoring advantage.*

### What SAGE Already Has

| Capability | SAGE Component | Status |
|-----------|---------------|--------|
| Object detection | GridVisionIRP + Andy's perception | Built, needs wiring |
| Action-outcome learning | GameMemory + membot cartridges | Working (McNugget demo) |
| Goal detection | **GAP** | Level-up signal exists, visual goal detection missing |
| Spatial reasoning | GridObservation structured output | Dataclass defined, LLM consumes it |
| Hypothesis formation | LLM reasoning in consciousness loop | Working (McNugget demo) |
| Coordinate interaction | GameActionEffector with x,y | Built, tested |
| Multi-step planning | Sequence planning in game runner v2 | Working (3-5 action sequences) |
| Cross-level transfer | Membot cartridges + SNARC | Architecture exists, not yet wired end-to-end |

### What SAGE Still Needs

1. **Goal detection module** — Watches for level_completed changes, identifies visual goal markers, tracks progress indicators. This is the #1 gap. Without it, SAGE explores forever.

2. **Fast exploration phase** — First 20-50 actions should be rapid systematic probing (try each action, try clicking each color, try clicking each region). No LLM needed. Pure observation. Build the action-outcome map fast.

3. **Exploration → exploitation transition** — Once SAGE has a hypothesis with supporting evidence, shift from exploring to executing. Trust posture already models this (confidence → strategy), but it needs to be wired to the game loop.

4. **State hashing / cycle detection** — Detect when the game is in a previously-seen state. Break loops. Don't waste actions repeating what didn't work.

5. **Reward signal amplification** — When a level completes, that signal should dominate everything. Record the last N actions, mark them as the most valuable experience in the cartridge, and prioritize similar patterns.

### RHAE Implications

RHAE = min(1, human_actions / agent_actions)². The squared penalty means:

- **Efficiency over completeness** — solving 3/8 levels in 50 actions beats solving 4/8 in 500 actions
- **Fast exploration matters** — every wasted exploration action reduces RHAE
- **The right hypothesis early is worth more than exhaustive search**
- **Memory saves actions** — recalling "this worked before" avoids re-exploring

### Fleet Role (Development, Not Competition)

The fleet's role is to **develop and validate SAGE's general capabilities**, not to build game-specific solvers:

| Machine | Development Role |
|---------|-----------------|
| **Sprout** (8GB) | Stress-test perception + fast exploration on edge hardware. If SAGE works here, it works on the competition's RTX 5090. |
| **McNugget** (16GB) | Primary integration testing. Game runner development. Hypothesis formation quality. |
| **Thor** (122GB) | Deep reasoning experiments. Cross-level transfer. Memory architecture validation. |
| **CBP** (16GB) | Coordination. Strategy. This document. |
| **Andy's team** | Perception layer (GridCartridge, embeddings, visual memory). The eyes. |
