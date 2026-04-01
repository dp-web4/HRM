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

## Strategic Priorities for Competition

### Tier 1 — Most solvable with current architecture (click/explore + simple reasoning)
These games have clear click-based mechanics where systematic exploration can score:
- **lp85** (rotation — already scoring 4/8)
- **vc33** (track routing — already scoring 1/7)
- **ft09** (constraint coloring — click to solve)
- **cn04** (jigsaw — spatial alignment)
- **sb26** (sequence matching — already scoring 1/8)

### Tier 2 — Solvable with code analysis + lightweight planning
These need game-specific solvers informed by source code analysis:
- **bp35** (gravity platformer — compute fall trajectories)
- **tu93** (maze navigation — BFS to exit)
- **g50t** (pac-man — ghost avoidance + pathfinding)
- **lf52** (merge puzzle — brute-force small grids)
- **tr87** (rule application — direct rule lookup)

### Tier 3 — Requires significant spatial reasoning
These need genuine spatial intelligence or physics simulation:
- **sp80** (pipe flow — simulate fluid)
- **m0r0** (mirrored tokens — simultaneous constraint solving)
- **ka59** (sokoban + bombs — multi-step planning)
- **su15** (merge physics — predict vacuum dynamics)
- **re86** (shape deformation — predict barrier effects)

### Tier 4 — Hardest (complex multi-system reasoning)
- **dc22** (crane + bridges + pressure plates + color cycling)
- **sc25** (spell-casting dungeon — multi-spell sequencing)
- **tn36** (program construction — encode transformation sequences)
- **ar25** (reflection — symmetry + axis planning)
- **s5i5** (kinematic chains — hierarchical arm control)
- **ls20** (state cycling under fog of war)
- **wa30** (autonomous agent coordination)
- **sk48** (conveyor sequence alignment)
- **cd82** (octagonal sector painting)
- **r11l** (indirect centroid positioning)

### Action Budget Strategy

RHAE scoring is squared — taking 2× human baseline gives 25% score, not 50%. Therefore:

1. **Solve fewer games more efficiently** rather than attempting all 25
2. **Game-specific solvers** (informed by source code) will always beat general exploration
3. **Brute-force exploration** (5000+ steps/sec without LLM) finds goal conditions fast
4. **LLM reasoning** (1-8 sec/step) should only be used for strategy, not exploration
5. **Two-phase per game**: random exploration to discover mechanics, then targeted solver

### Fleet Division

| Machine | Role | Model | Games |
|---------|------|-------|-------|
| **Sprout** (8GB) | Brute-force exploration | None (pure logic) | All 25 — find which games respond to clicks |
| **McNugget** (16GB) | Hypothesis + solver dev | Gemma 3 12B | Tier 1-2 games — build solvers |
| **Thor** (122GB) | Deep reasoning + code analysis | Qwen 27B | Tier 3-4 games — plan strategies |
| **CBP** (16GB) | Coordinator + testing | Qwen 3.5 0.8B | Validate solvers across games |
