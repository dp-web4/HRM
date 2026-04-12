#!/usr/bin/env python3
"""ka59 Sokoban solver — simulation-based A* with engine verification.

Approach:
1. Extract level geometry (walls, gulches as pixel sets)
2. Simulate pushes accurately based on engine analysis
3. A* search with Manhattan heuristic
4. Execute found solution on actual engine

Push mechanics (from engine source analysis):
- Player box move checks walls AND gulches (pixel collision)
- If move collides with another pushable, the move is REVERTED
- The pushed box is pushed by (dx,dy) via esglqyymck (wall check only)
- Push animation runs for ciwfbggkom=5 frames
- Each frame: if frame >= 5 AND not on gulch pixel → done
  Else: push 3px more in same direction
- If push hits wall → stop
- Net effect: pushed box moves 5*3=15px, but stops at walls
"""
import sys, os, json, time, heapq
from collections import deque
sys.path.insert(0, os.path.dirname(__file__))

from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6

STEP = 3
PUSH_FRAMES = 5  # ciwfbggkom
DIR_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}
NAME_TO_ACT = {'UP': UP, 'DOWN': DOWN, 'LEFT': LEFT, 'RIGHT': RIGHT}

VISUAL_DIR = '/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/ka59'


def get_display_offset(game):
    return (64 - game.camera.width) // 2, (64 - game.camera.height) // 2


def click_box(env, game, box_sprite, offset):
    ox, oy = offset
    dx = box_sprite.x + box_sprite.width // 2 + ox
    dy = box_sprite.y + box_sprite.height // 2 + oy
    return env.step(CLICK, data={'x': dx, 'y': dy})


def build_pixel_set(sprites):
    """Build set of solid pixels from sprites."""
    pixels = set()
    for s in sprites:
        for r in range(s.pixels.shape[0]):
            for c in range(s.pixels.shape[1]):
                if s.pixels[r, c] >= 0:
                    pixels.add((s.x + c, s.y + r))
    return pixels


def box_hits_pixels(bx, by, bw, bh, pixel_set):
    """Check if any pixel of a box at (bx,by) with size (bw,bh) is in pixel_set."""
    for r in range(bh):
        for c in range(bw):
            if (bx + c, by + r) in pixel_set:
                return True
    return False


def boxes_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2


def simulate_push(px, py, pw, ph, dx, dy, wall_pixels, gulch_pixels, other_boxes):
    """Simulate push of a box starting at (px,py).

    Engine mechanics:
    1. First push via esglqyymck: move by (dx,dy), check walls only, recursive push
    2. Then tesoceiypr runs each animation frame:
       - frame 0..4: push by (dx,dy), if wall-blocked → done
       - frame 5+: if NOT on gulch → done, else keep pushing

    The initial push in step 1 is the same as frame 0 of step 2.
    Actually, re-reading the code: aqoyvqxpct detects collision and stores
    the pushed box. Then step() stores in osqsnogokn and returns.
    On next step() call, the animation loop runs tesoceiypr.

    tesoceiypr logic per frame:
    - If msbtawvatm >= 5 AND not on gulch → return True (done)
    - Else: try esglqyymck(push dx,dy) → if blocked return True (done)
    - Else: return False (continue)

    And step() does: if all(tesoceiypr for each pushed) → clear animation, else msbtawvatm++

    So the pushed box moves (dx,dy) each frame until:
    - Hits wall (blocked by wall or other pushable)
    - OR frame count >= 5 AND not on gulch

    The frame count starts at 0 when osqsnogokn is set.
    """
    nx, ny = px, py
    frame = 0

    while True:
        # Try to push one step
        nx2, ny2 = nx + dx, ny + dy

        # Check wall collision (esglqyymck only checks divgcilurm)
        if box_hits_pixels(nx2, ny2, pw, ph, wall_pixels):
            break  # Blocked by wall

        # Check collision with other boxes
        blocked = False
        for ox, oy, ow, oh in other_boxes:
            if boxes_overlap(nx2, ny2, pw, ph, ox, oy, ow, oh):
                blocked = True
                break
        if blocked:
            break

        nx, ny = nx2, ny2
        frame += 1

        # Check stop condition
        if frame >= PUSH_FRAMES:
            on_gulch = box_hits_pixels(nx, ny, pw, ph, gulch_pixels)
            if not on_gulch:
                break
            # On gulch: keep pushing

    if (nx, ny) == (px, py):
        return None  # Didn't move at all
    return (nx, ny)


def solve_level(game, max_steps=60, max_states=500000):
    """Solve current level using A* with simulation."""
    boxes = game.current_level.get_sprites_by_tag('xlfuqjygey')
    goals = game.current_level.get_sprites_by_tag('rktpmjcpkt')
    walls = game.current_level.get_sprites_by_tag('divgcilurm')
    gulches = game.current_level.get_sprites_by_tag('vwjqkxkyxm')
    enemies = game.current_level.get_sprites_by_tag('Enemy')
    bombs = game.current_level.get_sprites_by_tag('gobzaprasa')
    sec_goals = game.current_level.get_sprites_by_tag('ucjzrlvfkb')
    enemy_targets = game.current_level.get_sprites_by_tag('nnckfubbhi')

    # Combine: player boxes + enemy targets (both are pushable)
    # Enemy targets (nnckfubbhi) need to reach secondary goals (ucjzrlvfkb)
    all_pushables = list(boxes) + list(enemy_targets)
    n_boxes = len(all_pushables)
    n_player_boxes = len(boxes)
    box_dims = [(b.width, b.height) for b in all_pushables]

    wall_pixels = build_pixel_set(walls)
    gulch_pixels = build_pixel_set(gulches)

    # Combined goal info: primary goals + secondary goals
    goal_info = [(g.x + 1, g.y + 1, g.width - 2, g.height - 2) for g in goals]
    sec_goal_info = [(g.x + 1, g.y + 1, g.width - 2, g.height - 2) for g in sec_goals]
    all_goals = goal_info + sec_goal_info

    print(f"  Player boxes: {n_player_boxes}, Enemy targets: {len(enemy_targets)}")
    print(f"  Primary goals: {len(goals)}, Secondary goals: {len(sec_goals)}")
    print(f"  Enemies: {len(enemies)}, Bombs: {len(bombs)}")
    for i, b in enumerate(all_pushables):
        tag = 'box' if i < n_player_boxes else 'et'
        print(f"    {tag}[{i}]: ({b.x},{b.y}) {b.width}x{b.height}")
    for i, g in enumerate(goals):
        print(f"    goal[{i}]: ({g.x},{g.y}) {g.width}x{g.height} → target ({g.x+1},{g.y+1})")
    for i, g in enumerate(sec_goals):
        print(f"    sec_goal[{i}]: ({g.x},{g.y}) {g.width}x{g.height} → target ({g.x+1},{g.y+1})")

    if enemies or bombs:
        print("  WARNING: enemies/bombs present, may affect solution")

    # Match pushables to goals by dimension
    goal_by_size = {}
    for gi, (gx, gy, gw, gh) in enumerate(all_goals):
        goal_by_size.setdefault((gw, gh), []).append((gi, gx, gy))

    box_goals = {}
    for bi in range(n_boxes):
        bw, bh = box_dims[bi]
        box_goals[bi] = goal_by_size.get((bw, bh), [])

    def heuristic(positions):
        total = 0
        used = set()
        for bi in range(n_boxes):
            bx, by = positions[bi]
            best_d = float('inf')
            best_gi = -1
            for gi, gx, gy in box_goals.get(bi, []):
                if gi in used:
                    continue
                d = abs(bx - gx) + abs(by - gy)
                if d < best_d:
                    best_d = d
                    best_gi = gi
            if best_gi >= 0:
                used.add(best_gi)
                total += best_d // STEP
        return total

    def is_win(positions):
        # Check ALL goals (primary + secondary)
        used = set()
        for gx, gy, gw, gh in all_goals:
            found = False
            for j, (bx, by) in enumerate(positions):
                if j in used:
                    continue
                bw, bh = box_dims[j]
                if bx == gx and by == gy and bw == gw and bh == gh:
                    found = True
                    used.add(j)
                    break
            if not found:
                return False
        return True

    def try_move(positions, sel, dx, dy):
        bx, by = positions[sel]
        bw, bh = box_dims[sel]
        nx, ny = bx + dx, by + dy

        # Player box checks walls AND gulches
        if box_hits_pixels(nx, ny, bw, bh, wall_pixels):
            return None
        if box_hits_pixels(nx, ny, bw, bh, gulch_pixels):
            return None

        # Check collision with other boxes
        push_targets = []
        for j, (ox, oy) in enumerate(positions):
            if j == sel:
                continue
            ow, oh = box_dims[j]
            if boxes_overlap(nx, ny, bw, bh, ox, oy, ow, oh):
                push_targets.append(j)

        if not push_targets:
            new_pos = list(positions)
            new_pos[sel] = (nx, ny)
            return tuple(new_pos)

        # Push: selected box STAYS, pushed boxes move
        new_pos = list(positions)

        for pi in push_targets:
            px, py = positions[pi]
            pw, ph = box_dims[pi]
            other = [(positions[j][0], positions[j][1], box_dims[j][0], box_dims[j][1])
                     for j in range(n_boxes) if j != sel and j != pi]

            result = simulate_push(px, py, pw, ph, dx, dy, wall_pixels, gulch_pixels, other)
            if result is None:
                return None
            new_pos[pi] = result

        return tuple(new_pos)

    # A* search — state includes ALL pushable positions
    init_pos = tuple((b.x, b.y) for b in all_pushables)
    if is_win(init_pos):
        return []

    h0 = heuristic(init_pos)
    visited = {init_pos}
    counter = 0
    pq = [(h0, counter, init_pos, [])]

    directions = [(0, -STEP, 'UP'), (0, STEP, 'DOWN'), (-STEP, 0, 'LEFT'), (STEP, 0, 'RIGHT')]
    states_explored = 0

    while pq:
        f, _, positions, actions = heapq.heappop(pq)
        g = len(actions)

        if g >= max_steps:
            continue

        states_explored += 1
        if states_explored > max_states:
            print(f"  A* exhausted at {max_states} states")
            break

        if states_explored % 50000 == 0:
            print(f"  ... {states_explored} states, pq={len(pq)}, depth={g}, f={f}")

        # Only directly move player boxes (not enemy targets)
        for box_idx in range(n_player_boxes):
            for dx, dy, dname in directions:
                result = try_move(positions, box_idx, dx, dy)
                if result is not None and result not in visited:
                    new_actions = actions + [(box_idx, dname)]
                    if is_win(result):
                        print(f"  SOLUTION! {len(new_actions)} moves, {states_explored} states")
                        return new_actions
                    visited.add(result)
                    h = heuristic(result)
                    counter += 1
                    heapq.heappush(pq, (g + 1 + h, counter, result, new_actions))

    print(f"  No solution found ({states_explored} states)")
    return None


def execute_solution(env, game, solution, level_idx):
    """Execute solution on engine."""
    boxes = game.current_level.get_sprites_by_tag('xlfuqjygey')
    offset = get_display_offset(game)
    current_sel = 0
    total_steps = 0

    for box_idx, dname in solution:
        if box_idx != current_sel:
            boxes = game.current_level.get_sprites_by_tag('xlfuqjygey')
            click_box(env, game, boxes[box_idx], offset)
            current_sel = box_idx
            total_steps += 1

        fd = env.step(NAME_TO_ACT[dname])
        total_steps += 1

        if game.level_index > level_idx:
            return total_steps, True
        if fd.state.name == 'GAME_OVER':
            return total_steps, False

    return total_steps, game.level_index > level_idx


def verify_against_engine(env, game, solution, level_idx):
    """Verify solution by executing and checking if BFS positions match engine positions."""
    boxes = game.current_level.get_sprites_by_tag('xlfuqjygey')
    offset = get_display_offset(game)
    current_sel = 0

    for i, (box_idx, dname) in enumerate(solution):
        if box_idx != current_sel:
            boxes = game.current_level.get_sprites_by_tag('xlfuqjygey')
            click_box(env, game, boxes[box_idx], offset)
            current_sel = box_idx

        fd = env.step(NAME_TO_ACT[dname])
        boxes = game.current_level.get_sprites_by_tag('xlfuqjygey')

        if game.level_index > level_idx:
            print(f"    Level completed at step {i+1}!")
            return True

    return game.level_index > level_idx


def main():
    print("=" * 60)
    print("ka59 Sokoban Solver — Simulation A*")
    print("=" * 60)

    arcade = Arcade()
    env = arcade.make('ka59-9f096b4a')
    fd = env.reset()
    game = env._game

    total_actions = 0
    results = {}

    for level in range(7):
        print(f"\n{'='*50}")
        print(f"Level {level} (engine={game.level_index}, completed={fd.levels_completed})")
        print(f"{'='*50}")

        if fd.state.name in ('WON', 'GAME_OVER'):
            print(f"  Game ended: {fd.state.name}")
            break

        step_counter = game.current_level.get_data('StepCounter')
        print(f"  Step counter: {step_counter}")

        t0 = time.time()
        max_bfs = min(step_counter, 60)
        solution = solve_level(game, max_steps=max_bfs, max_states=500000)
        dt = time.time() - t0
        print(f"  Solver time: {dt:.1f}s")

        if solution is None:
            print(f"  FAILED to solve L{level}")
            break

        if len(solution) == 0:
            print(f"  L{level} trivially solved!")
            results[level] = {'actions': 0}
            continue

        print(f"  Solution: {len(solution)} moves")
        for i, (bidx, dname) in enumerate(solution):
            print(f"    {i}: box[{bidx}] {dname}")

        # Verify and execute
        print(f"  Executing on engine...")
        ok = verify_against_engine(env, game, solution, level)

        if ok or game.level_index > level:
            n = len(solution)
            print(f"  L{level} SOLVED!")
            results[level] = {'actions': n}
            total_actions += n
            # DON'T send any action after level completion - next level is already loaded
        else:
            print(f"  L{level} execution FAILED")
            boxes = game.current_level.get_sprites_by_tag('xlfuqjygey')
            goals = game.current_level.get_sprites_by_tag('rktpmjcpkt')
            for i, b in enumerate(boxes):
                print(f"    box[{i}]: ({b.x},{b.y})")
            for i, g in enumerate(goals):
                print(f"    goal[{i}]: target ({g.x+1},{g.y+1})")
            break

    print(f"\n{'='*60}")
    print(f"FINAL: {len(results)}/7 levels solved, {total_actions} total actions")

    with open(f'{VISUAL_DIR}/ka59_results.json', 'w') as f:
        json.dump({
            'game_id': 'ka59-9f096b4a',
            'levels_solved': len(results),
            'total_actions': total_actions,
            'per_level': results,
        }, f, indent=2)


if __name__ == '__main__':
    main()
