#!/usr/bin/env python3
"""ka59 Sokoban solver — pixel-perfect simulation with A* search.

Complete simulation including:
- Pixel-perfect collision (matching engine's collides_with)
- Push animation (pushed box slides until frame>=5 or wall-blocked or off-gulch)
- Recursive push chains (esglqyymck)
- Bomb consumption and explosion mechanics
- Enemy target (nnckfubbhi) pushing for secondary goals

Key engine mechanics:
- All collision is PIXEL_PERFECT (default BlockingMode)
- Push animation resolves entirely within one env.step() call
- Player box stays in place on collision (position reverted), pushed box slides
- Bombs consume one row per player action, explode when fully consumed
- Explosion pushes nearby objects in direction based on bomb rotation
"""
import sys, os, json, time, heapq, copy
from collections import deque
sys.path.insert(0, os.path.dirname(__file__))

from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from PIL import Image

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6

STEP = 3
PUSH_FRAMES = 5
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


def save_frame(env, game, filename):
    """Save current frame as PNG by rendering the game camera."""
    try:
        os.makedirs(VISUAL_DIR, exist_ok=True)
        frame = game.camera.render(game.current_level.get_sprites())
        if frame is not None:
            frame = np.array(frame)
            img_data = np.zeros((*frame.shape, 3), dtype=np.uint8)
            palette = {
                0: (0, 0, 0),       # black
                1: (0, 0, 200),     # blue (background)
                2: (200, 0, 0),     # red (padding)
                4: (200, 200, 0),   # yellow
                5: (0, 200, 0),     # green
                9: (200, 100, 0),   # orange
                11: (150, 0, 150),  # purple
                12: (100, 100, 100),# gray
                13: (50, 50, 50),   # dark gray
                14: (255, 255, 255),# white
                15: (0, 150, 150),  # teal
            }
            for color_id, rgb in palette.items():
                mask = frame == color_id
                for c in range(3):
                    img_data[:,:,c][mask] = rgb[c]
            img = Image.fromarray(img_data)
            img = img.resize((256, 256), Image.NEAREST)
            img.save(os.path.join(VISUAL_DIR, filename))
    except Exception as e:
        print(f"  Warning: could not save frame: {e}")


# ============================================================
# Pixel-perfect collision simulation
# ============================================================

def pixel_collides(s1_x, s1_y, s1_pixels, s2_x, s2_y, s2_pixels):
    """Pixel-perfect collision between two sprites."""
    s1_h, s1_w = s1_pixels.shape
    s2_h, s2_w = s2_pixels.shape

    if (s1_x >= s2_x + s2_w or s1_x + s1_w <= s2_x or
        s1_y >= s2_y + s2_h or s1_y + s1_h <= s2_y):
        return False

    x_min = max(s1_x, s2_x)
    x_max = min(s1_x + s1_w, s2_x + s2_w)
    y_min = max(s1_y, s2_y)
    y_max = min(s1_y + s1_h, s2_y + s2_h)

    s1_region = s1_pixels[y_min - s1_y:y_max - s1_y, x_min - s1_x:x_max - s1_x]
    s2_region = s2_pixels[y_min - s2_y:y_max - s2_y, x_min - s2_x:x_max - s2_x]

    return bool(np.any((s1_region != -1) & (s2_region != -1)))


def rotate_pixels(pixels, rotation):
    """Rotate pixel array by 0/90/180/270 degrees (matching engine's render)."""
    if rotation == 0:
        return pixels
    elif rotation == 90:
        return np.rot90(pixels, k=3)  # 90 CW = 3x CCW
    elif rotation == 180:
        return np.rot90(pixels, k=2)
    elif rotation == 270:
        return np.rot90(pixels, k=1)  # 270 CW = 1x CCW
    return pixels


class LevelSim:
    """Simulation of a ka59 level with pixel-perfect collision."""

    def __init__(self, game):
        level = game.current_level

        # Extract wall and gulch sprites (using render() for proper rotation)
        self.walls = []
        for s in level.get_sprites_by_tag('divgcilurm'):
            self.walls.append((s.x, s.y, s.render().copy()))

        self.gulches = []
        for s in level.get_sprites_by_tag('vwjqkxkyxm'):
            self.gulches.append((s.x, s.y, s.render().copy()))

        # Extract pushable sprites: boxes, enemy targets, bombs
        self.player_sprites = level.get_sprites_by_tag('xlfuqjygey')
        self.et_sprites = level.get_sprites_by_tag('nnckfubbhi')
        self.bomb_sprites = level.get_sprites_by_tag('gobzaprasa')

        # All pushables in order: player_boxes, enemy_targets, bombs
        self.n_player = len(self.player_sprites)
        self.n_et = len(self.et_sprites)
        self.n_bombs = len(self.bomb_sprites)
        self.n_push = self.n_player + self.n_et + self.n_bombs

        all_sprites = list(self.player_sprites) + list(self.et_sprites) + list(self.bomb_sprites)

        # Store rendered pixel arrays (with rotation applied) for pushables
        self.push_pixels = [s.render().copy() for s in all_sprites]
        self.push_dims = [(p.shape[1], p.shape[0]) for p in self.push_pixels]

        # Bomb info: rotation, initial remaining rows, initial positions
        self.bomb_rotations = [s.rotation for s in self.bomb_sprites]
        self.bomb_raw_pixels = [s.pixels.copy() for s in self.bomb_sprites]

        # Count remaining unconsumed rows for each bomb
        self.bomb_init_rows = []
        for bp in self.bomb_raw_pixels:
            remaining = 0
            for r in range(bp.shape[0]):
                if bp[r, 0] != 12:  # znqlwwhver = 12 (consumed)
                    remaining += 1
            self.bomb_init_rows.append(remaining)

        # Bomb dimensions (for explosion sprite lookup)
        self.bomb_widths = [s.width for s in self.bomb_sprites]

        # Precompute explosion sprite pixels (rendered with rotation)
        # Engine uses: sprites[f"explode-{bomb.width}-{frame_index}"]
        # Rotated by (bomb.rotation + 180) % 360
        self.explosion_sprites = {}  # (bomb_width, frame_index, rotation) -> pixels
        # We need access to the game's sprite dict
        import importlib
        game_module = sys.modules.get('ka59')
        if game_module is None:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../environment_files/ka59/9f096b4a'))
            import ka59 as game_module
        game_sprites = game_module.sprites

        for bw in set(self.bomb_widths):
            for frame_idx in range(1, 4):
                key = f"explode-{bw}-{frame_idx}"
                if key in game_sprites:
                    base_pixels = game_sprites[key].pixels.copy()
                    for rot in [0, 90, 180, 270]:
                        rendered = rotate_pixels(base_pixels, rot)
                        self.explosion_sprites[(bw, frame_idx, rot)] = rendered

        # Goals
        self.goals = []
        for s in level.get_sprites_by_tag('rktpmjcpkt'):
            self.goals.append((s.x + 1, s.y + 1, s.width - 2, s.height - 2))

        self.sec_goals = []
        for s in level.get_sprites_by_tag('ucjzrlvfkb'):
            self.sec_goals.append((s.x + 1, s.y + 1, s.width - 2, s.height - 2))

        self.step_limit = level.get_data('StepCounter')

    def collides_with_walls(self, sx, sy, spix):
        for wx, wy, wpix in self.walls:
            if pixel_collides(sx, sy, spix, wx, wy, wpix):
                return True
        return False

    def collides_with_walls_and_gulches(self, sx, sy, spix):
        for wx, wy, wpix in self.walls:
            if pixel_collides(sx, sy, spix, wx, wy, wpix):
                return True
        for gx, gy, gpix in self.gulches:
            if pixel_collides(sx, sy, spix, gx, gy, gpix):
                return True
        return False

    def on_gulch(self, sx, sy, spix):
        for gx, gy, gpix in self.gulches:
            if pixel_collides(sx, sy, spix, gx, gy, gpix):
                return True
        return False

    def find_pushable_collisions(self, sx, sy, spix, positions, exclude_idx=-1):
        collisions = []
        for i in range(self.n_push):
            if i == exclude_idx:
                continue
            px, py = positions[i]
            if px is None:  # Destroyed bomb
                continue
            ppix = self.push_pixels[i]
            if pixel_collides(sx, sy, spix, px, py, ppix):
                collisions.append(i)
        return collisions

    def try_push_recursive(self, idx, dx, dy, positions, depth=0):
        """Push sprite idx by (dx,dy), recursively pushing others.
        Checks walls only (not gulches) — matching esglqyymck."""
        if depth > self.n_push:
            return None

        px, py = positions[idx]
        if px is None:
            return None
        ppix = self.push_pixels[idx]
        nx, ny = px + dx, py + dy

        if self.collides_with_walls(nx, ny, ppix):
            return None

        collisions = self.find_pushable_collisions(nx, ny, ppix, positions, exclude_idx=idx)

        if not collisions:
            new_pos = list(positions)
            new_pos[idx] = (nx, ny)
            return tuple(new_pos)

        new_pos = list(positions)
        new_pos[idx] = (nx, ny)
        for ci in collisions:
            result = self.try_push_recursive(ci, dx, dy, tuple(new_pos), depth + 1)
            if result is None:
                return None
            new_pos = list(result)

        return tuple(new_pos)

    def simulate_push_animation(self, pushed_indices, push_dirs, positions):
        """Simulate push animation matching engine's internal loop."""
        new_pos = list(positions)
        frame = 0
        MAX_FRAMES = 200

        while frame < MAX_FRAMES:
            all_done = True
            for pi in pushed_indices:
                dx, dy = push_dirs[pi]
                ppix = self.push_pixels[pi]
                cur_x, cur_y = new_pos[pi]
                if cur_x is None:
                    continue

                if frame >= PUSH_FRAMES:
                    if not self.on_gulch(cur_x, cur_y, ppix):
                        continue

                result = self.try_push_recursive(pi, dx, dy, tuple(new_pos))
                if result is None:
                    continue

                new_pos = list(result)
                all_done = False

            if all_done:
                break
            frame += 1

        return tuple(new_pos)

    def simulate_explosion(self, bomb_idx_in_bombs, positions):
        """Simulate bomb explosion for a fully consumed bomb.

        bomb_idx_in_bombs: index into self.bomb_sprites (not pushable index)
        Returns new positions after explosion effects.
        """
        push_idx = self.n_player + self.n_et + bomb_idx_in_bombs
        bomb_x, bomb_y = positions[push_idx]
        if bomb_x is None:
            return positions

        bomb_rot = self.bomb_rotations[bomb_idx_in_bombs]
        bomb_w = self.bomb_widths[bomb_idx_in_bombs]
        explosion_rot = (bomb_rot + 180) % 360

        # Determine push direction from bomb rotation
        if bomb_rot == 0:
            edx, edy = 0, STEP    # Push DOWN
        elif bomb_rot == 90:
            edx, edy = -STEP, 0   # Push LEFT
        elif bomb_rot == 180:
            edx, edy = 0, -STEP   # Push UP
        elif bomb_rot == 270:
            edx, edy = STEP, 0    # Push RIGHT

        new_pos = list(positions)
        push_dirs = {}

        # 3 explosion frames
        for frame_idx in range(1, 4):
            offset = STEP * frame_idx
            exp_key = (bomb_w, frame_idx, explosion_rot)
            if exp_key not in self.explosion_sprites:
                continue

            exp_pixels = self.explosion_sprites[exp_key]
            exp_x = bomb_x - offset
            exp_y = bomb_y - offset

            # Check collision with all pushables
            for pi in range(self.n_push):
                if pi == push_idx:
                    continue  # Don't push the bomb itself
                px, py = new_pos[pi]
                if px is None:
                    continue
                ppix = self.push_pixels[pi]

                if pixel_collides(exp_x, exp_y, exp_pixels, px, py, ppix):
                    # Push this object
                    result = self.try_push_recursive(pi, edx, edy, tuple(new_pos))
                    if result is not None:
                        new_pos = list(result)
                        push_dirs[pi] = (edx, edy)

        # After explosion: bomb pixels blanked to 13 (zugimcgfks)
        # Bomb stays in place but all rows become "unconsumed" again
        # This is handled by resetting bomb_rows in apply_bomb_effects

        # Check if any pushed objects are on gulch and continue moving
        for pi, (pdx, pdy) in push_dirs.items():
            px, py = new_pos[pi]
            if px is None:
                continue
            ppix = self.push_pixels[pi]
            if self.on_gulch(px, py, ppix):
                # Object on gulch: keep moving until off gulch or blocked
                while True:
                    result = self.try_push_recursive(pi, pdx, pdy, tuple(new_pos))
                    if result is None:
                        break
                    new_pos = list(result)
                    npx, npy = new_pos[pi]
                    if not self.on_gulch(npx, npy, self.push_pixels[pi]):
                        break

        return tuple(new_pos)

    def apply_bomb_effects(self, positions, bomb_rows):
        """After each player action, consume bomb rows and handle explosions.

        Engine flow: wzfqyavgic -> egsdqzlidg for each bomb.
        After explosion, eohhtwksbe blanks all pixels to 13, making all rows
        unconsumed again. Bombs cycle: consume -> explode -> blank -> consume...
        """
        if not self.n_bombs:
            return positions, bomb_rows

        new_rows = list(bomb_rows)
        new_pos = positions

        # Consume one row from each bomb
        exploding = []
        for bi in range(self.n_bombs):
            if new_rows[bi] <= 0:
                continue  # Shouldn't happen, but safety check
            push_idx = self.n_player + self.n_et + bi
            if new_pos[push_idx][0] is None:
                continue  # Somehow destroyed

            new_rows[bi] -= 1
            if new_rows[bi] == 0:
                exploding.append(bi)

        # Process explosions
        for bi in exploding:
            new_pos = self.simulate_explosion(bi, new_pos)
            # After explosion, bomb is blanked: all rows become unconsumed
            # Reset to full height
            bomb_height = self.push_pixels[self.n_player + self.n_et + bi].shape[0]
            new_rows[bi] = bomb_height

        return new_pos, tuple(new_rows)

    def try_move(self, positions, sel, dx, dy, bomb_rows):
        """Try to move selected box in direction (dx,dy).

        Returns (new_positions, new_bomb_rows) or (None, None) if blocked.
        """
        bx, by = positions[sel]
        bpix = self.push_pixels[sel]
        nx, ny = bx + dx, by + dy

        # Player box checks walls AND gulches
        if self.collides_with_walls_and_gulches(nx, ny, bpix):
            # Move blocked, but bombs still consume!
            new_pos, new_rows = self.apply_bomb_effects(positions, bomb_rows)
            if new_pos == positions:
                return None, None  # No change at all
            return new_pos, new_rows

        # Check collision with other pushables
        push_targets = self.find_pushable_collisions(nx, ny, bpix, positions, exclude_idx=sel)

        if not push_targets:
            # Free move
            new_pos = list(positions)
            new_pos[sel] = (nx, ny)
            new_pos, new_rows = self.apply_bomb_effects(tuple(new_pos), bomb_rows)
            return new_pos, new_rows

        # Push: selected box STAYS at original position (aqoyvqxpct reverts)
        # The actual pushing happens in the animation loop (tesoceiypr)
        push_dirs = {pi: (dx, dy) for pi in push_targets}

        # Simulate the full push animation starting from current positions
        # tesoceiypr at frame 0 does the first eslqyymck push
        final_pos = self.simulate_push_animation(push_targets, push_dirs, positions)

        # If nothing moved, push was blocked from the start
        if final_pos == positions:
            new_pos, new_rows = self.apply_bomb_effects(positions, bomb_rows)
            if new_pos == positions:
                return None, None
            return new_pos, new_rows

        # Apply bomb effects after push
        final_pos, new_rows = self.apply_bomb_effects(final_pos, bomb_rows)

        return final_pos, new_rows

    def is_win(self, positions):
        # Primary goals: covered by player boxes
        for gx, gy, gw, gh in self.goals:
            found = False
            for j in range(self.n_player):
                bx, by = positions[j]
                if bx is None:
                    continue
                bw, bh = self.push_dims[j]
                if bx == gx and by == gy and bw == gw and bh == gh:
                    found = True
                    break
            if not found:
                return False

        # Secondary goals: covered by enemy targets
        for gx, gy, gw, gh in self.sec_goals:
            found = False
            for j in range(self.n_player, self.n_player + self.n_et):
                bx, by = positions[j]
                if bx is None:
                    continue
                bw, bh = self.push_dims[j]
                if bx == gx and by == gy and bw == gw and bh == gh:
                    found = True
                    break
            if not found:
                return False

        return True

    def heuristic(self, positions):
        total = 0
        for gx, gy, gw, gh in self.goals:
            best_d = float('inf')
            for j in range(self.n_player):
                bx, by = positions[j]
                if bx is None:
                    continue
                bw, bh = self.push_dims[j]
                if bw == gw and bh == gh:
                    d = abs(bx - gx) + abs(by - gy)
                    if d < best_d:
                        best_d = d
            if best_d < float('inf'):
                total += best_d // STEP

        for gx, gy, gw, gh in self.sec_goals:
            best_d = float('inf')
            for j in range(self.n_player, self.n_player + self.n_et):
                bx, by = positions[j]
                if bx is None:
                    continue
                bw, bh = self.push_dims[j]
                if bw == gw and bh == gh:
                    d = abs(bx - gx) + abs(by - gy)
                    if d < best_d:
                        best_d = d
            if best_d < float('inf'):
                total += best_d // STEP

        return total


def solve_level(sim, max_steps=80, max_states=2000000):
    """A* search with pixel-perfect simulation."""
    init_pos = tuple((p.x, p.y) for p in
                     list(sim.player_sprites) + list(sim.et_sprites) + list(sim.bomb_sprites))
    init_bomb_rows = tuple(sim.bomb_init_rows)

    print(f"  Player boxes: {sim.n_player}, Enemy targets: {sim.n_et}, Bombs: {sim.n_bombs}")
    print(f"  Primary goals: {len(sim.goals)}, Secondary goals: {len(sim.sec_goals)}")
    for i in range(sim.n_push):
        if i < sim.n_player:
            tag = 'box'
        elif i < sim.n_player + sim.n_et:
            tag = 'et'
        else:
            tag = 'bomb'
        px, py = init_pos[i]
        w, h = sim.push_dims[i]
        print(f"    {tag}[{i}]: ({px},{py}) {w}x{h}")
    for i, (gx, gy, gw, gh) in enumerate(sim.goals):
        print(f"    goal[{i}]: target ({gx},{gy}) {gw}x{gh}")
    for i, (gx, gy, gw, gh) in enumerate(sim.sec_goals):
        print(f"    sec_goal[{i}]: target ({gx},{gy}) {gw}x{gh}")
    if sim.n_bombs:
        print(f"  Bomb rows remaining: {init_bomb_rows}")
        print(f"  Bomb rotations: {sim.bomb_rotations}")

    if sim.is_win(init_pos):
        return []

    # State includes positions AND bomb rows
    init_state = (init_pos, init_bomb_rows)
    h0 = sim.heuristic(init_pos)
    visited = {init_state}
    counter = 0
    pq = [(h0, counter, init_pos, init_bomb_rows, [])]

    directions = [(0, -STEP, 'UP'), (0, STEP, 'DOWN'), (-STEP, 0, 'LEFT'), (STEP, 0, 'RIGHT')]
    states_explored = 0

    while pq:
        f, _, positions, bomb_rows, actions = heapq.heappop(pq)
        g = len(actions)

        if g >= max_steps:
            continue

        states_explored += 1
        if states_explored > max_states:
            print(f"  A* exhausted at {max_states} states")
            break

        if states_explored % 100000 == 0:
            print(f"  ... {states_explored} states, pq={len(pq)}, depth={g}, f={f}")

        for box_idx in range(sim.n_player):
            if positions[box_idx][0] is None:
                continue  # Destroyed player box (shouldn't happen)
            for dx, dy, dname in directions:
                result_pos, result_rows = sim.try_move(positions, box_idx, dx, dy, bomb_rows)
                if result_pos is None:
                    continue

                state = (result_pos, result_rows)
                if state in visited:
                    continue

                new_actions = actions + [(box_idx, dname)]
                if sim.is_win(result_pos):
                    print(f"  SOLUTION! {len(new_actions)} moves, {states_explored} states")
                    return new_actions

                visited.add(state)
                h = sim.heuristic(result_pos)
                counter += 1
                heapq.heappush(pq, (g + 1 + h, counter, result_pos, result_rows, new_actions))

    print(f"  No solution found ({states_explored} states)")
    return None


def execute_solution(env, game, solution, level_idx):
    """Execute solution on engine."""
    boxes = game.current_level.get_sprites_by_tag('xlfuqjygey')
    offset = get_display_offset(game)
    current_sel = 0
    total_steps = 0
    fd = None

    for step_i, (box_idx, dname) in enumerate(solution):
        if box_idx != current_sel:
            boxes = game.current_level.get_sprites_by_tag('xlfuqjygey')
            fd_click = click_box(env, game, boxes[box_idx], offset)
            current_sel = box_idx
            total_steps += 1
            # Check if click triggered level change
            if game.level_index > level_idx:
                return total_steps, True
            if hasattr(fd_click, 'state') and fd_click.state.name in ('WON', 'GAME_OVER'):
                return total_steps, fd_click.state.name == 'WON'

        fd = env.step(NAME_TO_ACT[dname])
        total_steps += 1

        if game.level_index > level_idx:
            return total_steps, True
        if hasattr(fd, 'state') and fd.state.name in ('WON', 'WIN'):
            return total_steps, True
        if hasattr(fd, 'state') and fd.state.name == 'GAME_OVER':
            return total_steps, False

    won = game.level_index > level_idx
    if fd is not None and hasattr(fd, 'state') and fd.state.name in ('WON', 'WIN'):
        won = True
    return total_steps, won


def main():
    print("=" * 60)
    print("ka59 Sokoban Solver — Pixel-Perfect A*")
    print("=" * 60)

    os.makedirs(VISUAL_DIR, exist_ok=True)

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

        if fd.state.name in ('WON', 'WIN', 'GAME_OVER'):
            print(f"  Game ended: {fd.state.name}")
            break

        step_counter = game.current_level.get_data('StepCounter')
        print(f"  Step counter: {step_counter}")

        save_frame(env, game, f'L{level}_initial.png')

        t0 = time.time()
        sim = LevelSim(game)
        max_bfs = min(step_counter, 80)
        solution = solve_level(sim, max_steps=max_bfs, max_states=2000000)
        dt = time.time() - t0
        print(f"  Solver time: {dt:.1f}s")

        if solution is None:
            print(f"  FAILED to solve L{level}")
            save_frame(env, game, f'L{level}_failed.png')
            break

        if len(solution) == 0:
            print(f"  L{level} trivially solved!")
            results[level] = {'actions': 0, 'time': dt}
            continue

        print(f"  Solution: {len(solution)} moves")
        for i, (bidx, dname) in enumerate(solution):
            print(f"    {i}: box[{bidx}] {dname}")

        print(f"  Executing on engine...")
        n_steps, ok = execute_solution(env, game, solution, level)

        if ok or game.level_index > level or (hasattr(fd, 'state') and fd.state.name in ('WON', 'WIN')):
            print(f"  L{level} SOLVED! ({n_steps} engine steps)")
            results[level] = {'actions': len(solution), 'engine_steps': n_steps, 'time': dt}
            total_actions += len(solution)
            save_frame(env, game, f'L{level}_solved.png')
        else:
            print(f"  L{level} execution FAILED")
            boxes = game.current_level.get_sprites_by_tag('xlfuqjygey')
            goals = game.current_level.get_sprites_by_tag('rktpmjcpkt')
            ets = game.current_level.get_sprites_by_tag('nnckfubbhi')
            sgs = game.current_level.get_sprites_by_tag('ucjzrlvfkb')
            for i, b in enumerate(boxes):
                print(f"    box[{i}]: ({b.x},{b.y}) {b.width}x{b.height}")
            for i, g in enumerate(goals):
                print(f"    goal[{i}]: target ({g.x+1},{g.y+1})")
            for i, e in enumerate(ets):
                print(f"    et[{i}]: ({e.x},{e.y}) {e.width}x{e.height}")
            for i, s in enumerate(sgs):
                print(f"    sec_goal[{i}]: target ({s.x+1},{s.y+1})")
            save_frame(env, game, f'L{level}_failed.png')
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
