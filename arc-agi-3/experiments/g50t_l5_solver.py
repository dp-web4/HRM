#!/usr/bin/env python3
"""
g50t L5 BFS Solver — models full game state including UNDO/ghost mechanics.

Strategy: phased BFS
  Phase 0: BFS from start, find shortest paths to each modifier (ghost1 candidates)
  Phase 1: For each ghost1 path, BFS from start with ghost1 replaying (ghost2 candidates)
  Phase 2: For each (ghost1, ghost2) pair, BFS final run to goal
"""

import sys, os, json, copy
from collections import deque
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # UP DOWN LEFT RIGHT
DIR_NAMES = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'UNDO'}
STEP = 6

# First, replay L1-L4 solutions to reach L5
L1_SOL = 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT LEFT LEFT UP'
L2_SOL = 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT'
L3_SOL = 'UP UP RIGHT RIGHT UNDO DOWN DOWN DOWN DOWN UP DOWN UP DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT'
L4_SOL = 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT'

NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}

def replay_to_level(env, target_level):
    """Replay solutions for L1-L4 to reach target level."""
    sols = [L1_SOL, L2_SOL, L3_SOL, L4_SOL]
    fd = env.reset()
    for i, sol in enumerate(sols):
        if fd.levels_completed >= target_level - 1:
            break
        for name in sol.split():
            fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
        print(f"  L{i+1} done, levels_completed={fd.levels_completed}")
    return fd


def extract_l5_state(env):
    """Extract all L5 game state for solver."""
    game = env._game
    lc = game.vgwycxsxjz

    player = lc.dzxunlkwxt
    goal = lc.whftgckbcu
    arena = lc.afbbgvkpip
    obstacles = lc.uwxkstolmf
    modifiers = lc.hamayflsib
    indicators = lc.drofvwhbxb

    print(f"\n=== L5 State ===")
    print(f"Player: ({player.x}, {player.y})")
    print(f"Goal: ({goal.x}, {goal.y})")
    print(f"Arena: ({arena.x}, {arena.y}) {arena.width}x{arena.height}")
    print(f"Start: ({lc.yugzlzepkr}, {lc.vgpdqizwwm})")
    print(f"Step size: {STEP}")
    print(f"Undo count: {lc.rlazdofsxb}")
    print(f"Max undos: {len(indicators) - 1}")
    print(f"areahjypvy: {lc.areahjypvy}")
    print(f"Ghost paths: {dict(lc.rloltuowth)}")

    print(f"\nObstacles ({len(obstacles)}):")
    for i, obs in enumerate(obstacles):
        print(f"  obs[{i}]: ({obs.x}, {obs.y}) visible={obs.is_visible} {obs.width}x{obs.height}")

    print(f"\nModifiers ({len(modifiers)}):")
    for i, mod in enumerate(modifiers):
        print(f"  mod[{i}]: ({mod.x}, {mod.y})")
        # Check signal chain
        pc = mod.nexhtmlmxh  # path constrainer
        if pc:
            obs_list = pc.ytztewxdin
            print(f"    → clears obstacles: {[(o.x, o.y) for o in obs_list]}")
            # Check rotation
            print(f"    rotation: {mod.rotation}")

    # Map walkable grid using game's collision check
    walkable = set()
    obstacle_cells = set()
    orig_x, orig_y = player.x, player.y

    ax, ay = arena.x, arena.y
    for px in range(ax - 6, ax + 70, STEP):
        for py in range(ay - 6, ay + 70, STEP):
            player.set_position(px, py)
            if lc.xvkyljflji(player, arena):
                if lc.vjpujwqrto(player):
                    obstacle_cells.add((px, py))
                else:
                    walkable.add((px, py))

    player.set_position(orig_x, orig_y)

    print(f"\nWalkable cells: {len(walkable)}")
    print(f"Obstacle-blocked cells: {len(obstacle_cells)}")

    # Also map walkable WITHOUT obstacles (potential cells if obstacles cleared)
    # Temporarily hide obstacles
    for obs in obstacles:
        obs.is_visible = False

    all_cells = set()
    for px in range(ax - 6, ax + 70, STEP):
        for py in range(ay - 6, ay + 70, STEP):
            player.set_position(px, py)
            if lc.xvkyljflji(player, arena):
                all_cells.add((px, py))

    for obs in obstacles:
        obs.is_visible = True
    player.set_position(orig_x, orig_y)

    print(f"Total cells (no obstacles): {len(all_cells)}")
    print(f"Cells blocked by obstacles: {obstacle_cells}")

    # Build modifier → obstacle mapping
    mod_to_obs = {}
    for i, mod in enumerate(modifiers):
        pc = mod.nexhtmlmxh
        if pc:
            for j, obs in enumerate(obstacles):
                if obs in pc.ytztewxdin:
                    mod_to_obs[i] = j
    print(f"\nmod_to_obs: {mod_to_obs}")

    # Check obstacle shift directions
    for i, mod in enumerate(modifiers):
        rot = mod.rotation
        if rot == 0:
            shift = "DOWN"
        elif rot == 90:
            shift = "LEFT"
        elif rot == 180:
            shift = "UP"
        elif rot == 270:
            shift = "RIGHT"
        else:
            shift = f"rot={rot}"
        print(f"  mod[{i}] rot={rot} → obstacle shifts {shift}")

    return {
        'walkable': walkable,
        'all_cells': all_cells,
        'obstacle_cells': obstacle_cells,
        'player_start': (lc.yugzlzepkr, lc.vgpdqizwwm),
        'player_pos': (player.x, player.y),
        'goal': (goal.x, goal.y),
        'obstacles': [(obs.x, obs.y) for obs in obstacles],
        'modifiers': [(mod.x, mod.y) for mod in modifiers],
        'mod_to_obs': mod_to_obs,
        'max_undos': len(indicators) - 1,
        'areahjypvy': list(lc.areahjypvy),
    }


class G50TSim:
    """Minimal simulator for g50t ghost/undo mechanics."""

    def __init__(self, state_data):
        self.walkable = state_data['all_cells']  # all cells, obstacles handled separately
        self.player_start = state_data['player_start']
        self.goal = state_data['goal']
        self.obstacles = list(state_data['obstacles'])  # base positions
        self.modifiers = state_data['modifiers']
        self.mod_to_obs = state_data['mod_to_obs']
        self.max_undos = state_data['max_undos']
        self.init_path = list(state_data['areahjypvy'])

        # Obstacle shift when modifier activated (rotation-based)
        # Will be filled in by extract
        self.obs_shift = {}  # obs_idx → (dx, dy) shift when modifier activates

    def initial_state(self):
        """Return initial game state."""
        return {
            'px': self.player_start[0] + len(self.init_path) * STEP if self.init_path else self.player_start[0],
            'py': self.player_start[1],
            'undo_count': 0,
            'path': list(self.init_path),  # areahjypvy
            'ghosts': [],  # list of (start_x, start_y, path)
            'step_in_run': len(self.init_path),  # ghost replay index
        }

    def get_ghost_positions(self, ghosts, step_idx):
        """Get current position of each ghost at given step index."""
        positions = []
        for gsx, gsy, gpath in ghosts:
            # Replay path from start
            gx, gy = gsx, gsy
            for s in range(min(step_idx, len(gpath))):
                dx, dy = gpath[s]
                nx, ny = gx + dx * STEP, gy + dy * STEP
                # Check if move is valid (simplified — assume ghost paths are pre-validated)
                gx, gy = nx, ny
            positions.append((gx, gy))
        return positions

    def get_active_mods(self, player_pos, ghost_positions):
        """Return set of modifier indices that are active (entity on them)."""
        active = set()
        for i, (mx, my) in enumerate(self.modifiers):
            if player_pos == (mx, my):
                active.add(i)
            for gp in ghost_positions:
                if gp == (mx, my):
                    active.add(i)
        return active

    def get_cleared_obstacles(self, active_mods):
        """Return set of obstacle indices that are cleared."""
        cleared = set()
        for mod_idx in active_mods:
            if mod_idx in self.mod_to_obs:
                cleared.add(self.mod_to_obs[mod_idx])

        return cleared

    def is_walkable(self, x, y, cleared_obs):
        """Check if position is walkable given cleared obstacles."""
        if (x, y) not in self.walkable:
            return False
        # Check obstacles
        for i, (ox, oy) in enumerate(self.obstacles):
            if i in cleared_obs:
                continue
            # Obstacle collision: check if player at (x,y) overlaps obstacle at (ox,oy)
            # For simplicity, assume obstacle blocks the cell it's on
            if (x, y) == (ox, oy):
                return False
        return True

    def check_win(self, px, py):
        """Check win condition: player.x == goal.x + 1 AND player.y == goal.y + 1"""
        gx, gy = self.goal
        return px == gx + 1 and py == gy + 1


def solve_with_sdk(env, state_data):
    """
    Use SDK directly for BFS — most accurate simulation.
    Reset to L5, try actions, use env state for correctness.
    """
    game = env._game
    lc = game.vgwycxsxjz

    # Can we deepcopy the game state? Let's try.
    import pickle
    try:
        state_bytes = pickle.dumps(lc)
        print(f"State pickle: {len(state_bytes)} bytes — deepcopy possible!")
    except Exception as e:
        print(f"Can't pickle state: {e}")
        # Try copy.deepcopy
        try:
            lc_copy = copy.deepcopy(lc)
            print("deepcopy works!")
        except Exception as e2:
            print(f"Can't deepcopy either: {e2}")
            return None

    return None  # placeholder


def phase0_bfs(state_data):
    """Phase 0: Find shortest paths from start to each modifier (no ghosts yet)."""
    walkable = state_data['all_cells']
    obstacles_set = set()
    for i, (ox, oy) in enumerate(state_data['obstacles']):
        obstacles_set.add((ox, oy))

    # Cells walkable with no modifiers active = walkable - obstacles
    base_walkable = walkable - obstacles_set

    start = state_data['player_start']
    # But player might start at a different position due to init_path
    # If init_path is non-empty, player has moved from start
    px, py = start
    for dx, dy in state_data['areahjypvy']:
        px += dx * STEP
        py += dy * STEP
    actual_start = (px, py)

    print(f"\nPhase 0: BFS from actual_start={actual_start} (undo_start={start})")
    print(f"Base walkable (no mods active): {len(base_walkable)} cells")

    # BFS from actual_start
    # State: (x, y)
    # We record the path (sequence of (dx,dy) moves)
    queue = deque()
    queue.append((actual_start, []))
    visited = {actual_start}

    paths_to_mods = {}

    while queue:
        (x, y), path = queue.popleft()

        # Check if we're on a modifier
        for i, (mx, my) in enumerate(state_data['modifiers']):
            if (x, y) == (mx, my) and i not in paths_to_mods:
                paths_to_mods[i] = list(path)
                print(f"  Found path to mod[{i}] at ({mx},{my}): {len(path)} moves")

        if len(paths_to_mods) == len(state_data['modifiers']):
            break  # found all

        for action, (dx, dy) in DIRS.items():
            nx, ny = x + dx * STEP, y + dy * STEP
            if (nx, ny) in base_walkable and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(dx, dy)]))

    print(f"\nReachable modifiers (no ghosts): {list(paths_to_mods.keys())}")
    return paths_to_mods


def simulate_ghost_at_step(ghost_start, ghost_path, step_idx, all_cells, obstacles):
    """Get ghost position at a given step index."""
    gx, gy = ghost_start
    for s in range(min(step_idx, len(ghost_path))):
        dx, dy = ghost_path[s]
        nx, ny = gx + dx * STEP, gy + dy * STEP
        # Simplified: assume ghost moves are valid (they were recorded as valid moves)
        gx, gy = nx, ny
    return (gx, gy)


def phase2_bfs(state_data, ghost1_full_path, ghost2_full_path):
    """
    Phase 2: BFS final run with both ghosts replaying.

    ghost1_full_path and ghost2_full_path are the FULL paths including init_path prefix.
    In the final run, at step k, ghost replays step k of its full path.
    If k >= len(path), ghost is SKIPPED (stays at final position, modifier stays active).
    """
    start = state_data['player_start']
    goal = state_data['goal']
    all_cells = state_data['all_cells']
    obstacles = state_data['obstacles']
    modifiers = state_data['modifiers']
    mod_to_obs = state_data['mod_to_obs']

    # Ghost start positions = undo_start (player_start)
    g1_start = start
    g2_start = start

    # State: (player_x, player_y, step_counter)
    # Ghost positions are deterministic from step_counter
    # Max steps before we give up
    MAX_STEPS = 80

    queue = deque()
    queue.append((start[0], start[1], 0, []))  # (px, py, step, actions)
    visited = set()
    visited.add((start[0], start[1], 0))

    while queue:
        px, py, step, actions = queue.popleft()

        if step > MAX_STEPS:
            continue

        # Try each direction
        for action, (dx, dy) in DIRS.items():
            nx, ny = px + dx * STEP, py + dy * STEP
            new_step = step + 1

            if (nx, ny) not in all_cells:
                continue

            # Determine ghost positions AFTER this step
            # Ghost processes at step_index = new_step - 1 (0-indexed)
            # Actually, in move(): after player moves, areahjypvy.append,
            # step_index = len(areahjypvy) - 1 = new_step - 1 (since we start with empty path)
            # Wait, in final run, areahjypvy starts empty again after last UNDO
            step_idx = new_step  # step index for ghost replay

            # Get ghost positions BEFORE their move at this step
            g1_pos_before = simulate_ghost_at_step(g1_start, ghost1_full_path, step_idx - 1, all_cells, obstacles)
            g2_pos_before = simulate_ghost_at_step(g2_start, ghost2_full_path, step_idx - 1, all_cells, obstacles)

            # Check what modifiers are active BEFORE movement (from previous step's activation)
            # Actually, activation happens AFTER movement. So at the start of this step:
            # - entities are at their positions from end of previous step
            # - modifiers active based on those positions

            # Hmm, the exact timing is:
            # 1. Player moves (rdgrjozeoh: deactivate old mod, move)
            # 2. areahjypvy.append
            # 3. Each ghost: if path not exhausted, rdgrjozeoh (deactivate old mod, move)
            # 4. Activate mods for player and all ghosts

            # For obstacle collision during player move:
            # The check happens at the NEW position BEFORE any activation
            # But AFTER deactivation of old position
            # So we need to know which obstacles are blocking at the moment of player move

            # At the moment of player move attempt:
            # - Previous step's activations are still in effect? No...
            # - Actually, deactivation happens in rdgrjozeoh BEFORE the move
            # - Activation happens AFTER ALL movements

            # So the sequence for a full step is:
            # 1. Player: deactivate mod at old pos, try move to new pos (check obstacles)
            # 2. Ghost1: deactivate mod at old pos, try move (check obstacles)
            # 3. Ghost2: deactivate mod at old pos, try move (check obstacles)
            # 4. Activate mod for player at new pos
            # 5. Activate mod for ghost1 at new pos
            # 6. Activate mod for ghost2 at new pos

            # Wait, that's not right either. Let me re-read the code flow.

            # In step() (the main game step):
            #   self.move(dx, dy)  → player moves, then each ghost replays
            #     move():
            #       rdgrjozeoh(player, dx, dy)  → deactivate(player,False), move player
            #       for ghost in ghosts:
            #         if path exhausted: continue (SKIP)
            #         rdgrjozeoh(ghost, gdx, gdy) → deactivate(ghost,False), move ghost
            #   THEN (back in step()):
            #     ayhgaxoxce(player, True)   → activate for player
            #     for subplayer: activate
            #     for ghost: activate

            # So during player move, the obstacle state depends on:
            # - Which modifiers were activated at END of previous step
            # - MINUS the player's old modifier (just deactivated)
            # - Ghost modifiers from previous step are still active until ghost processes

            # This is complex. Let me simplify:
            # The key question is: when the player tries to move to (nx,ny), which obstacles block?
            # At that moment:
            # - Player just deactivated its old modifier
            # - Ghosts haven't moved yet (their modifiers from prev step still active)
            # - Ghosts with exhausted paths: their modifier is still active from prev step

            # Actually wait, deactivation in rdgrjozeoh calls ayhgaxoxce(entity, False).
            # ayhgaxoxce removes entity from modifier's vbqvjbxkfm set.
            # Modifier is active IFF vbqvjbxkfm is non-empty.
            # So if ghost is ALSO on the same modifier, it stays active even after player deactivates.

            # For the simplified solver, let me track which entities are on which modifiers.

            # Positions at end of previous step (after activation):
            if step_idx == 1:
                # First step: positions are start positions
                player_old = (px, py)  # should be start
                g1_old = g1_start
                g2_old = g2_start
            else:
                player_old = (px, py)
                g1_old = simulate_ghost_at_step(g1_start, ghost1_full_path, step_idx - 1, all_cells, obstacles)
                g2_old = simulate_ghost_at_step(g2_start, ghost2_full_path, step_idx - 1, all_cells, obstacles)

            # Entities on each modifier at end of previous step
            active_mods_prev = set()
            for mi, (mx, my) in enumerate(modifiers):
                entities_on = set()
                if player_old == (mx, my):
                    entities_on.add('player')
                if g1_old == (mx, my):
                    entities_on.add('g1')
                if g2_old == (mx, my):
                    entities_on.add('g2')
                if entities_on:
                    active_mods_prev.add(mi)

            # After player deactivates (removes self from old modifier):
            active_mods_during_player_move = set()
            for mi, (mx, my) in enumerate(modifiers):
                entities_on = set()
                # Player removed from old mod
                if g1_old == (mx, my):
                    entities_on.add('g1')
                if g2_old == (mx, my):
                    entities_on.add('g2')
                if entities_on:
                    active_mods_during_player_move.add(mi)

            # Cleared obstacles during player move
            cleared_during_player = set()
            for mi in active_mods_during_player_move:
                if mi in mod_to_obs:
                    cleared_during_player.add(mod_to_obs[mi])

            # Check if player can move to (nx, ny)
            blocked = False
            for oi, (ox, oy) in enumerate(obstacles):
                if oi in cleared_during_player:
                    continue
                if (nx, ny) == (ox, oy):
                    blocked = True
                    break

            if blocked:
                continue

            # Player successfully moves to (nx, ny)
            # Now ghosts process
            g1_new = simulate_ghost_at_step(g1_start, ghost1_full_path, step_idx, all_cells, obstacles)
            g2_new = simulate_ghost_at_step(g2_start, ghost2_full_path, step_idx, all_cells, obstacles)

            # Check win after all movement + activation
            gx, gy = goal
            if nx == gx + 1 and ny == gy + 1:
                print(f"  SOLUTION FOUND! {len(actions)+1} actions")
                final_actions = actions + [action]
                return final_actions

            state_key = (nx, ny, new_step)
            if state_key not in visited:
                visited.add(state_key)
                queue.append((nx, ny, new_step, actions + [action]))

    return None


def main():
    arcade = Arcade()
    env = arcade.make('g50t-5849a774')

    print("Replaying L1-L4...")
    fd = replay_to_level(env, 5)
    print(f"\nAt level {fd.levels_completed}, state={fd.state.name}")

    if fd.levels_completed < 4:
        print("ERROR: Failed to reach L5")
        return

    # Extract L5 state
    state_data = extract_l5_state(env)

    # Phase 0: Find paths to modifiers (no ghosts)
    paths_to_mods = phase0_bfs(state_data)

    # Try SDK state save/restore
    solve_with_sdk(env, state_data)

    # Phase 2 test: Try some ghost path combinations
    # Ghost paths include init_path prefix + moves to modifier + stay
    init_path = state_data['areahjypvy']

    print("\n=== Trying ghost combinations ===")
    for mod_i, path_i in paths_to_mods.items():
        for mod_j, path_j in paths_to_mods.items():
            if mod_i == mod_j:
                continue

            # Ghost1 full path = init_path + path_to_mod[i]
            ghost1_full = init_path + path_i
            # Ghost2 full path = init_path + path_to_mod[j]
            # BUT: ghost2's recording happens AFTER ghost1 exists
            # During ghost2 recording, ghost1 is replaying
            # For now, assume ghost2 can reach mod_j without ghost1's help
            ghost2_full = init_path + path_j

            print(f"\nTrying ghost1→mod[{mod_i}] ({len(ghost1_full)} steps), ghost2→mod[{mod_j}] ({len(ghost2_full)} steps)")

            result = phase2_bfs(state_data, ghost1_full, ghost2_full)
            if result:
                action_names = [DIR_NAMES[a] for a in result]
                print(f"  SOLUTION: {' '.join(action_names)}")

                # Build full action sequence
                phase0_names = [dir_to_name(dx, dy) for dx, dy in path_i]
                phase1_names = [dir_to_name(dx, dy) for dx, dy in path_j]

                full = phase0_names + ['UNDO'] + phase1_names + ['UNDO'] + action_names
                print(f"\n  FULL SEQUENCE ({len(full)} actions): {' '.join(full)}")
                return full

    print("\nNo solution found with direct modifier paths.")
    print("May need ghost paths that go THROUGH obstacles cleared by other ghosts.")


def dir_to_name(dx, dy):
    if dx == 0 and dy == -1: return 'UP'
    if dx == 0 and dy == 1: return 'DOWN'
    if dx == -1 and dy == 0: return 'LEFT'
    if dx == 1 and dy == 0: return 'RIGHT'
    return f'({dx},{dy})'


if __name__ == '__main__':
    main()
