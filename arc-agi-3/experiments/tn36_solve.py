#!/usr/bin/env python3
"""
tn36 World-Model Solver — Programmable block movement.

Approach: For each level, extract grid state (block, target, walls, portals),
then BFS over grid positions to find path, convert path to instructions,
toggle arrow bits, click confirm.

For multi-play levels (L6-L7): chain waypoint-to-waypoint paths.
"""
import sys, os, json, time
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

GAME_ID = 'tn36-ab4f63cc'
GRID = 4  # oocxrjijjq


def get_frame(fd):
    grid = np.array(fd.frame)
    return grid[-1] if grid.ndim == 3 else grid


# Movement commands
CMD_MOVES = {
    1: (-GRID, 0),    # LEFT
    2: (GRID, 0),     # RIGHT
    3: (0, GRID),     # DOWN
    33: (0, -GRID),   # UP
    10: (GRID*2, 0),  # RIGHT*2
    12: (-GRID*2, 0), # LEFT*2
    34: (-GRID, 0),   # LEFT (same as 1)
}

# Rotation commands
CMD_ROTS = {5: 90, 6: -90, 7: 180, 16: 270}

# Scale commands
CMD_SCALES = {8: 1, 9: -1}


def boxes_overlap(ax, ay, aw, ah, bx, by, bw, bh):
    """Check if two axis-aligned boxes overlap."""
    return (ax < bx + bw and ax + aw > bx and
            ay < by + bh and ay + ah > by)


class TN36WorldModel:
    """Lightweight tn36 world model for pathfinding."""

    def __init__(self, game):
        ctrl = game.tsflfunycx
        right = ctrl.xsseeglmfh

        # Block starting state
        self.bx = right.kudckufewr
        self.by = right.rhoxghutkw
        self.brot = right.lixqebpwoq
        self.bscale = right.wedqibqkhx
        self.bcolor = right.cydzcfhdzi
        # Base (unscaled) block dimensions
        self.bw = right.ravxreuqho.width // max(right.ravxreuqho.scale, 1)
        self.bh = right.ravxreuqho.height // max(right.ravxreuqho.scale, 1)

        # Target state
        target = right.ddzsdagbti
        self.tx = target.x
        self.ty = target.y
        self.trot = target.rotation
        self.tscale = target.scale
        self.tcolor = target.dtxpbtpcbh

        # Walls (boundary obstacles)
        self.walls = []
        for w in right.nyxtqtuuny:
            self.walls.append((w.x, w.y, w.width, w.height))

        # Portals (toggle visibility, can kill block)
        self.portals = []
        for p in right.cbbgkmkxku:
            self.portals.append((p.x, p.y, p.width, p.height, p.is_visible))

        # Waypoints
        self.waypoints = []
        for wp in right.wpilazztrp:
            self.waypoints.append((wp.x, wp.y, wp.width, wp.height, wp.scale))

        # Tablet info
        tablet = right.fcrcpemsnv
        self.n_cols = len(tablet.thofkgziyd)
        self.current_instrs = tablet.ylczjoyapu
        self.columns = []
        for col in tablet.thofkgziyd:
            bits = []
            for bit in col.puakvdstpr:
                bits.append({
                    'x': bit.x + bit.width // 2,
                    'y': bit.y + bit.height // 2,
                    'on': bit.hokejgzome,
                })
            self.columns.append(bits)

        # Confirm button
        self.confirm_pos = None
        confirms = game.current_level.get_sprites_by_tag("rlqfpkqktk")
        for c in confirms:
            if right.jfctiffjzp(c.x, c.y):
                self.confirm_pos = (c.x + c.width // 2, c.y + c.height // 2)
                break

        # Tab selectors
        self.tabs = []
        for t in ctrl.mcxkhvobyv:
            self.tabs.append({
                'x': t.x + t.width // 2,
                'y': t.y + t.height // 2,
                'active': t.rdgasfufbq,
            })

        # Multi-tab program data
        self.programs = ctrl.yobglothsl
        self.positions = ctrl.qllkobvsuk
        self.rotations = ctrl.jeikvpsqnt
        self.scales = ctrl.sdtjsirkgr
        self.resets = ctrl.lulfpcdyzl

    def check_wall_collision(self, x, y, scale=1):
        """Check if position collides with any wall."""
        sw = self.bw * scale
        sh = self.bh * scale
        for wx, wy, ww, wh in self.walls:
            if boxes_overlap(x, y, sw, sh, wx, wy, ww, wh):
                return True
        return False

    def check_portal_collision(self, x, y, w, h, visible_portals):
        """Check if position collides with a visible portal."""
        for i, (px, py, pw, ph, _) in enumerate(self.portals):
            if visible_portals[i] and boxes_overlap(x, y, w, h, px, py, pw, ph):
                return True
        return False

    def simulate_instructions(self, instrs):
        """Simulate instruction execution, return final (x, y, rot, scale, color, alive)."""
        x, y = self.bx, self.by
        rot, scale, color = self.brot, self.bscale, self.bcolor
        w, h = self.bw, self.bh
        alive = True

        # Portal visibility state
        portal_vis = [v for _, _, _, _, v in self.portals]

        for step_idx, cmd in enumerate(instrs):
            if not alive:
                break

            # Movement
            if cmd in CMD_MOVES:
                dx, dy = CMD_MOVES[cmd]
                new_x, new_y = x + dx, y + dy
                # Wall check
                if self.check_wall_collision(new_x, new_y, scale):
                    pass  # Movement blocked, stay in place
                else:
                    x, y = new_x, new_y
                    # Portal check
                    if self.check_portal_collision(x, y, self.bw * scale, self.bh * scale, portal_vis):
                        alive = False

            # Rotation
            elif cmd in CMD_ROTS:
                rot = (rot + CMD_ROTS[cmd]) % 360

            # Scale
            elif cmd in CMD_SCALES:
                new_scale = scale + CMD_SCALES[cmd]
                if new_scale > 0:
                    if not self.check_wall_collision(x, y, new_scale):
                        scale = new_scale
                        if self.check_portal_collision(x, y, self.bw * scale, self.bh * scale, portal_vis):
                            alive = False

            # Color
            elif cmd == 14:
                color = 9
            elif cmd == 15:
                color = 8
            elif cmd == 63:
                color = 15

            # Portal toggle every 3rd step (step_idx is 0-based, check at step_idx%3==2)
            if step_idx < len(instrs) - 1 and step_idx % 3 == 2:
                for i in range(len(portal_vis)):
                    portal_vis[i] = not portal_vis[i]
                    if portal_vis[i] and alive:
                        px, py, pw, ph, _ = self.portals[i]
                        if boxes_overlap(x, y, w * scale, h * scale, px, py, pw, ph):
                            alive = False

        return x, y, rot, scale, color, alive

    def find_instructions_to(self, goal_x, goal_y, goal_rot=None, goal_scale=None, goal_color=None, max_depth=None):
        """BFS over instruction sequences to find one reaching a goal position."""
        if max_depth is None:
            max_depth = self.n_cols
        if goal_rot is None: goal_rot = self.brot
        if goal_scale is None: goal_scale = self.bscale
        if goal_color is None: goal_color = self.bcolor

        dx = goal_x - self.bx
        dy = goal_y - self.by
        rot_need = (goal_rot - self.brot) % 360
        scale_need = goal_scale - self.bscale
        color_need = goal_color != self.bcolor

        cmds = set()
        if dx > 0: cmds.update([2, 10])
        if dx < 0: cmds.update([1, 12])
        if dy > 0: cmds.add(3)
        if dy < 0: cmds.add(33)
        cmds.add(0)

        if rot_need == 90: cmds.add(5)
        elif rot_need == 180: cmds.add(7)
        elif rot_need == 270: cmds.add(6)

        if scale_need > 0: cmds.add(8)
        elif scale_need < 0: cmds.add(9)

        if color_need:
            if goal_color == 9: cmds.add(14)
            elif goal_color == 8: cmds.add(15)
            elif goal_color == 15: cmds.add(63)

        cmds = sorted(cmds)

        queue = deque()
        queue.append(([], self.bx, self.by, self.brot, self.bscale, self.bcolor))
        visited = set()
        visited.add((self.bx, self.by, self.brot, self.bscale, self.bcolor, 0))

        while queue:
            instrs, x, y, rot, scale, color = queue.popleft()
            step = len(instrs)

            if step >= max_depth:
                if (x == goal_x and y == goal_y and
                    rot == goal_rot and scale == goal_scale and
                    color == goal_color):
                    return instrs
                continue

            for cmd in cmds:
                new_instrs = instrs + [cmd]
                nx, ny = x, y
                nrot, nscale, ncolor = rot, scale, color

                if cmd in CMD_MOVES:
                    ddx, ddy = CMD_MOVES[cmd]
                    test_x, test_y = nx + ddx, ny + ddy
                    if not self.check_wall_collision(test_x, test_y, nscale):
                        nx, ny = test_x, test_y
                elif cmd in CMD_ROTS:
                    nrot = (nrot + CMD_ROTS[cmd]) % 360
                elif cmd in CMD_SCALES:
                    ns = nscale + CMD_SCALES[cmd]
                    if ns > 0 and not self.check_wall_collision(nx, ny, ns):
                        nscale = ns
                elif cmd == 14: ncolor = 9
                elif cmd == 15: ncolor = 8
                elif cmd == 63: ncolor = 15

                state_key = (nx, ny, nrot, nscale, ncolor, step + 1)
                if state_key in visited:
                    continue
                visited.add(state_key)

                if (nx == goal_x and ny == goal_y and
                    nrot == goal_rot and nscale == goal_scale and
                    ncolor == goal_color):
                    while len(new_instrs) < max_depth:
                        new_instrs.append(0)
                    return new_instrs

                queue.append((new_instrs, nx, ny, nrot, nscale, ncolor))

        return None

    def find_instructions_bfs(self, max_depth=None):
        """BFS over instruction sequences to find one that reaches the target."""
        return self.find_instructions_to(
            self.tx, self.ty, self.trot, self.tscale, self.tcolor, max_depth)

    def find_instructions_bfs_UNUSED(self, max_depth=None):
        """Original BFS (kept for reference)."""
        if max_depth is None:
            max_depth = self.n_cols

        # Determine which commands are potentially useful
        dx = self.tx - self.bx
        dy = self.ty - self.by
        rot_need = (self.trot - self.brot) % 360
        scale_need = self.tscale - self.bscale
        color_need = self.tcolor != self.bcolor

        cmds = set()
        if dx > 0: cmds.update([2, 10])
        if dx < 0: cmds.update([1, 12])
        if dy > 0: cmds.add(3)
        if dy < 0: cmds.add(33)
        cmds.add(0)  # NOP always useful

        if rot_need == 90: cmds.add(5)
        elif rot_need == 180: cmds.add(7)
        elif rot_need == 270: cmds.add(6)

        if scale_need > 0: cmds.add(8)
        elif scale_need < 0: cmds.add(9)

        if color_need:
            if self.tcolor == 9: cmds.add(14)
            elif self.tcolor == 8: cmds.add(15)
            elif self.tcolor == 15: cmds.add(63)

        cmds = sorted(cmds)

        # BFS: state = (x, y, rot, scale, color, step)
        # Expand by trying each command at the current step
        queue = deque()
        queue.append(([], self.bx, self.by, self.brot, self.bscale, self.bcolor))
        visited = set()
        visited.add((self.bx, self.by, self.brot, self.bscale, self.bcolor, 0))

        while queue:
            instrs, x, y, rot, scale, color = queue.popleft()
            step = len(instrs)

            if step >= max_depth:
                # Check if we've reached the target
                if (x == self.tx and y == self.ty and
                    rot == self.trot and scale == self.tscale and
                    color == self.tcolor):
                    return instrs
                continue

            for cmd in cmds:
                new_instrs = instrs + [cmd]
                nx, ny = x, y
                nrot, nscale, ncolor = rot, scale, color
                alive = True

                if cmd in CMD_MOVES:
                    ddx, ddy = CMD_MOVES[cmd]
                    test_x, test_y = nx + ddx, ny + ddy
                    if not self.check_wall_collision(test_x, test_y, nscale):
                        nx, ny = test_x, test_y
                elif cmd in CMD_ROTS:
                    nrot = (nrot + CMD_ROTS[cmd]) % 360
                elif cmd in CMD_SCALES:
                    ns = nscale + CMD_SCALES[cmd]
                    if ns > 0 and not self.check_wall_collision(nx, ny, ns):
                        nscale = ns
                elif cmd == 14: ncolor = 9
                elif cmd == 15: ncolor = 8
                elif cmd == 63: ncolor = 15

                if not alive:
                    continue

                state_key = (nx, ny, nrot, nscale, ncolor, step + 1)
                if state_key in visited:
                    continue
                visited.add(state_key)

                # Early termination: if at target and remaining can be NOPs
                if (nx == self.tx and ny == self.ty and
                    nrot == self.trot and nscale == self.tscale and
                    ncolor == self.tcolor):
                    # Pad with NOPs
                    while len(new_instrs) < max_depth:
                        new_instrs.append(0)
                    return new_instrs

                queue.append((new_instrs, nx, ny, nrot, nscale, ncolor))

        return None

    def compute_clicks(self, target_instrs):
        """Compute click positions to set instruction values and hit confirm."""
        clicks = []
        for col_idx, target_val in enumerate(target_instrs):
            if col_idx >= len(self.columns):
                break
            current_val = self.current_instrs[col_idx]
            diff = current_val ^ target_val

            for bit_idx, bit in enumerate(self.columns[col_idx]):
                if diff & (1 << bit_idx):
                    clicks.append((bit['x'], bit['y']))

        if self.confirm_pos:
            clicks.append(self.confirm_pos)

        return clicks


def solve_all_levels(verbose=True):
    """Solve all 7 tn36 levels."""
    arcade = Arcade()
    env = arcade.make(GAME_ID)
    fd = env.reset()
    game = env._game

    all_clicks = []
    all_level_clicks = []
    total = 0

    for level_idx in range(7):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Level {level_idx + 1} / 7")

        ctrl = game.tsflfunycx
        model = TN36WorldModel(game)

        if verbose:
            print(f"  Block: ({model.bx},{model.by}) rot={model.brot} scale={model.bscale} color={model.bcolor}")
            print(f"  Target: ({model.tx},{model.ty}) rot={model.trot} scale={model.tscale} color={model.tcolor}")
            print(f"  Cols: {model.n_cols}, Walls: {len(model.walls)}, Portals: {len(model.portals)}")
            print(f"  Waypoints: {len(model.waypoints)}, Tabs: {len(model.tabs)}")
            print(f"  Current instrs: {model.current_instrs}")

        # Check if this is a multi-play level (has waypoints)
        if model.waypoints:
            if verbose:
                print(f"  Multi-play level with {len(model.waypoints)} waypoints")

            # Plan path: start -> wp0 -> wp1 -> ... -> target
            # At each waypoint, block docks and home updates
            path = [(wp[0], wp[1]) for wp in model.waypoints] + [(model.tx, model.ty)]

            if verbose:
                print(f"  Path: ({model.bx},{model.by}) -> " + " -> ".join(f"({x},{y})" for x, y in path))

            # For each leg, find instructions and execute
            multiplay_clicks = []
            current_home = (model.bx, model.by)
            current_rot = model.brot
            current_scale = model.bscale
            current_color = model.bcolor

            for leg_idx, (gx, gy) in enumerate(path):
                is_final = (leg_idx == len(path) - 1)
                goal_rot = model.trot if is_final else current_rot
                goal_scale = model.tscale if is_final else current_scale
                goal_color = model.tcolor if is_final else current_color

                if verbose:
                    print(f"  Leg {leg_idx+1}: ({current_home[0]},{current_home[1]}) -> ({gx},{gy})")

                # Build a fresh model for this leg
                leg_model = TN36WorldModel.__new__(TN36WorldModel)
                leg_model.bx, leg_model.by = current_home
                leg_model.brot = current_rot
                leg_model.bscale = current_scale
                leg_model.bcolor = current_color
                leg_model.bw = model.bw
                leg_model.bh = model.bh
                leg_model.tx, leg_model.ty = gx, gy
                leg_model.trot = goal_rot
                leg_model.tscale = goal_scale
                leg_model.tcolor = goal_color
                leg_model.walls = model.walls
                leg_model.portals = model.portals
                leg_model.waypoints = model.waypoints
                leg_model.n_cols = model.n_cols
                leg_model.current_instrs = model.current_instrs  # Will be updated
                leg_model.columns = model.columns
                leg_model.confirm_pos = model.confirm_pos
                leg_model.tabs = model.tabs
                leg_model.programs = model.programs

                instrs = leg_model.find_instructions_to(gx, gy, goal_rot, goal_scale, goal_color)

                if instrs is None:
                    if verbose:
                        print(f"    Failed to find instructions for leg {leg_idx+1}")
                    break

                if verbose:
                    print(f"    Instructions: {instrs}")

                # Compute clicks for this leg
                clicks = model.compute_clicks(instrs)
                multiplay_clicks.extend(clicks)

                # Execute clicks on live env
                for x, y in clicks:
                    fd = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
                    for _ in range(500):
                        if not ctrl.nwjrtjcxpo: break
                        fd = env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})

                if fd.levels_completed > level_idx:
                    break

                # Update model state for next leg
                # After confirm + animation, the block resets to home (which may be updated)
                # Read new state from game
                ctrl = game.tsflfunycx
                right = ctrl.xsseeglmfh
                current_home = (right.kudckufewr, right.rhoxghutkw)
                current_rot = right.lixqebpwoq
                current_scale = right.wedqibqkhx
                current_color = right.cydzcfhdzi
                model.current_instrs = right.fcrcpemsnv.ylczjoyapu
                # Re-read column positions (they may shift with tab changes)
                model.columns = []
                for col in right.fcrcpemsnv.thofkgziyd:
                    bits = []
                    for bit in col.puakvdstpr:
                        bits.append({
                            'x': bit.x + bit.width // 2,
                            'y': bit.y + bit.height // 2,
                            'on': bit.hokejgzome,
                        })
                    model.columns.append(bits)

                if verbose:
                    print(f"    New home: ({current_home[0]},{current_home[1]})")

            if fd.levels_completed > level_idx:
                if verbose:
                    print(f"  SOLVED L{level_idx+1} in {len(multiplay_clicks)} actions")
                all_clicks.extend(multiplay_clicks)
                all_level_clicks.append(multiplay_clicks)
                total += len(multiplay_clicks)
                continue
            else:
                if verbose:
                    print(f"  Multi-play failed for L{level_idx+1}")

        # For simple single-play levels, find instructions via BFS
        instrs = model.find_instructions_bfs()

        if instrs is not None:
            if verbose:
                print(f"  Found instructions: {instrs}")

            clicks = model.compute_clicks(instrs)
            if verbose:
                print(f"  Clicks: {len(clicks)}")

            # Execute clicks
            for x, y in clicks:
                fd = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
                for _ in range(500):
                    if not ctrl.nwjrtjcxpo:
                        break
                    fd = env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})

            if fd.levels_completed > level_idx:
                if verbose:
                    print(f"  SOLVED L{level_idx+1} in {len(clicks)} actions")
                all_clicks.extend(clicks)
                all_level_clicks.append(clicks)
                total += len(clicks)
                continue
            else:
                if verbose:
                    print(f"  Instructions found but level didn't advance.")
                    print(f"  Win check: {ctrl.yxabhsirzl}")
                    right = ctrl.xsseeglmfh
                    print(f"  Block final: ({right.ravxreuqho.x},{right.ravxreuqho.y})")
                    print(f"  Target: ({right.ddzsdagbti.x},{right.ddzsdagbti.y})")

                # Verify with engine simulation
                verified = model.simulate_instructions(instrs)
                if verbose:
                    print(f"  Simulated: pos=({verified[0]},{verified[1]}) rot={verified[2]} scale={verified[3]} color={verified[4]} alive={verified[5]}")

                # The world model might be inaccurate for portal timing.
                # Fall through to engine-based verification approach.

        # If BFS failed or result didn't match, try engine-based brute force
        if verbose:
            print(f"  Trying engine-verified BFS...")

        # For engine-based BFS we need to replay from scratch for each candidate
        # This is slow but correct for portal/wall edge cases

        # Generate candidates from the world model BFS but verify with engine
        # Actually let's try a different approach: just use the engine directly
        # for levels that fail the model-based approach

        # Reconstruct env
        env2 = arcade.make(GAME_ID)
        fd2 = env2.reset()
        game2 = env2._game
        ctrl2 = game2.tsflfunycx

        for x, y in all_clicks:
            fd2 = env2.step(GameAction.ACTION6, data={'x': x, 'y': y})
            for _ in range(500):
                if not ctrl2.nwjrtjcxpo: break
                fd2 = env2.step(GameAction.ACTION6, data={'x': 0, 'y': 0})

        # Now at the right level with clean state
        right2 = ctrl2.xsseeglmfh
        model2 = TN36WorldModel(game2)

        # Try all promising instruction sequences with engine verification
        dx = model2.tx - model2.bx
        dy = model2.ty - model2.by

        found = False
        # Generate candidate instruction sequences more aggressively
        move_cmds = []
        if dx > 0: move_cmds.extend([2, 10])
        if dx < 0: move_cmds.extend([1, 12])
        if dy > 0: move_cmds.append(3)
        if dy < 0: move_cmds.append(33)
        move_cmds.append(0)

        # For manageable search: enumerate sequences up to n_cols
        from itertools import product as iprod
        n = model2.n_cols

        # Limit search space by filtering to reasonable combinations
        tested = 0
        for seq in iprod(move_cmds, repeat=n):
            seq = list(seq)
            tested += 1
            if tested > 50000:
                break

            # Quick position check with model
            sim = model2.simulate_instructions(seq)
            if sim[0] != model2.tx or sim[1] != model2.ty:
                continue
            if sim[2] != model2.trot or sim[3] != model2.tscale:
                continue
            if sim[4] != model2.tcolor:
                continue
            if not sim[5]:
                continue

            # Candidate found — verify with engine
            env3 = arcade.make(GAME_ID)
            fd3 = env3.reset()
            game3 = env3._game
            ctrl3 = game3.tsflfunycx

            for x, y in all_clicks:
                fd3 = env3.step(GameAction.ACTION6, data={'x': x, 'y': y})
                for _ in range(500):
                    if not ctrl3.nwjrtjcxpo: break
                    fd3 = env3.step(GameAction.ACTION6, data={'x': 0, 'y': 0})

            right3 = ctrl3.xsseeglmfh
            model3 = TN36WorldModel(game3)
            clicks = model3.compute_clicks(seq)

            for x, y in clicks:
                fd3 = env3.step(GameAction.ACTION6, data={'x': x, 'y': y})
                for _ in range(500):
                    if not ctrl3.nwjrtjcxpo: break
                    fd3 = env3.step(GameAction.ACTION6, data={'x': 0, 'y': 0})

            if fd3.levels_completed > level_idx:
                if verbose:
                    print(f"  Engine-verified solution: {seq} ({tested} tried)")
                    print(f"  SOLVED L{level_idx+1} in {len(clicks)} actions")

                # Apply to live env
                env = arcade.make(GAME_ID)
                fd = env.reset()
                game = env._game
                ctrl = game.tsflfunycx

                for x, y in all_clicks:
                    fd = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
                    for _ in range(500):
                        if not ctrl.nwjrtjcxpo: break
                        fd = env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})

                right = ctrl.xsseeglmfh
                model_final = TN36WorldModel(game)
                real_clicks = model_final.compute_clicks(seq)

                for x, y in real_clicks:
                    fd = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
                    for _ in range(500):
                        if not ctrl.nwjrtjcxpo: break
                        fd = env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})

                all_clicks.extend(real_clicks)
                all_level_clicks.append(real_clicks)
                total += len(real_clicks)
                found = True
                break

        if not found:
            if verbose:
                print(f"  FAILED L{level_idx+1} after {tested} engine tests")
            all_level_clicks.append([])
            break

    return all_level_clicks, all_clicks, total, fd, env, game


def replay_and_capture(arcade, all_clicks, verbose=True):
    """Replay with visual capture."""
    try:
        from arc_session_writer import SessionWriter
    except ImportError:
        SessionWriter = None

    env = arcade.make(GAME_ID)
    fd = env.reset()
    game = env._game
    grid = get_frame(fd)
    ctrl = game.tsflfunycx

    writer = None
    if SessionWriter:
        writer = SessionWriter(
            game_id=GAME_ID,
            win_levels=fd.win_levels,
            available_actions=[int(a) for a in fd.available_actions],
            baseline=250,
            player="tn36_world_model_solver"
        )
        writer.record_step(0, grid, levels_completed=0, state="PLAYING")

    total = 0
    for x, y in all_clicks:
        fd = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
        total += 1
        grid = get_frame(fd)

        anim_frames = [grid]
        for _ in range(500):
            if not ctrl.nwjrtjcxpo: break
            fd = env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})
            anim_frames.append(get_frame(fd))

        if writer:
            writer.record_step(6, grid,
                all_frames=anim_frames if len(anim_frames) > 1 else None,
                levels_completed=fd.levels_completed, x=x, y=y,
                state=fd.state.name)
        if fd.state.name == 'GAME_OVER':
            break

    if writer:
        writer.record_game_end(state=fd.state.name, levels_completed=fd.levels_completed)
    return fd, total


def main():
    import time as _time
    print("=" * 60)
    print("tn36 World-Model Solver")
    print("=" * 60)

    t0 = _time.monotonic()
    all_level_clicks, all_clicks, total, fd, env, game = solve_all_levels(verbose=True)
    t1 = _time.monotonic()

    print(f"\n{'='*60}")
    print(f"Result: {fd.levels_completed}/7 levels, {total} actions (baseline 250)")
    print(f"Time: {t1-t0:.1f}s")

    if fd.levels_completed >= 5:
        print(f"\nReplaying with visual capture...")
        arcade = Arcade()
        fd2, total2 = replay_and_capture(arcade, all_clicks, verbose=True)
        print(f"Replay: {fd2.levels_completed}/7, {total2} actions captured")

    sol_data = {
        "levels_solved": fd.levels_completed,
        "total_actions": total,
        "all_clicks": all_clicks,
    }
    for li, lc in enumerate(all_level_clicks):
        sol_data[f"L{li+1}"] = {"actions": len(lc), "clicks": lc}

    out_path = os.path.join(os.path.dirname(__file__), "tn36_action_sequence.json")
    with open(out_path, "w") as f:
        json.dump(sol_data, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
