#!/usr/bin/env python3
"""
vc33 World-Model Solver — Seesaw-balance puzzle.

Builds a lightweight world model from the game engine's initial state,
then uses BFS over trigger clicks to find optimal solutions.

Key insight: gel() always modifies pixels.shape[0] (row count). Whether that
maps to width or height depends on sprite rotation. All sprites in a level
share the same rotation.

Rotation mapping:
- 0/180: width = shape[1], height = shape[0]  -> gel changes height
- 90/270: width = shape[0], height = shape[1] -> gel changes width
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

GAME_ID = 'vc33-9851e02b'


def get_frame(fd):
    grid = np.array(fd.frame)
    return grid[-1] if grid.ndim == 3 else grid


class Obj:
    """Generic game object with position and dimensions."""
    __slots__ = ['x', 'y', 'w', 'h', 'prows', 'pcols', 'idx', 'color']
    def __init__(self, x, y, w, h, prows, pcols, idx=0, color=-1):
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.prows, self.pcols = prows, pcols
        self.idx = idx
        self.color = color
    def copy(self):
        return Obj(self.x, self.y, self.w, self.h, self.prows, self.pcols, self.idx, self.color)

def obj_from_sprite(s, idx=0, color=-1):
    return Obj(s.x, s.y, s.width, s.height, s.pixels.shape[0], s.pixels.shape[1], idx, color)


class VC33Model:
    """Lightweight vc33 world model for BFS solving."""

    def __init__(self):
        self.beams = []
        self.hqbs = []
        self.fzks = []
        self.pivots = []
        self.walls = []
        self.zgd = []       # (beam1_idx, beam2_idx) per trigger
        self.zgd_pos = []   # (x, y) per trigger
        self.zhk = []       # Obj per trigger
        self.oro = [0, 0]
        self.energy = 0
        self.cam_w = 64
        self.cam_h = 64
        self.rot90 = False  # True if rotation is 90/270 (prows maps to width)

    @staticmethod
    def from_game(game):
        m = VC33Model()
        level = game.current_level
        m.oro = list(game.oro)
        m.energy = game.vrr.olv
        m.cam_w = game.camera.width
        m.cam_h = game.camera.height

        # Determine rotation mode
        beams_s = level.get_sprites_by_tag("rDn")
        if beams_s:
            rot = beams_s[0].rotation
            m.rot90 = (rot in (90, 270))

        # Beams
        beam_sprite_map = {}
        for i, s in enumerate(beams_s):
            b = obj_from_sprite(s, idx=i)
            m.beams.append(b)
            beam_sprite_map[id(s)] = i

        # HQBs
        for i, s in enumerate(level.get_sprites_by_tag("HQB")):
            color = int(s.pixels[-1, -1])
            m.hqbs.append(obj_from_sprite(s, idx=i, color=color))

        # fZKs
        for i, s in enumerate(level.get_sprites_by_tag("fZK")):
            color = int(s.pixels[-1, -1]) if s.pixels.size > 0 else -1
            m.fzks.append(obj_from_sprite(s, idx=i, color=color))

        # Pivots
        for s in level.get_sprites_by_tag("UXg"):
            m.pivots.append(obj_from_sprite(s))

        # Walls
        for s in level.get_sprites_by_tag("WuO"):
            m.walls.append(obj_from_sprite(s))

        # ZGd triggers
        zgd_s = level.get_sprites_by_tag("ZGd")
        for s in zgd_s:
            if s in game.dzy:
                b1_s, b2_s = game.dzy[s]
                bi1 = beam_sprite_map.get(id(b1_s), -1)
                bi2 = beam_sprite_map.get(id(b2_s), -1)
                m.zgd.append((bi1, bi2))
                m.zgd_pos.append((s.x, s.y))

        # zHk triggers
        for s in level.get_sprites_by_tag("zHk"):
            m.zhk.append(obj_from_sprite(s))

        return m

    def copy(self):
        m = VC33Model()
        m.beams = [b.copy() for b in self.beams]
        m.hqbs = [h.copy() for h in self.hqbs]
        m.fzks = [f.copy() for f in self.fzks]
        m.pivots = [p.copy() for p in self.pivots]
        m.walls = [w.copy() for w in self.walls]
        m.zgd = list(self.zgd)
        m.zgd_pos = list(self.zgd_pos)
        m.zhk = [t.copy() for t in self.zhk]
        m.oro = list(self.oro)
        m.energy = self.energy
        m.cam_w = self.cam_w
        m.cam_h = self.cam_h
        m.rot90 = self.rot90
        return m

    # Coordinate helpers (match game engine)
    def _hor(self):
        return self.oro[0] != 0

    def _lia(self):
        return self.oro[0] > 0 or self.oro[1] > 0

    def _urh(self, o):
        return o.y if self._hor() else o.x

    def _ebl(self, o):
        return o.x if self._hor() else o.y

    def _brj(self, o):
        return o.h if self._hor() else o.w

    def _jqo(self, o):
        return o.w if self._hor() else o.h

    def _lho(self, o):
        return self._ebl(o) + self._jqo(o) if self._lia() else self._ebl(o)

    def _utq(self, o):
        return self._ebl(o) if self._lia() else self._ebl(o) + self._jqo(o)

    def _gdu(self, hqb, beam):
        return (self._urh(hqb) >= self._urh(beam) and
                self._urh(hqb) < self._urh(beam) + self._brj(beam) and
                self._lho(hqb) == self._utq(beam))

    def _pth(self, beam):
        return [h for h in self.hqbs if self._gdu(h, beam)]

    def _suo(self, beam):
        return [p for p in self.pivots
                if self._urh(p) + self._brj(p) == self._urh(beam) or
                   self._urh(p) == self._urh(beam) + self._brj(beam)]

    def _cdr(self, beam):
        if self._lia():
            return [w for w in self.walls
                    if self._urh(w) == self._urh(beam) and self._lho(w) < self._lho(beam)]
        else:
            return [w for w in self.walls
                    if self._urh(w) == self._urh(beam) and self._lho(w) > self._lho(beam)]

    def _mud(self, beam):
        walls = self._cdr(beam)
        if walls:
            hqbs = self._pth(beam)
            if hqbs:
                if self.oro[0] == -3:
                    return max(self._lho(w) - 6 for w in walls)
                else:
                    return max(self._lho(w) - 4 for w in walls)
            else:
                return max(self._lho(w) for w in walls)

        pivots = self._suo(beam)
        if not pivots:
            return self._ebl(beam)  # fallback
        if self._lia():
            return max(self._utq(p) for p in pivots)
        else:
            return min(self._utq(p) for p in pivots)

    def _resize_beam(self, beam, delta_prows):
        """Resize beam by changing pixel row count.
        delta_prows < 0 means shrink, > 0 means grow.
        With rot90=True, prows maps to width. Otherwise to height."""
        beam.prows += delta_prows
        if self.rot90:
            beam.w += delta_prows
        else:
            beam.h += delta_prows

    def gel(self, trigger_idx):
        """Execute ZGd beam transfer."""
        bi1, bi2 = self.zgd[trigger_idx]
        pmj = self.beams[bi1]  # source (shrinks)
        chd = self.beams[bi2]  # target (grows)

        if self._jqo(pmj) <= 0:
            return False

        can_grow = ((self._lia() and self._ebl(chd) > self._mud(chd)) or
                    (not self._lia() and self._utq(chd) < self._mud(chd)))
        if not can_grow:
            return False

        rsi, qir = self.oro

        # Move HQBs on source (same direction as transfer)
        for h in self._pth(pmj):
            h.x += rsi
            h.y += qir

        # Move HQBs on target (opposite direction)
        for h in self._pth(chd):
            h.x -= rsi
            h.y -= qir

        # Move beam positions
        if rsi >= 0:
            pmj.x += rsi
            chd.x -= rsi
        if qir >= 0:
            pmj.y += qir
            chd.y -= qir

        # Resize beams (always modifies pixel rows)
        step = abs(rsi) if rsi != 0 else abs(qir)
        self._resize_beam(pmj, -step)
        self._resize_beam(chd, step)

        self.energy -= 1
        return True

    def krt(self, zhk_idx):
        """Check if zHk rotation trigger is active."""
        t = self.zhk[zhk_idx]
        flanking = [b for b in self.beams if self._utq(b) == self._lho(t)]
        left = [b for b in flanking if self._urh(b) + self._brj(b) == self._urh(t)]
        right = [b for b in flanking if self._urh(b) == self._urh(t) + self._brj(t)]
        return bool(left) and bool(right)

    def teu(self, zhk_idx):
        """Execute zHk rotation — swap HQBs between adjacent beams."""
        t = self.zhk[zhk_idx]
        flanking = [b for b in self.beams if self._utq(b) == self._lho(t)]
        left_beams = [b for b in flanking if self._urh(b) + self._brj(b) == self._urh(t)]
        right_beams = [b for b in flanking if self._urh(b) == self._urh(t) + self._brj(t)]

        if not left_beams or not right_beams:
            return False

        lb = left_beams[0]
        rb = right_beams[0]

        left_hqbs = self._pth(lb)
        right_hqbs = self._pth(rb)

        # Swap positions
        for h in left_hqbs:
            if self._hor():
                h.y = rb.y + rb.h // 2 - h.h // 2
            else:
                h.x = rb.x + rb.w // 2 - h.w // 2

        for h in right_hqbs:
            if self._hor():
                h.y = lb.y + lb.h // 2 - h.h // 2
            else:
                h.x = lb.x + lb.w // 2 - h.w // 2

        self.energy -= 1
        return True

    def gug(self):
        """Check win condition."""
        for hqb in self.hqbs:
            beam = None
            for b in self.beams:
                if self._gdu(hqb, b):
                    beam = b
                    break
            if beam is None:
                return False

            adj_pivots = self._suo(beam)
            matched = False
            for fzk in self.fzks:
                if fzk.color != hqb.color:
                    continue
                if self._ebl(hqb) != self._ebl(fzk):
                    continue
                # Check fZK is on an adjacent pivot
                for p in adj_pivots:
                    if (fzk.x < p.x + p.w and fzk.x + fzk.w > p.x and
                        fzk.y < p.y + p.h and fzk.y + fzk.h > p.y):
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                return False
        return True

    def state_key(self):
        parts = []
        for h in sorted(self.hqbs, key=lambda h: h.idx):
            parts.append(h.x)
            parts.append(h.y)
        for b in sorted(self.beams, key=lambda b: b.idx):
            parts.append(b.x)
            parts.append(b.y)
            parts.append(b.w)
            parts.append(b.h)
        return tuple(parts)

    def get_actions(self):
        actions = []
        for i in range(len(self.zgd)):
            actions.append(('zgd', i))
        for i in range(len(self.zhk)):
            if self.krt(i):
                actions.append(('zhk', i))
        return actions

    def apply_action(self, action):
        atype, aidx = action
        if atype == 'zgd':
            return self.gel(aidx)
        elif atype == 'zhk':
            return self.teu(aidx)
        return False

    def action_display_coords(self, action):
        atype, aidx = action
        if atype == 'zgd':
            gx, gy = self.zgd_pos[aidx]
        else:
            t = self.zhk[aidx]
            gx, gy = t.x, t.y
        scale = 64 // self.cam_w
        ox = (64 - self.cam_w * scale) // 2
        oy = (64 - self.cam_h * scale) // 2
        return gx * scale + ox + 1, gy * scale + oy + 1


def bfs_solve(model, max_depth=60, verbose=False):
    """BFS over world model to find shortest click sequence."""
    if model.gug():
        return []

    queue = deque()
    queue.append(([], model))
    visited = {model.state_key()}
    nodes = 0

    while queue:
        actions, m = queue.popleft()
        nodes += 1

        if len(actions) >= max_depth:
            continue

        if nodes % 50000 == 0 and verbose:
            print(f"    BFS: {nodes} nodes, depth {len(actions)}, queue {len(queue)}, visited {len(visited)}")

        for action in m.get_actions():
            m2 = m.copy()
            if not m2.apply_action(action):
                continue
            if m2.energy <= 0:
                continue

            if m2.gug():
                if verbose:
                    print(f"    BFS solved: depth {len(actions)+1}, {nodes} nodes explored")
                return actions + [action]

            key = m2.state_key()
            if key in visited:
                continue
            visited.add(key)
            queue.append((actions + [action], m2))

    if verbose:
        print(f"    BFS exhausted: {nodes} nodes, {len(visited)} states, max_depth={max_depth}")
    return None


def solve_all_levels(verbose=True):
    """Solve all 7 vc33 levels."""
    arcade = Arcade()
    known_counts = [3, 7, 23, 21, 44, 20, 49]

    env = arcade.make(GAME_ID)
    fd = env.reset()
    game = env._game

    all_solutions = []
    total = 0

    for level_idx in range(7):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Level {level_idx + 1} / 7 (expected ~{known_counts[level_idx]} actions)")

        model = VC33Model.from_game(game)

        if verbose:
            print(f"  Beams: {len(model.beams)}, HQBs: {len(model.hqbs)}, fZKs: {len(model.fzks)}")
            print(f"  ZGd: {len(model.zgd)}, zHk: {len(model.zhk)}, Energy: {model.energy}")
            print(f"  Direction: {model.oro}, rot90: {model.rot90}")

        max_d = known_counts[level_idx] + 10
        solution = bfs_solve(model, max_depth=max_d, verbose=verbose)

        if solution is not None:
            n = len(solution)
            if verbose:
                print(f"  SOLVED L{level_idx+1} in {n} actions")
            all_solutions.append(solution)
            total += n

            # Apply to live engine to advance
            for action in solution:
                dx, dy = model.action_display_coords(action)
                fd = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
                for _ in range(200):
                    if game.vai is None:
                        break
                    fd = env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})

            if verbose:
                print(f"  Engine levels_completed: {fd.levels_completed}")
        else:
            if verbose:
                print(f"  FAILED to solve L{level_idx+1}")
            break

    return all_solutions, total, fd


def replay_and_capture(arcade, all_solutions, verbose=True):
    """Replay all solutions through SDK with visual capture."""
    try:
        from arc_session_writer import SessionWriter
    except ImportError:
        SessionWriter = None

    env = arcade.make(GAME_ID)
    fd = env.reset()
    game = env._game
    grid = get_frame(fd)

    writer = None
    if SessionWriter:
        writer = SessionWriter(
            game_id=GAME_ID,
            win_levels=fd.win_levels,
            available_actions=[int(a) for a in fd.available_actions],
            baseline=307,
            player="vc33_world_model_solver"
        )
        writer.record_step(0, grid, levels_completed=0, state="PLAYING")

    total_actions = 0

    for level_idx, level_solution in enumerate(all_solutions):
        # Build model to get display coords
        model = VC33Model.from_game(game)

        if verbose:
            print(f"  Replaying L{level_idx + 1}: {len(level_solution)} actions")

        for action in level_solution:
            dx, dy = model.action_display_coords(action)
            fd = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
            total_actions += 1
            grid = get_frame(fd)

            # Drain animation
            anim_frames = [grid]
            for _ in range(200):
                if game.vai is None:
                    break
                fd = env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})
                anim_frames.append(get_frame(fd))

            if writer:
                writer.record_step(
                    6, grid,
                    all_frames=anim_frames if len(anim_frames) > 1 else None,
                    levels_completed=fd.levels_completed,
                    x=dx, y=dy,
                    state=fd.state.name
                )

            if fd.levels_completed > level_idx:
                if writer:
                    writer.record_level_up(fd.levels_completed)
                break

    if writer:
        writer.record_game_end(
            state=fd.state.name,
            levels_completed=fd.levels_completed
        )

    return fd, total_actions


def main():
    import time as _time
    print("=" * 60)
    print("vc33 World-Model Solver")
    print("=" * 60)

    t0 = _time.monotonic()
    all_solutions, total, fd = solve_all_levels(verbose=True)
    t1 = _time.monotonic()

    print(f"\n{'='*60}")
    print(f"Result: {fd.levels_completed}/7 levels, {total} actions (baseline 307)")
    print(f"Time: {t1-t0:.1f}s")

    if fd.levels_completed == 7:
        print(f"\nReplaying with visual capture...")
        arcade = Arcade()
        fd2, total2 = replay_and_capture(arcade, all_solutions, verbose=True)
        print(f"Replay: {fd2.levels_completed}/7, {total2} actions captured")

    # Save solutions
    sol_data = {}
    for li, sol in enumerate(all_solutions):
        sol_data[f"L{li+1}"] = {
            "actions": len(sol),
            "sequence": [(a[0], a[1]) for a in sol],
        }
    sol_data["total_actions"] = total
    sol_data["levels_solved"] = fd.levels_completed

    out_path = os.path.join(os.path.dirname(__file__), "vc33_action_sequence.json")
    with open(out_path, "w") as f:
        json.dump(sol_data, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
