#!/usr/bin/env python3
"""
lf52 L10 winning replay (WIP).

Hint (Yarflam, Discord): "L10 makes the levels the fastest to solve.
You just need to bring a blue ball and an empty cart."

Priors from prior investigations:
  - 2 N, 10 B pieces. Win = reduce N from 2 to 1.
  - N@(4,0) and N@(6,9). Green+ goal is at (6,9)-area (based on image).
  - Phase 1 (verified): 8x ACTION1 moves x=8 column blocks up; topmost reaches (8,1).
  - Phase 2 (verified): 5x ACTION3 traverses topmost "7" (blue+block+wall) through row-1 channel.

This script is built iteratively. Run it to see live grid state; it
writes frames and incrementally appends to run.json.
"""
import os, sys, json, time
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.setrecursionlimit(50000)
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, GameState
import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

OUT_DIR = "/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/lf52/run_L10_winning"
FRAMES_DIR = f"{OUT_DIR}/frames"
TRACE_PATH = f"{OUT_DIR}/run.json"
os.makedirs(FRAMES_DIR, exist_ok=True)

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}
DIR_ACTIONS = {'UP': GameAction.ACTION1, 'DOWN': GameAction.ACTION2,
               'LEFT': GameAction.ACTION3, 'RIGHT': GameAction.ACTION4}

# ------- tracing wrapper -------
class TracingEnv:
    def __init__(self, env):
        self.env = env
        self._game = env._game
        self.trace = []
        self.step_n = 0

    @property
    def fd(self):
        return self._last_fd

    def reset(self):
        self._last_fd = self.env.reset()
        return self._last_fd

    def step(self, action, data=None):
        pre = snapshot_state(self._game)
        if data is None:
            self._last_fd = self.env.step(action)
        else:
            self._last_fd = self.env.step(action, data=data)
        self.step_n += 1
        post = snapshot_state(self._game)
        action_name = getattr(action, 'name', str(action))
        note = describe_delta(pre, post)
        entry = {
            "step": self.step_n,
            "action": action_name,
            "data": data,
            "pre_state": pre,
            "post_state": post,
            "game_state": str(self._last_fd.state),
            "levels_completed": self._last_fd.levels_completed,
            "note": note,
        }
        self.trace.append(entry)
        return self._last_fd

# ------- helpers -------
def eq(game):
    return game.ikhhdzfmarl

def snapshot_state(game):
    e = eq(game)
    grid = e.hncnfaqaddg
    N, R, B = [], [], []
    blocks = set()
    for y in range(15):
        for x in range(10):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            for n in names:
                if n == 'fozwvlovdui':
                    N.append([x, y])
                elif n == 'fozwvlovdui_red':
                    R.append([x, y])
                elif n == 'fozwvlovdui_blue':
                    B.append([x, y])
            if any(n == 'hupkpseyuim2' for n in names):
                blocks.add((x, y))
    sel = None
    try:
        s = e.eqkdklybpg
        if s is not None:
            sel = list(s)
    except Exception:
        pass
    return {
        'N': N, 'R': R, 'B': B,
        'blocks': sorted(blocks),
        'selected': sel,
        'step': e.asqvqzpfdi,
    }

def describe_delta(pre, post):
    parts = []
    if pre['N'] != post['N']:
        parts.append(f"N {pre['N']}->{post['N']}")
    if pre['R'] != post['R']:
        parts.append(f"R {pre['R']}->{post['R']}")
    if pre['B'] != post['B']:
        parts.append(f"B {len(pre['B'])}->{len(post['B'])}")
    if pre['blocks'] != post['blocks']:
        parts.append(f"blocks moved")
    if pre['selected'] != post['selected']:
        parts.append(f"sel {pre['selected']}->{post['selected']}")
    return "; ".join(parts) or "no-change"

def save_trace(tracer, final_note=""):
    data = {
        "game": "lf52-271a04aa",
        "level": 10,
        "method": "game.set_level(9) + refresh step",
        "rules": ["no eq.win()", "only env.step()"],
        "total_steps": tracer.step_n,
        "state": str(tracer._last_fd.state) if hasattr(tracer, '_last_fd') else "unknown",
        "levels_completed": tracer._last_fd.levels_completed if hasattr(tracer, '_last_fd') else 0,
        "final_note": final_note,
        "trace": tracer.trace,
    }
    with open(TRACE_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)

def save_frame(tracer, label):
    if not _HAS_PIL:
        return
    try:
        fd = tracer._last_fd
        arr = fd.frame
        # frame may be nested list; coerce to ndarray
        arr = np.asarray(arr)
        # squeeze leading singleton dims (sometimes (1,H,W) or (1,H,W,3))
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.dtype != np.uint8:
            # frame values are palette indices 0-15 — scale to 0-255 for viewing
            if arr.max() < 16:
                arr = (arr * 16).clip(0, 255)
            arr = arr.astype(np.uint8)
        img = Image.fromarray(arr)
        # upscale 4x for readability
        w, h = img.size
        img = img.resize((w*4, h*4), Image.NEAREST)
        path = f"{FRAMES_DIR}/{tracer.step_n:03d}_{label}.png"
        img.save(path)
    except Exception as e:
        print(f"save_frame error: {e}")

def print_grid(game, rows=15, cols=10):
    e = eq(game)
    grid = e.hncnfaqaddg
    print("   " + "".join(str(x) for x in range(cols)))
    for y in range(rows):
        s = f"{y:2d} "
        for x in range(cols):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            ch = ' '
            has_N = any(n == 'fozwvlovdui' for n in names)
            has_R = any(n == 'fozwvlovdui_red' for n in names)
            has_Bl = any(n == 'fozwvlovdui_blue' for n in names)
            has_block = any(n == 'hupkpseyuim2' for n in names)
            has_wall = any('kraubslpehi' in n for n in names)
            has_peg = any(n == 'dgxfozncuiz' for n in names)
            has_floor = 'hupkpseyuim' in names
            has_goal = any('green' in n.lower() or 'goal' in n.lower() for n in names)
            if has_N and has_block: ch = '@'
            elif has_N: ch = 'N'
            elif has_R and has_block: ch = 'r'
            elif has_R: ch = 'R'
            elif has_Bl and has_block and has_wall: ch = '7'  # "cart with blue"
            elif has_Bl and has_block: ch = 'b'
            elif has_Bl: ch = 'B'
            elif has_block and has_wall: ch = ';'  # cart on wall
            elif has_block: ch = 'o'
            elif has_goal: ch = 'G'
            elif has_wall: ch = '#'
            elif has_peg: ch = '.'
            elif has_floor: ch = '-'
            s += ch
        print(s)

def list_obj_names(game):
    """List all unique object names on the grid, for discovering goal tile."""
    e = eq(game)
    grid = e.hncnfaqaddg
    names = {}
    for y in range(15):
        for x in range(10):
            objs = grid.ijpoqzvnjt(x, y)
            for o in objs:
                names.setdefault(o.name, []).append((x, y))
    return names

def click_cell(tracer, game, wx, wy, dx_half=0, dy_half=0, label=""):
    """Click world-cell (wx,wy). Arrow offset: dx_half, dy_half in half-cells (±1 = 12 px)."""
    e = eq(game)
    off = e.hncnfaqaddg.cdpcbbnfdp
    vp_x = wx * 6 + off[0] + 3 + dx_half * 12
    vp_y = wy * 6 + off[1] + 3 + dy_half * 12
    return tracer.step(GameAction.ACTION6, data={'x': vp_x, 'y': vp_y})

def check_all_jumps(game):
    e = eq(game)
    valid = []
    for y in range(15):
        for x in range(10):
            for d in DIRS:
                if e.qikmikecdf((x, y), d):
                    dx, dy = d
                    valid.append(((x, y), (x+2*dx, y+2*dy), DIR_NAMES[d]))
    return valid

# ------- main -------
def main():
    arc = Arcade(operation_mode='offline')
    env = arc.make('lf52-271a04aa')
    env.reset()
    game = env._game
    game.set_level(9)  # L10 (0-indexed)
    # refresh frame (sentinel step to populate fd)
    env.step(GameAction.ACTION1)  # NOTE: this is *already* a push UP
    # But we actually want to *start* from pristine L10. Better: use a fresh wrapper
    # and re-set. The instructions explicitly say: reset + set_level(9) + refresh step.

    # So reset again:
    env.reset()
    game = env._game
    game.set_level(9)
    fd = env.step(GameAction.ACTION1)  # refresh per instructions
    # That first step counts as a push UP action on the real game. We must include
    # it in the replay sequence.
    tracer = TracingEnv(env)
    tracer._last_fd = fd
    tracer.step_n = 1
    tracer.trace.append({
        "step": 1,
        "action": "ACTION1",
        "data": None,
        "pre_state": None,
        "post_state": snapshot_state(game),
        "game_state": str(fd.state),
        "levels_completed": fd.levels_completed,
        "note": "initial refresh step (ACTION1=UP push 1/8)",
    })

    print("=== Initial state after first push ===")
    print_grid(game)
    print()
    s = snapshot_state(game)
    print(f"N: {s['N']}")
    print(f"R: {s['R']}")
    print(f"B: {len(s['B'])} blues: {s['B']}")
    print(f"blocks: {s['blocks']}")
    print(f"step counter: {s['step']}")

    print("\n=== Object names on grid ===")
    names = list_obj_names(game)
    for n, positions in sorted(names.items()):
        print(f"  {n}: {len(positions)} at {positions[:5]}{'...' if len(positions) > 5 else ''}")

    save_frame(tracer, "after_step1_up")
    save_trace(tracer, "initial state captured")

    # Phase 1 continued: UP x7 more (total 8 UP pushes to move topmost block to row 1)
    print("\n=== Phase 1: UP x7 more ===")
    for i in range(7):
        tracer.step(GameAction.ACTION1)
    save_frame(tracer, "after_phase1_up8")
    save_trace(tracer, "phase1 complete")
    s = snapshot_state(game)
    print(f"After UP x8: blocks={s['blocks']}")
    print_grid(game)

    # Phase 2: LEFT x5 to bring topmost block across row 1
    print("\n=== Phase 2: LEFT x5 ===")
    for i in range(5):
        tracer.step(GameAction.ACTION3)
    save_frame(tracer, "after_phase2_left5")
    save_trace(tracer, "phase2 complete")
    s = snapshot_state(game)
    print(f"After LEFT x5: blocks={s['blocks']}")
    print_grid(game)

    # Check what jumps are now available
    print("\n=== Valid jumps after phase 2 ===")
    for src, dst, name in check_all_jumps(game):
        print(f"  {src} {name} -> {dst}")

    # Save final state
    save_trace(tracer, f"state={fd.state} after phases 1+2")
    print(f"\nDone. State: {tracer._last_fd.state}, levels_completed: {tracer._last_fd.levels_completed}")

if __name__ == "__main__":
    main()
