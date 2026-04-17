#!/usr/bin/env python3
"""
lf52 L10 winning replay (WIP — PARTIAL progress, not yet winning).

Mission (CBP, 2026-04-17): produce a winning trace for L10 per Discord hint
from Yarflam: "You just need to bring a blue ball and an empty cart."

STATE OF INVESTIGATION (after live-engine probe):
  Pristine L10 state:
    - 2 N: (4,0) and (6,9)
    - 10 blue: (5,5), (6,5), (3,6), (4,7), (2,8), (8,9-13)
    - 8 blocks (pushable carts): (2,2), (4,2), (6,2) empty; (8,9-13) with blues
    - Walls form a channel: row 1 is wall channel; x=0 and x=8 are vertical walls
    - Goal: reduce ddaguepwkt (count of fozwvlovdui pieces - initial blue count)
      from 2 to 1, which requires either an N-over-N jump (removes 1 N) or somehow
      removing a piece.

WIN MECHANICS (from engine source lf52.py lines 5374-5591):
  - cfilhtifcb (jump handler): ddaguepwkt = len(fozwvlovdui*) - lzoqlpcwzpu
  - Since blues can never be removed as middles (line 5393 skips "blue"),
    and jumper/middle same-name removes one non-blue (line 5396-5397),
    the ONLY way to win is N-over-N (removes an N).
  - lzoqlpcwzpu fixed at init to initial blue count (=10 for L10)

TOPOLOGY ANALYSIS:
  - N@(4,0) is trapped in row 0. To jump, needs orthogonal piece at (3,0),
    (4,1), or (5,0) and valid landing at corresponding (2,0), (4,2), or (6,0).
  - (2,0) and (6,0) are empty (0 obj) — INVALID landings.
  - (4,2) has only a wall — needs a block added for valid landing.
  - Only possible jump direction for N@(4,0): DOWN, requiring (4,1)=blue on
    cart AND (4,2)=empty cart (block+wall).
  - Row 2 starts with blocks at (2,2),(4,2),(6,2). Push UP clears (4,2)
    but fills (4,1). Bringing blue to (4,1) requires push sequence that
    also disturbs (4,2).
  - Multiple push orderings tried (see /tmp/lf52_l10_*.py) — cannot
    simultaneously achieve blue+block at (4,1) AND empty-cart at (4,2).
  - N@(6,9) is reachable but N@(4,0) and N@(6,9) cannot meet via jump chain
    (requires a piece bridge through the row-1 wall channel that pushes cannot build).

PRIOR ART:
  - Thor solved L1-L6, L8-L9 but marked L7 and L10 as structurally unsolvable
    in this engine version after 1.16M-state unified A* search.
  - CBP's prior investigation (shared-context/arc-agi-3/fleet-learning/cbp/
    lf52_blue_cart_investigation.md) verified the solver model matches the
    engine exactly — no model-engine discrepancy.
  - This agent's live-engine probe: confirms same findings.

PARTIAL TRACE (this script produces):
  Phase 1: 8x ACTION1 (UP) — blue+cart reaches (8,1)
  Phase 2: 5x ACTION3 (LEFT) — blue+cart traverses row-1 channel to (3,1)
  Phase 3: 1x ACTION4 (RIGHT) — blue+cart moves to (4,1), directly below N

  At this point:
    - (4,1) has block+wall+blue (valid jump middle)
    - (4,2) has ONLY wall (invalid landing — need block here)
    - N@(4,0) cannot execute the DOWN jump because landing is invalid.

  Attempts to resolve this (DOWN 1 first then UP etc.) have all failed —
  each configuration either puts blue and block in (4,2) together, or
  leaves (4,2) empty of block.

WHAT WOULD BE NEEDED FOR A WIN:
  A novel mechanic not yet discovered — e.g., a click sequence that removes
  a blue, or a specific state that triggers a code path other than jump-based
  count reduction. Neither is evident from engine source inspection.

HONEST CONCLUSION:
  Based on live engine walk + prior art, L10 in engine 271a04aa appears
  structurally unsolvable via env.step() alone. The Discord hint may apply
  to a different engine version than 271a04aa (the only version available
  locally).

This script produces the best partial trace (setup to "blue ball + empty cart
adjacent to N") and documents the blocker for future investigators.
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


class TracingEnv:
    """Wraps env to capture every step. No state mutation outside env.step()."""
    def __init__(self, env):
        self.env = env
        self._game = env._game
        self.trace = []
        self.step_n = 0
        self._last_fd = None

    def reset(self):
        self._last_fd = self.env.reset()
        return self._last_fd

    def step(self, action, data=None, note=""):
        pre = snapshot_state(self._game)
        if data is None:
            self._last_fd = self.env.step(action)
        else:
            self._last_fd = self.env.step(action, data=data)
        self.step_n += 1
        post = snapshot_state(self._game)
        action_name = getattr(action, 'name', str(action))
        delta = describe_delta(pre, post)
        entry = {
            "step": self.step_n,
            "action": action_name,
            "data": data,
            "pre_state": pre,
            "post_state": post,
            "game_state": str(self._last_fd.state),
            "levels_completed": self._last_fd.levels_completed,
            "delta": delta,
            "note": note,
        }
        self.trace.append(entry)
        return self._last_fd


def snapshot_state(game):
    e = game.ikhhdzfmarl
    grid = e.hncnfaqaddg
    N, B = [], []
    blocks = set()
    for y in range(15):
        for x in range(10):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            for n in names:
                if n == 'fozwvlovdui':
                    N.append([x, y])
                elif n == 'fozwvlovdui_blue':
                    B.append([x, y])
            if any(n == 'hupkpseyuim2' for n in names):
                blocks.add((x, y))
    return {
        'N': sorted(N),
        'B': sorted(B),
        'blocks': sorted(blocks),
        'step': e.asqvqzpfdi,
    }


def describe_delta(pre, post):
    parts = []
    if pre['N'] != post['N']:
        parts.append(f"N {pre['N']}->{post['N']}")
    if pre['B'] != post['B']:
        parts.append(f"B {len(pre['B'])}->{len(post['B'])}")
    if pre['blocks'] != post['blocks']:
        parts.append("blocks moved")
    return "; ".join(parts) or "no-change"


def save_trace(tracer, final_note=""):
    data = {
        "game": "lf52-271a04aa",
        "level": 10,
        "method": "arc.make('lf52-271a04aa') + env.reset() + game.set_level(9) + first env.step(ACTION1) is the refresh",
        "rules": ["no eq.win()", "only env.step()"],
        "total_steps": tracer.step_n,
        "state": str(tracer._last_fd.state) if tracer._last_fd else "unknown",
        "levels_completed": tracer._last_fd.levels_completed if tracer._last_fd else 0,
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
        arr = np.asarray(fd.frame)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (3,):
            # (C, H, W) — transpose to (H, W, C)
            if arr.shape[0] == 3 and arr.shape[1] > 3 and arr.shape[2] > 3:
                arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.dtype != np.uint8:
            if arr.max() < 16:
                arr = (arr * 16).clip(0, 255)
            arr = arr.astype(np.uint8)
        img = Image.fromarray(arr)
        w, h = img.size
        img = img.resize((w*4, h*4), Image.NEAREST)
        path = f"{FRAMES_DIR}/{tracer.step_n:03d}_{label}.png"
        img.save(path)
    except Exception as e:
        print(f"save_frame error: {e}")


def print_grid(game):
    e = game.ikhhdzfmarl
    grid = e.hncnfaqaddg
    print("   " + "".join(str(x) for x in range(10)))
    for y in range(15):
        s = f"{y:2d} "
        for x in range(10):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            ch = ' '
            has_N = any(n == 'fozwvlovdui' for n in names)
            has_B = any(n == 'fozwvlovdui_blue' for n in names)
            has_block = any(n == 'hupkpseyuim2' for n in names)
            has_wall = any('kraubslpehi' in n for n in names)
            has_floor = 'hupkpseyuim' in names
            if has_N: ch = 'N'
            elif has_B and has_block and has_wall: ch = '7'
            elif has_B and has_block: ch = 'b'
            elif has_B: ch = 'B'
            elif has_block and has_wall: ch = ';'
            elif has_block: ch = 'o'
            elif has_wall: ch = '#'
            elif has_floor: ch = '.'
            s += ch
        print(s)


def check_all_jumps(game):
    """Return all VALID jumps with a real piece at the source."""
    e = game.ikhhdzfmarl
    grid = e.hncnfaqaddg
    valid = []
    for y in range(15):
        for x in range(10):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            piece_type = None
            for n in names:
                if n in ('fozwvlovdui', 'fozwvlovdui_red', 'fozwvlovdui_blue'):
                    piece_type = n
                    break
            if piece_type is None:
                continue
            for d in DIRS:
                if e.qikmikecdf((x, y), d):
                    dx, dy = d
                    valid.append(((x, y, piece_type), (x+2*dx, y+2*dy), DIR_NAMES[d]))
    return valid


def do_jump(tracer, src, direction, note=""):
    e = tracer._game.ikhhdzfmarl
    off = e.hncnfaqaddg.cdpcbbnfdp
    sx, sy = src
    dx, dy = direction
    # Click source
    vp_x = sx * 6 + off[0] + 3
    vp_y = sy * 6 + off[1] + 3
    tracer.step(GameAction.ACTION6, data={'x': vp_x, 'y': vp_y}, note=f"SELECT {src} {note}")
    # Click arrow
    arrow_x = sx * 6 + off[0] + 3 + dx * 12
    arrow_y = sy * 6 + off[1] + 3 + dy * 12
    fd = tracer.step(GameAction.ACTION6, data={'x': arrow_x, 'y': arrow_y},
                     note=f"JUMP {DIR_NAMES[direction]} from {src} {note}")
    return fd


def main():
    arc = Arcade(operation_mode='offline')
    env = arc.make('lf52-271a04aa')
    env.reset()
    game = env._game
    game.set_level(9)

    tracer = TracingEnv(env)
    # per instructions: reset + set_level(9) + refresh step
    # The first env.step is the "refresh" but ACTION1 is a real push UP.
    # To get clean frame without side effects, use ACTION7 (no-op frame sync).
    fd = tracer.step(GameAction.ACTION7, note="refresh frame (no-op)")

    print("=== Initial L10 state (pristine) ===")
    print_grid(game)
    print(snapshot_state(game))
    save_frame(tracer, "pristine")

    print("\n=== Valid jumps at pristine ===")
    for j in check_all_jumps(game):
        print(f"  {j}")

    # Phase 1: Push UP x8 — moves topmost x=8 cart+blue to row 1.
    print("\n=== PHASE 1: Push UP x 8 ===")
    for i in range(8):
        tracer.step(GameAction.ACTION1, note=f"UP {i+1}/8 (bring blue+cart toward row 1)")
    save_frame(tracer, "after_phase1_up8")
    save_trace(tracer, "phase 1 complete (blue+cart at (8,1))")
    print_grid(game)

    # Phase 2: Push LEFT x5 — traverse row-1 channel.
    print("\n=== PHASE 2: Push LEFT x 5 ===")
    for i in range(5):
        tracer.step(GameAction.ACTION3, note=f"LEFT {i+1}/5 (traverse row-1 channel)")
    save_frame(tracer, "after_phase2_left5")
    save_trace(tracer, "phase 2 complete (blue+cart at (3,1))")
    print_grid(game)

    # Phase 3: Push RIGHT x1 — position blue directly below N@(4,0).
    print("\n=== PHASE 3: Push RIGHT x 1 ===")
    tracer.step(GameAction.ACTION4, note="RIGHT 1/1 (blue+cart to (4,1) — below N@(4,0))")
    save_frame(tracer, "after_phase3_right1")
    save_trace(tracer, "phase 3 complete (blue at (4,1), below N)")
    print_grid(game)

    # Diagnose
    eq = game.ikhhdzfmarl
    grid = eq.hncnfaqaddg
    print("\n=== Diagnosis ===")
    print(f"(3,0): {[o.name for o in grid.ijpoqzvnjt(3, 0)]}")
    print(f"(4,0): {[o.name for o in grid.ijpoqzvnjt(4, 0)]}")
    print(f"(5,0): {[o.name for o in grid.ijpoqzvnjt(5, 0)]}")
    print(f"(4,1): {[o.name for o in grid.ijpoqzvnjt(4, 1)]}  <- blue on cart (middle for jump)")
    print(f"(4,2): {[o.name for o in grid.ijpoqzvnjt(4, 2)]}  <- only wall — INVALID landing")
    print()
    for d in DIRS:
        v = eq.qikmikecdf((4, 0), d)
        print(f"  N@(4,0) {DIR_NAMES[d]}: qikmikecdf = {v}")

    # Attempt: try clicking N@(4,0) to see what arrows appear
    print("\n=== Attempting to select N@(4,0) ===")
    off = eq.hncnfaqaddg.cdpcbbnfdp
    vp_x = 4 * 6 + off[0] + 3
    vp_y = 0 * 6 + off[1] + 3
    tracer.step(GameAction.ACTION6, data={'x': vp_x, 'y': vp_y}, note="click N@(4,0)")

    # Final state check
    fd = tracer._last_fd
    print(f"\nFinal state: {fd.state}, levels_completed: {fd.levels_completed}")
    save_frame(tracer, "after_n_select")
    save_trace(tracer, "N@(4,0) selected but no valid jumps available — blocked at landing (4,2)")

    # ---- Attempt alternate orderings ----
    # Try a few variants after reset to search for a winning state
    print("\n=== Trying alternate push orderings ===")
    variants = [
        ("DOWN1_RIGHT1_UP8_LEFT5_RIGHT1", ['DOWN', 'RIGHT'] + ['UP']*8 + ['LEFT']*5 + ['RIGHT']),
        ("UP8_LEFT5_RIGHT1_DOWN1", ['UP']*8 + ['LEFT']*5 + ['RIGHT', 'DOWN']),
        ("UP1_DOWN1_UP7_LEFT5_RIGHT1", ['UP', 'DOWN'] + ['UP']*7 + ['LEFT']*5 + ['RIGHT']),
    ]
    for name, sequence in variants:
        arc2 = Arcade(operation_mode='offline')
        env2 = arc2.make('lf52-271a04aa')
        env2.reset()
        game2 = env2._game
        game2.set_level(9)
        env2.step(GameAction.ACTION7)  # refresh
        for d in sequence:
            env2.step(DIR_ACTIONS[d])
        eq2 = game2.ikhhdzfmarl
        grid2 = eq2.hncnfaqaddg
        valid_n_down = eq2.qikmikecdf((4, 0), (0, 1))
        c41 = [o.name for o in grid2.ijpoqzvnjt(4, 1)]
        c42 = [o.name for o in grid2.ijpoqzvnjt(4, 2)]
        print(f"  {name}: N@(4,0) DOWN valid={valid_n_down}, (4,1)={c41}, (4,2)={c42}")

    # Save the final trace file
    save_trace(tracer, "PARTIAL — see note in header for honest conclusion")
    print(f"\n=== Done. Trace saved to {TRACE_PATH} ===")
    print(f"  steps: {tracer.step_n}, state: {tracer._last_fd.state}, "
          f"levels_completed: {tracer._last_fd.levels_completed}")


if __name__ == "__main__":
    main()
