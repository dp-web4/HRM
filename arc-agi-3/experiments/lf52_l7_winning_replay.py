#!/usr/bin/env python3
"""
L7 reframe replay — session continuation.

Task framing: yarflam's "blue ball + empty cart" strategy for L10 applies to L7
with red piece as transport/stepping-stone equivalent.

STATUS AFTER THIS SESSION (2026-04-17):
  - Phases 1-5 verified on live engine (red transported from (6,1) to (5,8))
  - Phase 6+: NOT FOUND. Extensive live-engine walk + wall-topology analysis
    could not identify a continuation that reaches a winning N-over-N.

KEY FINDINGS (new this session):
  1. Wall graph has 3 disconnected components:
     - Comp 0 (size 22): right side incl (22,3)(22,4)(20,5)(22,2) — where N@(22,6)
     - Comp 1 (size 21): middle incl (8,5-7)(9,3-5)(10,3)(10,5)(10,7)(10,8)(11,1-6)
       (12,1)(12,3)(13,1)(14,1)(14,2)
     - Comp 2 (size 16): left incl (0,3)(1,3-6,3)(1,5-6,5)(3,4)(1,6)(6,6)
     Blocks stay within their comp. Comp 0 has peg-blocks (18,3)(19,2)(20,3);
     comp 1 has only (14,2) plain block; comp 2 has only (5,5) plain block.

  2. ENGINE SCROLL TRIGGER at (8,8): source line 5493-5502 — landing any piece
     at (8,8) with initial offset (5,5) scrolls view (-44, 0). Not exploited
     here (cell unreachable via jumps — row 8 has no pegs at (6,8)/(7,8)/(9,8)).

  3. N@(0,1) is initially TRAPPED (peg (0,2) below, landing (0,3) is wall-only).
     But pushing block (5,5) via L,L,U,U,L,L,L places block at (0,3) creating
     valid landing (wall+block). N jumps DOWN onto block — N now ride-able.

  4. N-on-block transport confined to comp 2 walls (max reach: col 6). To reach
     (22,5) the N needs to jump between components. No such jump bridge found.

  5. The only N-over-N reduction with valid landing is:
     N_A@(22,5), N_B@(22,6), jumper direction: either N@(22,6) UP over N_A to
     (22,4)=wall+block valid landing. REQUIRES N at (22,5).

  6. (22,5) reachable ONLY via jump from (22,3) DOWN over (22,4) peg-middle.
     But (22,4) can't simultaneously be peg-middle (wall+peg+block = len 3, not
     valid landing for the SUBSEQUENT N@(22,6) → (22,4) final jump).

  Confirms prior agents' "structurally unsolvable" finding.

VERIFIED PHASES (executable):
  Phase 1: push L,L,U,U,R,R,R — moves block (5,5) to (6,3) [landing pad]
  Phase 2: jump red (6,1) -> (6,3) [onto block]
  Phase 3: push L,L,L,D,D,L,L,D — transports red-on-block to (1,6)
  Phase 4: jump red (1,6) -> (1,8) [dismount via peg (1,7)]
  Phase 5: jumps red (1,8) -> (3,8) -> (5,8) [bottom row pegs]
  Phase 6+: NO VALID CONTINUATION FOUND.

Actions used in trace:
  ACTION1/2/3/4 = push UP/DOWN/LEFT/RIGHT
  ACTION6 with data={'x','y'} = click
  Coords: vp_x = world_x*6 + offset[0] + 3
          vp_y = world_y*6 + offset[1] + 3
  Arrow click: + (dx_half*12, dy_half*12) from piece center
"""
import os
import sys
import json
from pathlib import Path

os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.setrecursionlimit(50000)
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from PIL import Image

from arc_agi import Arcade
from arcengine import GameAction
from lf52_solve_final import solve_level, PALETTE, extract_state

OUT_DIR = Path("/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/lf52/run_L7_winning")
FRAMES_DIR = OUT_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
RUN_JSON = OUT_DIR / "run.json"

DIR_ACTIONS = {'U': GameAction.ACTION1, 'D': GameAction.ACTION2,
               'L': GameAction.ACTION3, 'R': GameAction.ACTION4}

trace_steps = []
frame_counter = [0]


def save_frame(env, label):
    try:
        arr = np.array(env.observation_space.frame[0])
        h, w = arr.shape
        s = 6
        img = Image.new('RGB', (w*s, h*s))
        pix = img.load()
        for y in range(h):
            for x in range(w):
                c = PALETTE.get(int(arr[y, x]), (0, 0, 0))
                for dy in range(s):
                    for dx in range(s):
                        pix[x*s+dx, y*s+dy] = c
        fname = f"{frame_counter[0]:03d}_{label}.png"
        img.save(FRAMES_DIR / fname)
        frame_counter[0] += 1
    except Exception as e:
        print(f"  save_frame error: {e}")


def write_trace(status_note="in_progress"):
    data = {
        "game_id": "lf52-271a04aa",
        "player": "cbp-l7-reframe-replay",
        "target_level": 7,
        "status": status_note,
        "reframe": "red piece is transport/stepping-stone (like blue in L10)",
        "verified_phases": [
            "P1: L,L,U,U,R,R,R",
            "P2: jump (6,1) -> (6,3)",
            "P3: L,L,L,D,D,L,L,D",
            "P4: jump (1,6) -> (1,8)",
            "P5a: jump (1,8) -> (3,8)",
            "P5b: jump (3,8) -> (5,8)",
        ],
        "steps": trace_steps,
        "step_count": len(trace_steps),
    }
    RUN_JSON.write_text(json.dumps(data, indent=2))


def do_push(env, dir_char, label=""):
    fd = env.step(DIR_ACTIONS[dir_char])
    trace_steps.append({
        "type": "push", "dir": dir_char,
        "state": fd.state.name,
        "levels_completed": fd.levels_completed,
        "label": label,
    })
    write_trace()
    return fd


def do_jump(env, game, src, dst, label=""):
    eq = game.ikhhdzfmarl
    grid = eq.hncnfaqaddg
    sx, sy = src
    dx_, dy_ = dst

    # Verify the jump is valid via engine
    dir_vec = ((dx_ - sx) // 2, (dy_ - sy) // 2)
    assert eq.qikmikecdf(src, dir_vec), f"Jump {src} -> {dst} not valid per engine"

    off = grid.cdpcbbnfdp
    sxp = sx*6 + off[0] + 3
    syp = sy*6 + off[1] + 3
    fd = env.step(GameAction.ACTION6, data={'x': sxp, 'y': syp})
    trace_steps.append({
        "type": "click", "vp_x": sxp, "vp_y": syp,
        "sub": "select", "src": list(src), "dst": list(dst),
        "state": fd.state.name,
        "levels_completed": fd.levels_completed,
        "label": f"{label}: select {src}",
    })

    off = grid.cdpcbbnfdp
    half_dx = (dx_ - sx) // 2
    half_dy = (dy_ - sy) // 2
    axp = sx*6 + off[0] + half_dx*12 + 3
    ayp = sy*6 + off[1] + half_dy*12 + 3
    fd = env.step(GameAction.ACTION6, data={'x': axp, 'y': ayp})
    trace_steps.append({
        "type": "click", "vp_x": axp, "vp_y": ayp,
        "sub": "arrow", "src": list(src), "dst": list(dst),
        "state": fd.state.name,
        "levels_completed": fd.levels_completed,
        "label": f"{label}: arrow -> {dst}",
    })
    write_trace()
    return fd


def render_grid(eq, label):
    grid = eq.hncnfaqaddg
    print(f"\n--- {label} | offset={grid.cdpcbbnfdp} steps={eq.asqvqzpfdi} ---")
    for y in range(12):
        row = f"{y:3d} "
        for x in range(30):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            if not names:
                row += " "
            elif any(n == 'fozwvlovdui_red' for n in names):
                row += "R"
            elif any(n == 'fozwvlovdui_blue' for n in names):
                row += "B"
            elif any(n == 'fozwvlovdui' for n in names):
                row += "N"
            elif any(n == 'hupkpseyuim2' for n in names):
                if any('kraubslpehi' in nn for nn in names):
                    row += ";"
                else:
                    row += "b"
            elif any('dgxfozncuiz' in n for n in names):
                row += "o"
            elif any('kraubslpehi' in n for n in names):
                row += "#"
            elif any(n == 'hupkpseyuim' for n in names):
                row += "."
            else:
                row += "?"
        print(row)


def main():
    arc = Arcade(operation_mode='offline')
    env = arc.make('lf52-271a04aa')
    fd = env.reset()
    game = env._game

    print("=== Solving L1-L6 via solve_level ===")
    for lvl in range(6):
        fd = solve_level(env, game, lvl)
        if fd is None or fd.levels_completed <= lvl:
            print(f"FAIL at L{lvl+1}")
            write_trace("setup_failed")
            return 1

    eq = game.ikhhdzfmarl
    save_frame(env, "00_L7_start")
    render_grid(eq, "L7 initial")
    print(f"\n=== Beginning L7 (steps={eq.asqvqzpfdi}) ===")

    # Phase 1: L,L,U,U,R,R,R
    print("\n--- Phase 1: L,L,U,U,R,R,R ---")
    for c in "LLUURRR":
        do_push(env, c, f"P1: push {c}")
    save_frame(env, "01_after_phase1")
    render_grid(eq, "after Phase 1")

    # Phase 2: red (6,1) -> (6,3)
    print("\n--- Phase 2: red (6,1) -> (6,3) ---")
    do_jump(env, game, (6, 1), (6, 3), "P2: red jump")
    save_frame(env, "02_after_phase2")
    render_grid(eq, "after Phase 2")

    # Phase 3: L,L,L,D,D,L,L,D
    print("\n--- Phase 3: L,L,L,D,D,L,L,D ---")
    for c in "LLLDDLLD":
        do_push(env, c, f"P3: push {c}")
    save_frame(env, "03_after_phase3")
    render_grid(eq, "after Phase 3")

    # Phase 4: red (1,6) -> (1,8)
    print("\n--- Phase 4: red (1,6) -> (1,8) ---")
    do_jump(env, game, (1, 6), (1, 8), "P4: red down")
    save_frame(env, "04_after_phase4")
    render_grid(eq, "after Phase 4")

    # Phase 5: red (1,8) -> (3,8) -> (5,8)
    print("\n--- Phase 5: red (1,8) -> (3,8) -> (5,8) ---")
    do_jump(env, game, (1, 8), (3, 8), "P5a")
    do_jump(env, game, (3, 8), (5, 8), "P5b")
    save_frame(env, "05_after_phase5")
    render_grid(eq, "after Phase 5 (red @ (5,8))")

    # Summary of state
    state = extract_state(eq)
    print(f"\n=== State after Phase 5 ===")
    print(f"  Pieces: {state['pieces']}")
    print(f"  Blocks: {sorted(state['pushable'])}")
    print(f"  Step count: {eq.asqvqzpfdi}")
    print(f"  Levels completed: {fd.levels_completed}")
    print(f"  State: {fd.state.name}")

    if fd.levels_completed > 6 or fd.state.name == 'WIN':
        print("\n*** L7 WON! ***")
        save_frame(env, "99_win")
        write_trace("won")
        return 0

    print("\n*** L7 NOT WON — Phase 6+ not found in this session ***")
    print("    See module docstring for findings.")
    save_frame(env, "99_stuck_post_phase5")
    write_trace("stuck_post_phase5")
    return 2


if __name__ == "__main__":
    sys.exit(main())
