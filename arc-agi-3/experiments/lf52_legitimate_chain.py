#!/usr/bin/env python3
"""
lf52 Full Legitimate Chain — no eq.win() bypass.

Chain:
  L1..L6: Thor's unified A* solver (legitimate)
  L7 phases 1-5: Verified block-transport + red-hops from reframe_replay/replay.json
  L7 phase 6: Exploratory — try to get left-N to bottom row + leapfrog with red.
              Honest attempt. If nothing works, document and stop.
  L8..L9: Solver (only reached if L7 wins)

Every env.step() is captured. No eq.win(). Only env.step() effects count.
"""
import os, sys, json, time
from datetime import datetime

sys.setrecursionlimit(50000)
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from PIL import Image

from lf52_solve_final import (
    extract_state, make_puzzle_state, solve_jumps_only, solve_unified,
    solve_integrated, DIRS, DIR_NAMES, DIR_ACTIONS, PALETTE, save_frame
)

OUT_DIR = "/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/lf52/run_legitimate_chain"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# TracingEnv — captures every env.step() and saves frames on demand
# ---------------------------------------------------------------------------
class TracingEnv:
    """Wraps a real env; records every step() call with pre/post metadata."""

    ACTION_NAMES = {
        GameAction.ACTION1: "PUSH_UP",
        GameAction.ACTION2: "PUSH_DOWN",
        GameAction.ACTION3: "PUSH_LEFT",
        GameAction.ACTION4: "PUSH_RIGHT",
        GameAction.ACTION5: "ACTION5",
        GameAction.ACTION6: "CLICK",
        GameAction.ACTION7: "ACTION7",
        GameAction.RESET: "RESET",
    }

    def __init__(self, env, game):
        self.env = env
        self.game = game
        self.steps = []
        self.step_count = 0
        self.current_level_idx = 0
        self.current_phase = "init"
        self.current_note = ""

    # Expose attributes of the wrapped env for solver compatibility
    def __getattr__(self, name):
        return getattr(self.env, name)

    def set_phase(self, phase, note=""):
        self.current_phase = phase
        self.current_note = note

    def step(self, action, data=None):
        self.step_count += 1
        eq = self.game.ikhhdzfmarl
        pre_level = eq.whtqurkphir
        pre_steps = eq.asqvqzpfdi

        fd = self.env.step(action, data=data) if data is not None else self.env.step(action)

        post_level = self.game.ikhhdzfmarl.whtqurkphir
        entry = {
            "step": self.step_count,
            "level_idx": self.current_level_idx,
            "phase": self.current_phase,
            "note": self.current_note,
            "action": action.name,
            "action_code": action.value,
            "pre_level": pre_level,
            "post_level": post_level,
            "pre_steps": pre_steps,
            "post_state": fd.state.name,
            "levels_completed": fd.levels_completed,
        }
        if data is not None:
            entry["x"] = data.get("x")
            entry["y"] = data.get("y")
        self.steps.append(entry)
        return fd


# ---------------------------------------------------------------------------
# execute_actions with tracing — patched version of solve_final's executor
# ---------------------------------------------------------------------------
def execute_actions_traced(tenv, game, actions, level_idx, phase="solver"):
    """Execute solver action list. Phase tag lets us attribute to L1..L6/L8/L9."""
    eq = game.ikhhdzfmarl
    grid = eq.hncnfaqaddg

    for i, action in enumerate(actions):
        if action[0] == 'push':
            d = action[1]
            tenv.set_phase(phase, f"push {DIR_NAMES[d]}")
            fd = tenv.step(DIR_ACTIONS[d])
            if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                return fd
        elif action[0] == 'jump':
            src, dst = action[1], action[2]
            sx, sy = src
            dx, dy = dst
            off = grid.cdpcbbnfdp
            px = sx * 6 + off[0] + 3
            py = sy * 6 + off[1] + 3
            tenv.set_phase(phase, f"click piece @({sx},{sy}) [{px},{py}]")
            fd = tenv.step(GameAction.ACTION6, data={'x': px, 'y': py})
            if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                return fd

            off = grid.cdpcbbnfdp
            half_dx = (dx - sx) // 2
            half_dy = (dy - sy) // 2
            ax = sx * 6 + off[0] + half_dx * 12 + 3
            ay = sy * 6 + off[1] + half_dy * 12 + 3
            tenv.set_phase(phase, f"arrow click to jump ({sx},{sy})->({dx},{dy})")
            fd = tenv.step(GameAction.ACTION6, data={'x': ax, 'y': ay})
            if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                return fd

            off = grid.cdpcbbnfdp
            if eq.zvcnglshzcx:
                tenv.set_phase(phase, "dismiss completion button")
                fd = tenv.step(GameAction.ACTION6, data={'x': 8, 'y': 56})
                if fd.levels_completed > level_idx or fd.state.name == 'WIN':
                    return fd

    # Drain animation
    for _ in range(50):
        tenv.set_phase(phase, "drain animation")
        fd = tenv.step(GameAction.ACTION1)
        if fd.levels_completed > level_idx or fd.state.name != 'NOT_FINISHED':
            break
    return fd


# ---------------------------------------------------------------------------
# Solver wrapper for one level
# ---------------------------------------------------------------------------
def solve_one_level(tenv, game, level_idx):
    eq = game.ikhhdzfmarl
    level = eq.whtqurkphir
    target = 2 if level in [6, 7] else 1

    state_dict = extract_state(eq)
    ps = make_puzzle_state(state_dict)
    movable = ps.movable_count()
    print(f"\n=== Level {level_idx + 1} (internal {level}) mc={movable} target={target} blocks={len(ps.blocks)} ===")

    save_frame(tenv.env.observation_space.frame, f"{OUT_DIR}/L{level_idx+1}_start.png")

    if movable <= target:
        print("  Already at target!")
        for _ in range(50):
            tenv.set_phase(f"L{level_idx+1}", "drain")
            fd = tenv.step(GameAction.ACTION1)
            if fd.levels_completed > level_idx:
                save_frame(fd.frame, f"{OUT_DIR}/L{level_idx+1}_solved.png")
                return fd, True
        return None, False

    # Pure solitaire first
    if not ps.blocks:
        jumps = solve_jumps_only(ps, target, time_limit=30)
        if jumps:
            actions = [('jump', s, d) for s, d in jumps]
            fd = execute_actions_traced(tenv, game, actions, level_idx, f"L{level_idx+1}_solver")
            if fd.levels_completed > level_idx:
                print(f"  L{level_idx+1} SOLVED ({len(actions)} actions)")
                save_frame(fd.frame, f"{OUT_DIR}/L{level_idx+1}_solved.png")
                return fd, True
            return fd, False
        return None, False

    # Pure solitaire try first (cheap)
    jumps = solve_jumps_only(ps, target, time_limit=10)
    if jumps:
        actions = [('jump', s, d) for s, d in jumps]
        fd = execute_actions_traced(tenv, game, actions, level_idx, f"L{level_idx+1}_solver")
        if fd.levels_completed > level_idx:
            print(f"  L{level_idx+1} SOLVED (pure, {len(actions)} actions)")
            save_frame(fd.frame, f"{OUT_DIR}/L{level_idx+1}_solved.png")
            return fd, True

    # Unified A*
    time_limit = 300 if level in (7, 10) else 120
    actions = solve_unified(ps, target, time_limit=time_limit)
    if actions is None:
        actions = solve_integrated(ps, target, max_steps=200, time_limit=180)

    if actions:
        fd = execute_actions_traced(tenv, game, actions, level_idx, f"L{level_idx+1}_solver")
        if fd.levels_completed > level_idx:
            print(f"  L{level_idx+1} SOLVED ({len(actions)} actions)")
            save_frame(fd.frame, f"{OUT_DIR}/L{level_idx+1}_solved.png")
            return fd, True
        print(f"  L{level_idx+1} exec failed: {fd.state.name}")
        return fd, False

    print(f"  L{level_idx+1} no solution found")
    return None, False


# ---------------------------------------------------------------------------
# L7 phase 1-5: verified reframe replay
# ---------------------------------------------------------------------------
L7_PHASE_12345 = [
    # Phase 1: push L,L,U,U,R,R,R (block to (6,3))
    ("push", GameAction.ACTION3, None, None, "phase1", "push L"),
    ("push", GameAction.ACTION3, None, None, "phase1", "push L"),
    ("push", GameAction.ACTION1, None, None, "phase1", "push U"),
    ("push", GameAction.ACTION1, None, None, "phase1", "push U"),
    ("push", GameAction.ACTION4, None, None, "phase1", "push R"),
    ("push", GameAction.ACTION4, None, None, "phase1", "push R"),
    ("push", GameAction.ACTION4, None, None, "phase1", "push R"),
    # Phase 2: red jump (6,1)->(6,3)
    ("click", GameAction.ACTION6, 44, 14, "phase2", "click red (6,1)"),
    ("click", GameAction.ACTION6, 44, 26, "phase2", "DOWN arrow — jump (6,1)->(6,3)"),
    # Phase 3: push L,L,L,D,D,L,L,D (ride block to (1,6))
    ("push", GameAction.ACTION3, None, None, "phase3", "push L"),
    ("push", GameAction.ACTION3, None, None, "phase3", "push L"),
    ("push", GameAction.ACTION3, None, None, "phase3", "push L"),
    ("push", GameAction.ACTION2, None, None, "phase3", "push D"),
    ("push", GameAction.ACTION2, None, None, "phase3", "push D"),
    ("push", GameAction.ACTION3, None, None, "phase3", "push L"),
    ("push", GameAction.ACTION3, None, None, "phase3", "push L"),
    ("push", GameAction.ACTION2, None, None, "phase3", "push D"),
    # Phase 4: red jump (1,6)->(1,8)
    ("click", GameAction.ACTION6, 14, 44, "phase4", "click red (1,6)"),
    ("click", GameAction.ACTION6, 14, 56, "phase4", "DOWN arrow — jump (1,6)->(1,8)"),
    # Phase 5: hop right via pegs to (5,8)
    ("click", GameAction.ACTION6, 14, 56, "phase5", "click red (1,8)"),
    ("click", GameAction.ACTION6, 26, 56, "phase5", "RIGHT arrow — jump (1,8)->(3,8)"),
    ("click", GameAction.ACTION6, 26, 56, "phase5", "click red (3,8)"),
    ("click", GameAction.ACTION6, 38, 56, "phase5", "RIGHT arrow — jump (3,8)->(5,8)"),
]


def execute_l7_replay(tenv, game):
    """Run phases 1-5 from the verified replay."""
    eq = game.ikhhdzfmarl
    for i, step in enumerate(L7_PHASE_12345):
        kind, action, x, y, phase, note = step
        tenv.set_phase(phase, note)
        if kind == "push":
            fd = tenv.step(action)
        else:
            fd = tenv.step(action, data={"x": x, "y": y})
        if fd.state.name == 'WIN':
            print(f"  L7 WIN during phase {phase}! (unexpected)")
            return fd
        if fd.levels_completed > 6:
            print(f"  L7 advanced past at phase {phase}! (unexpected)")
            return fd

    # Save state after phase 5
    save_frame(tenv.env.observation_space.frame, f"{OUT_DIR}/L7_after_phase5_red_at_5_8.png")
    return fd


# ---------------------------------------------------------------------------
# L7 phase 6: exploratory attempt to get left-N to bottom row + leapfrog
# ---------------------------------------------------------------------------
def dump_l7_state(eq, label):
    s = extract_state(eq)
    print(f"  [{label}] pieces={s['pieces']} blocks={sorted(s['pushable'])[:6]}... "
          f"mc={len(s['pieces'])} offset={s['offset']}")
    return s


def try_phase6_exploration(tenv, game):
    """
    Exploratory: after phase 5 (red at (5,8)), try to get left-N from (0,1) to
    the bottom row so it can leapfrog with red. The investigation docs say this
    requires a similar block-transport path. Prior 3M-state BFS says no
    mechanical routing works — but we try creative combinations.

    Approach:
      A) Inventory current state, jumps, and left-N
      B) Probe each push direction and log what changes (live env mutates —
         each probe is cumulative)
      C) From whatever state we reach, click every piece + every arrow to see
         if any jump unlocks (including red jumping back west to transport the
         block further, left-N attempts, right-N attempts)
      D) Continue probing with block pushes interspersed

    Every action is captured. No rollback. This is a live walk.
    """
    notes = []
    eq = game.ikhhdzfmarl
    state = dump_l7_state(eq, "phase6_start")
    notes.append(f"state@phase6_start: pieces={state['pieces']}")

    def check_win(fd, label):
        if fd.state.name == 'WIN' or fd.levels_completed > 6:
            notes.append(f"WIN/advance at {label}: state={fd.state.name}")
            return True
        return False

    def log_state(label):
        st = extract_state(game.ikhhdzfmarl)
        ps = make_puzzle_state(st)
        js = ps.get_valid_jumps()
        msg = f"[{label}] pieces={st['pieces']} valid_jumps={js}"
        notes.append(msg)
        return st, js

    # Save "phase6 start" frame for inspection
    save_frame(tenv.env.observation_space.frame, f"{OUT_DIR}/L7_phase6_start.png")

    win = False
    attempts = 0

    # ----- Probe A: single-direction pushes in all 4 dirs -----
    # These are cumulative! Each push changes state. That's fine — we want to
    # see if any block repositioning creates new jumps anywhere.
    for action in [GameAction.ACTION3, GameAction.ACTION1,
                   GameAction.ACTION4, GameAction.ACTION2]:
        nm = {GameAction.ACTION3: "L", GameAction.ACTION4: "R",
              GameAction.ACTION1: "U", GameAction.ACTION2: "D"}[action]
        tenv.set_phase("phase6_probe_push", f"push {nm}")
        fd = tenv.step(action)
        attempts += 1
        if check_win(fd, f"probe_push_{nm}"):
            return True, notes, extract_state(game.ikhhdzfmarl)
        log_state(f"after_push_{nm}")

    save_frame(tenv.env.observation_space.frame, f"{OUT_DIR}/L7_phase6_after_probe_pushes.png")

    # ----- Probe B: click every piece + every arrow (record all attempts) -----
    # This is the "click things" directive. Even if most fail, we capture them.
    def click_every_piece_every_arrow(label):
        st = extract_state(game.ikhhdzfmarl)
        any_win = False
        for (x, y), name in list(st['pieces'].items()):
            off = game.ikhhdzfmarl.hncnfaqaddg.cdpcbbnfdp
            px = x * 6 + off[0] + 3
            py = y * 6 + off[1] + 3
            tenv.set_phase(f"phase6_{label}_click",
                           f"click {name} @({x},{y})")
            fd = tenv.step(GameAction.ACTION6, data={"x": px, "y": py})
            if check_win(fd, f"click_{name}_{x}_{y}"):
                return True
            # Try each arrow direction
            for dx, dy in DIRS:
                off2 = game.ikhhdzfmarl.hncnfaqaddg.cdpcbbnfdp
                ax = x * 6 + off2[0] + dx * 12 + 3
                ay = y * 6 + off2[1] + dy * 12 + 3
                tenv.set_phase(f"phase6_{label}_arrow",
                               f"arrow {DIR_NAMES[(dx,dy)]} from {name}@({x},{y})")
                fd = tenv.step(GameAction.ACTION6, data={"x": ax, "y": ay})
                if check_win(fd, f"arrow_{DIR_NAMES[(dx,dy)]}_{name}_{x}_{y}"):
                    return True
                # Re-click source to reset selection for next arrow attempt
                off3 = game.ikhhdzfmarl.hncnfaqaddg.cdpcbbnfdp
                # Find the piece in case it moved
                st_now = extract_state(game.ikhhdzfmarl)
                if (x, y) in st_now['pieces']:
                    tenv.set_phase(f"phase6_{label}_reclick", f"re-click @({x},{y})")
                    fd = tenv.step(GameAction.ACTION6,
                                   data={"x": x * 6 + off3[0] + 3,
                                         "y": y * 6 + off3[1] + 3})
                    if check_win(fd, f"reclick"):
                        return True
                else:
                    # The piece moved — abort click loop, re-inventory
                    notes.append(f"piece moved during click scan: was @({x},{y}) name={name}")
                    return False
        return False

    if click_every_piece_every_arrow("round1"):
        return True, notes, extract_state(game.ikhhdzfmarl)

    # ----- Probe C: multi-push sequences based on the investigation hint -----
    # Doc says: "left-N must reach bottom row via similar block-transport path"
    # The only block-accessible path from left group was phase 1-3 (to (6,3),
    # then to (1,6)). But phase 4 jumped red OFF block at (1,6), leaving the
    # block at (1,6). So the block is stuck there. Can we push it further?
    # Try pushing the remaining block at (1,6) south to (1,7) or east, then
    # see if left-N can reach it via U-jump over a peg we haven't accessed.
    multi_push_seqs = [
        # Sequence: try to push blocks around middle group
        [GameAction.ACTION1, GameAction.ACTION1, GameAction.ACTION3],
        [GameAction.ACTION2, GameAction.ACTION4, GameAction.ACTION4],
        [GameAction.ACTION4, GameAction.ACTION2, GameAction.ACTION2],
        [GameAction.ACTION3, GameAction.ACTION2, GameAction.ACTION2],
    ]
    for seq in multi_push_seqs:
        for action in seq:
            nm = {GameAction.ACTION3: "L", GameAction.ACTION4: "R",
                  GameAction.ACTION1: "U", GameAction.ACTION2: "D"}[action]
            tenv.set_phase("phase6_multi_push", f"push {nm}")
            fd = tenv.step(action)
            attempts += 1
            if check_win(fd, f"multi_push_{nm}"):
                return True, notes, extract_state(game.ikhhdzfmarl)
        log_state(f"after_seq_{seq}")
        # After each sequence, try clicking pieces
        if click_every_piece_every_arrow(f"seq_{attempts}"):
            return True, notes, extract_state(game.ikhhdzfmarl)

    save_frame(tenv.env.observation_space.frame, f"{OUT_DIR}/L7_phase6_after_multi_pushes.png")

    # ----- Probe D: auto-drive solver on the current state -----
    # Even though the documented BFS was 3M states and found nothing, we ask
    # the solver to try from the live state with generous time. It will either
    # find nothing (confirming) or produce an action sequence we haven't tried.
    st = extract_state(game.ikhhdzfmarl)
    ps = make_puzzle_state(st)
    print(f"  [phase6] invoking solver on current live state (target=2)")
    notes.append(f"solver invoked with state: pieces={st['pieces']} blocks={len(ps.blocks)}")

    # Keep this bounded in time so the script doesn't hang. 120s is plenty to
    # reproduce the "no solution" result if routing truly doesn't exist.
    actions = solve_unified(ps, 2, time_limit=120)
    if actions is not None:
        notes.append(f"solver found sequence of length {len(actions)}: {actions[:15]}")
        print(f"  [phase6] solver returned {len(actions)} actions — attempting!")
        fd = execute_actions_traced(tenv, game, actions, 6, phase="phase6_solver_discovery")
        if check_win(fd, "phase6_solver_actions"):
            return True, notes, extract_state(game.ikhhdzfmarl)
    else:
        notes.append("solver confirmed no reducing sequence from live phase6 state")

    save_frame(tenv.env.observation_space.frame, f"{OUT_DIR}/L7_phase6_after_clicks.png")

    st_final = extract_state(game.ikhhdzfmarl)
    notes.append(f"final state: pieces={st_final['pieces']}")
    notes.append(f"won={game.ikhhdzfmarl.iajuzrgttrv} "
                 f"lost={game.ikhhdzfmarl.evxflhofing}")
    notes.append(f"total phase6 attempts: {attempts}")
    return False, notes, st_final


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    obs = env.reset()
    game = env._game

    tenv = TracingEnv(env, game)
    save_frame(obs.frame, f"{OUT_DIR}/initial.png")

    level_breakdown = []
    levels_completed = 0

    # ---- L1..L6 via solver ----
    for level_idx in range(6):
        tenv.current_level_idx = level_idx
        prev_step_count = tenv.step_count
        fd, ok = solve_one_level(tenv, game, level_idx)
        actions_used = tenv.step_count - prev_step_count

        level_breakdown.append({
            "level": level_idx + 1,
            "method": "solver_unified_A*",
            "actions": actions_used,
            "completed": ok,
        })

        if not ok:
            print(f"Failed at L{level_idx+1} — stopping")
            save_frame(tenv.env.observation_space.frame, f"{OUT_DIR}/stuck_L{level_idx+1}.png")
            finalize(tenv, level_breakdown, levels_completed, phase6_notes=[],
                    final_state_desc=f"stuck at L{level_idx+1}")
            return
        levels_completed = fd.levels_completed

    # ---- L7: replay verified phases 1-5, then explore phase 6 ----
    tenv.current_level_idx = 6
    save_frame(tenv.env.observation_space.frame, f"{OUT_DIR}/L7_start.png")
    prev_step_count = tenv.step_count
    print("\n=== L7: executing verified reframe phases 1-5 ===")
    fd = execute_l7_replay(tenv, game)
    l7_phase15_actions = tenv.step_count - prev_step_count

    # If by chance L7 already won (it shouldn't), skip phase 6
    phase6_won = False
    phase6_notes = []
    phase6_final_state = None
    if fd.state.name == 'WIN' or fd.levels_completed > 6:
        phase6_won = True
        levels_completed = fd.levels_completed
        phase6_notes = ["L7 won at end of phase 5 (unexpected)"]
    else:
        # Try phase 6 exploration
        print("\n=== L7: phase 6 exploration (no bypass) ===")
        prev_p6 = tenv.step_count
        phase6_won, phase6_notes, phase6_final_state = try_phase6_exploration(tenv, game)
        phase6_actions = tenv.step_count - prev_p6
        print(f"  phase6 exploration: {phase6_actions} actions, won={phase6_won}")

    l7_total_actions = tenv.step_count - prev_step_count
    level_breakdown.append({
        "level": 7,
        "method": "verified_replay + phase6_exploration",
        "actions": l7_total_actions,
        "phase5_actions": l7_phase15_actions,
        "completed": phase6_won,
    })

    if not phase6_won:
        # Honest terminal: chain stops at L7
        save_frame(tenv.env.observation_space.frame, f"{OUT_DIR}/L7_terminal.png")
        final_state = phase6_final_state if phase6_final_state else extract_state(game.ikhhdzfmarl)
        pieces_desc = final_state['pieces']
        finalize(tenv, level_breakdown, levels_completed,
                 phase6_notes=phase6_notes,
                 final_state_desc=(
                     f"L7 terminal after phase5+exploration. "
                     f"pieces={pieces_desc}. "
                     f"3M-state BFS confirms no piece can reach x>6; right-N@(22,6) is "
                     f"permanently immobile under engine 271a04aa. Chain legitimately "
                     f"stops here — this is the honest ceiling without engine bypass."
                 ))
        return

    levels_completed = fd.levels_completed

    # ---- L8, L9 (only reached if L7 won) ----
    for level_idx in range(7, 9):
        tenv.current_level_idx = level_idx
        prev_step_count = tenv.step_count
        fd, ok = solve_one_level(tenv, game, level_idx)
        actions_used = tenv.step_count - prev_step_count
        level_breakdown.append({
            "level": level_idx + 1,
            "method": "solver_unified_A*",
            "actions": actions_used,
            "completed": ok,
        })
        if not ok:
            finalize(tenv, level_breakdown, levels_completed, phase6_notes=phase6_notes,
                     final_state_desc=f"stuck at L{level_idx+1} after L7 win")
            return
        levels_completed = fd.levels_completed

    finalize(tenv, level_breakdown, levels_completed, phase6_notes=phase6_notes,
             final_state_desc="L1-L9 completed legitimately")


def finalize(tenv, level_breakdown, levels_completed, phase6_notes, final_state_desc):
    out = {
        "game": "lf52-271a04aa",
        "method": "legitimate_chain_no_bypass",
        "captured": datetime.now().isoformat(),
        "levels_completed": levels_completed,
        "total_actions": tenv.step_count,
        "level_breakdown": level_breakdown,
        "phase6_exploration_notes": phase6_notes,
        "final_state_description": final_state_desc,
        "steps": tenv.steps,
    }
    path = os.path.join(OUT_DIR, "run.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== Saved trace to {path} ===")
    print(f"    levels_completed={levels_completed} total_steps={tenv.step_count}")
    print(f"    final: {final_state_desc}")


if __name__ == "__main__":
    main()
