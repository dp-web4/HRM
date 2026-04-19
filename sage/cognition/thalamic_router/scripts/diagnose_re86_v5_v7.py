"""Side-by-side re86 diagnostic: v5 vs v7 action-head output on the same frames.

Loads both adapters, replays re86's solver trace (same frames as training),
and compares the action_head probability distribution at each step. A
meaningful dilution should show up as:
  - v7 giving higher probability to actions not in v5's top-K
  - v7 higher entropy (less confident) than v5
  - v7 ranking the solver's known-good action lower

Output: summary table + per-step traces for the first N steps.

Run: PYTHONPATH=<SAGE_ROOT> python3 -m sage.cognition.thalamic_router.scripts.diagnose_re86_v5_v7
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sage.cognition.thalamic_router.frame_router import (
    load_frame_router, build_scalar_vector,
    onehot_frame, N_ACTIONS, ACTION_NAMES, RECENT_ACTIONS_K,
    FRAME_H, FRAME_W, N_COLORS,
)
from sage.cognition.thalamic_router.gameplay_capture import (
    load_trace, _discover_trace,
)

V5 = Path("/mnt/c/exe/projects/ai-agents/private-context/training-data/router/_adapters/cbp-framerouter-v5nodyn-2026-04-18")
V7 = Path("/mnt/c/exe/projects/ai-agents/private-context/training-data/router/_adapters/cbp-framerouter-v7sub-2026-04-19")
GAME = "re86"
GAME_ID = "re86-4e57566e"   # from ARC-SAGE coord; the v7 JSON confirmed re86 in slug list


def run_adapter(adapter_path, trace, max_steps=30):
    """Replay the first max_steps of the trace through the adapter, return
    per-step action_probs + solver's known-good action."""
    model, cfg = load_frame_router(adapter_path)
    model.eval()

    game_idx = cfg.game_slugs.index(GAME) if GAME in cfg.game_slugs else 0
    mech_vec = None
    if cfg.use_mech_embedding:
        mech_vec = model.mech_embedding_table[game_idx].detach().cpu().numpy().tolist()

    results = []
    from arc_agi import Arcade
    arc = Arcade(operation_mode="offline")
    env = arc.make(trace.game_id) or arc.make(trace.game)
    fd = env.reset()
    zero_frame = np.zeros((N_COLORS, FRAME_H, FRAME_W), dtype=np.float32)
    prev_oh = zero_frame
    last_action = 0
    recent = deque([0] * RECENT_ACTIONS_K, maxlen=RECENT_ACTIONS_K)

    try:
        from arcengine import GameAction
        int_to_action = {ga.value: ga for ga in GameAction}
    except Exception:
        int_to_action = {}

    total = max(1, len(trace.steps))
    for i, step in enumerate(trace.steps[:max_steps]):
        curr_raw = np.asarray(getattr(fd, "frame", None))
        curr_oh = onehot_frame(curr_raw) if curr_raw is not None else zero_frame
        level = step.level if step.level is not None else 0
        step_frac = min(1.0, step.index / total)
        budget = max(0.0, 1.0 - step.index / total)

        scalar = build_scalar_vector(
            game_idx=game_idx, n_games=cfg.n_games,
            level=level, n_levels=cfg.n_levels,
            step_frac=step_frac, budget_remaining=budget,
            last_action=last_action, recent_actions=list(recent),
            available_actions=[1] * N_ACTIONS, batch_state=[0, 0, 0],
            include_last_action=cfg.include_last_action,
            mech_embedding=mech_vec,
        )

        with torch.no_grad():
            prev_t = torch.from_numpy(prev_oh).unsqueeze(0)
            curr_t = torch.from_numpy(curr_oh).unsqueeze(0)
            sc_t = torch.from_numpy(np.array(scalar, dtype=np.float32)).unsqueeze(0)
            out = model(prev_t, curr_t, sc_t)
            probs = F.softmax(out["action_logits"], dim=-1).squeeze(0).numpy()
            invoke = torch.sigmoid(out["invoke_logit"]).item()

        results.append({
            "step": step.index,
            "solver_action": int(step.action),
            "solver_name": ACTION_NAMES[int(step.action)],
            "probs": probs,
            "invoke": invoke,
            "top_action": int(np.argmax(probs)),
            "solver_rank": int((probs > probs[int(step.action)]).sum()),
        })

        ga = int_to_action.get(int(step.action))
        try:
            if ga is not None:
                fd = env.step(ga, data=step.data) if step.data else env.step(ga)
        except Exception:
            break
        recent.append(int(step.action))
        last_action = int(step.action)
        prev_oh = curr_oh

    return results, cfg


def compare():
    trace_path = _discover_trace(GAME, GAME_ID)
    if not trace_path or not trace_path.exists():
        print(f"ERROR: no trace for {GAME}")
        sys.exit(1)
    trace = load_trace(trace_path, GAME, GAME_ID)
    print(f"Loaded {GAME} trace: {len(trace.steps)} steps")

    print("\n=== V5 adapter ===")
    v5_results, v5_cfg = run_adapter(V5, trace, max_steps=30)
    print(f"  use_mech_embedding: {v5_cfg.use_mech_embedding}  n_games: {v5_cfg.n_games}  arch_v: {v5_cfg.architecture_version}")

    print("\n=== V7 adapter ===")
    v7_results, v7_cfg = run_adapter(V7, trace, max_steps=30)
    print(f"  use_mech_embedding: {v7_cfg.use_mech_embedding}  n_games: {v7_cfg.n_games}  arch_v: {v7_cfg.architecture_version}")

    # Comparison table
    print(f"\n{'step':>4} {'solver':>6}  {'v5 top':>6}(prob)  {'v7 top':>6}(prob)  v5 rank/v7 rank  v5 inv/v7 inv")
    v5_agree = v7_agree = v5_rank_sum = v7_rank_sum = 0
    v5_inv_sum = v7_inv_sum = 0.0
    n = min(len(v5_results), len(v7_results))
    for i in range(n):
        r5, r7 = v5_results[i], v7_results[i]
        solver = r5["solver_name"]
        v5_top = ACTION_NAMES[r5["top_action"]]
        v7_top = ACTION_NAMES[r7["top_action"]]
        v5_p = r5["probs"][r5["solver_action"]]
        v7_p = r7["probs"][r7["solver_action"]]
        print(f"{r5['step']:>4} {solver:>6}  "
              f"{v5_top:>4}({v5_p:.2f})  {v7_top:>4}({v7_p:.2f})  "
              f"{r5['solver_rank']:>4}/{r7['solver_rank']:<4}  "
              f"{r5['invoke']:.2f}/{r7['invoke']:.2f}")
        v5_agree += r5["top_action"] == r5["solver_action"]
        v7_agree += r7["top_action"] == r7["solver_action"]
        v5_rank_sum += r5["solver_rank"]
        v7_rank_sum += r7["solver_rank"]
        v5_inv_sum += r5["invoke"]
        v7_inv_sum += r7["invoke"]

    print(f"\n=== Summary over {n} steps ===")
    print(f"v5 top-action == solver: {v5_agree}/{n} ({v5_agree/n*100:.1f}%)")
    print(f"v7 top-action == solver: {v7_agree}/{n} ({v7_agree/n*100:.1f}%)")
    print(f"v5 mean solver rank: {v5_rank_sum/n:.2f}")
    print(f"v7 mean solver rank: {v7_rank_sum/n:.2f}")
    print(f"v5 mean invoke prob: {v5_inv_sum/n:.3f}")
    print(f"v7 mean invoke prob: {v7_inv_sum/n:.3f}")

    # Distribution shift: mean entropy over steps
    v5_ent = np.mean([-np.sum(r["probs"] * np.log(r["probs"] + 1e-9)) for r in v5_results[:n]])
    v7_ent = np.mean([-np.sum(r["probs"] * np.log(r["probs"] + 1e-9)) for r in v7_results[:n]])
    print(f"v5 mean action entropy: {v5_ent:.3f}  (max {np.log(N_ACTIONS):.3f})")
    print(f"v7 mean action entropy: {v7_ent:.3f}")

    # Per-action marginal probability — does v7 shift probability mass to specific actions?
    print(f"\n=== Mean action-prob across {n} steps ===")
    print(f"{'action':>6}  {'v5':>6}  {'v7':>6}  {'Δ':>7}")
    v5_mean = np.mean([r["probs"] for r in v5_results[:n]], axis=0)
    v7_mean = np.mean([r["probs"] for r in v7_results[:n]], axis=0)
    for i, name in enumerate(ACTION_NAMES):
        d = v7_mean[i] - v5_mean[i]
        marker = "  ←↑" if d > 0.05 else ("  ←↓" if d < -0.05 else "")
        print(f"{name:>6}  {v5_mean[i]:.3f}  {v7_mean[i]:.3f}  {d:+.3f}{marker}")


if __name__ == "__main__":
    compare()
